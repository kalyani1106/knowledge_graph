import json
import requests
import re
import psycopg

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

SEMANTIC_FILE = "semantic_catalog.json"

HOST = "localhost"
PORT = 5455
DATABASE = "knowledge_graph_demo"
USER = "postgres"
PASSWORD = "******"

GRAPH_NAME = "ai_data_graph"


# ============================================================
# LOAD SEMANTIC CATALOG
# ============================================================

with open(SEMANTIC_FILE, "r", encoding="utf-8") as file:
    catalog = json.load(file)

tables = catalog.get("tables", [])
relationships = catalog.get("relationships", [])

# Dynamic lookup indices
valid_table_map = {t["table"]: t for t in tables}
valid_rel_pairs = {
    (rel["from_table"], rel["to_table"]): rel
    for rel in relationships
}

# Dynamic helper: find human-readable descriptive column for any table
def find_descriptive_name_column(table_name):
    if table_name not in valid_table_map:
        return None
    cols = valid_table_map[table_name].get("columns", [])
    for c in cols:
        cn = c["column"].lower()
        if cn.endswith("_name") or cn == "name" or cn in ("title", "label", "description", "city", "category"):
            return c["column"]
    for c in cols:
        cm = c.get("meaning", "").lower()
        if any(k in cm for k in ["name", "description", "title", "city", "category"]):
            return c["column"]
    for c in cols:
        if not c["column"].lower().endswith("_id"):
            return c["column"]
    return cols[0]["column"] if cols else None

# Dynamic helper: find numeric metric column for any table
def find_numeric_metric_column(table_name):
    if table_name not in valid_table_map:
        return None
    cols = valid_table_map[table_name].get("columns", [])
    for c in cols:
        cn = c["column"].lower()
        cm = c.get("meaning", "").lower()
        if any(k in cn or k in cm for k in ["amount", "revenue", "price", "cost", "total", "value", "quantity", "score"]):
            return c["column"]
    return None


# ============================================================
# DYNAMIC SEMANTIC RETRIEVAL STAGE (Scales to 1000+ Tables)
# ============================================================

def find_relevant_context(question, max_tables=5):
    """
    Dynamically select relevant tables and relationships for questions
    scaling up to 1000+ tables.
    Includes 1-hop relationship expansion so filtering entities are retrieved.
    """
    question_lower = question.lower()
    question_words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", question_lower))

    scored_tables = []

    for table in tables:
        table_name = table.get("table", "")
        business_concept = table.get("business_concept", "")
        description = table.get("description", "")

        score = 0
        searchable_text = f"{table_name} {business_concept} {description}".lower()

        for word in question_words:
            if len(word) < 2:
                continue
            if word in table_name.lower():
                score += 5
            elif word in searchable_text:
                score += 2

        for column in table.get("columns", []):
            column_name = column.get("column", "")
            meaning = column.get("meaning", "")
            column_text = f"{column_name} {meaning}".lower()

            for word in question_words:
                if len(word) < 2:
                    continue
                if word in column_name.lower():
                    score += 3
                elif word in column_text:
                    score += 1

        scored_tables.append((score, table))

    scored_tables.sort(key=lambda item: item[0], reverse=True)

    selected_tables = [
        table for score, table in scored_tables[:max_tables]
        if score > 0
    ]

    if not selected_tables:
        selected_tables = tables[:max_tables]

    selected_table_names = {table["table"] for table in selected_tables}

    # 1-hop relationship expansion to bring connected filtering tables
    expanded_table_names = set(selected_table_names)
    selected_relationships = []

    for relationship in relationships:
        from_table = relationship.get("from_table")
        to_table = relationship.get("to_table")

        if from_table in selected_table_names or to_table in selected_table_names:
            selected_relationships.append(relationship)
            expanded_table_names.add(from_table)
            expanded_table_names.add(to_table)

    all_table_map = {table["table"]: table for table in tables}
    final_tables = [
        all_table_map[t_name]
        for t_name in expanded_table_names
        if t_name in all_table_map
    ]

    return {
        "tables": final_tables,
        "relationships": selected_relationships
    }


# ============================================================
# CONNECT TO APACHE AGE
# ============================================================

conn = psycopg.connect(
    host=HOST,
    port=PORT,
    dbname=DATABASE,
    user=USER,
    password=PASSWORD
)

cur = conn.cursor()

cur.execute("LOAD 'age';")
cur.execute(
    'SET search_path = ag_catalog, "$user", public;'
)


# ============================================================
# LLM SEMANTIC REASONING STAGE (STATELESS / INDEPENDENT QUERIES)
# ============================================================

def generate_age_query(question):

    relevant_context = find_relevant_context(question)

    prompt = f"""You are an expert Semantic Layer and Apache AGE Cypher Query Generator.

The semantic catalog represents an arbitrary relational database with row vertices and directed [:DATA_RELATION] edges.
Use ONLY the supplied semantic context below. Each user question must be answered independently without assuming any context from previous queries.

============================================================
RELEVANT SEMANTIC CONTEXT
============================================================

{json.dumps(relevant_context, indent=2)}

============================================================
USER QUESTION
============================================================

{question}

============================================================
CRITICAL CYPHER RULES
============================================================

1. READ-ONLY APACHE AGE CYPHER ONLY:
   - Generate exactly ONE valid Apache AGE Cypher query starting with MATCH.
   - Return raw Cypher string only. NO markdown, NO ``` blocks, NO explanatory text, NO semicolons.
   - NEVER generate SQL syntax (NO SELECT, NO GROUP BY, NO HAVING, NO IN (MATCH...), NO SQL subqueries).

2. STRICT RELATIONSHIP DIRECTION & GRAPH TRAVERSAL:
   - [:DATA_RELATION] edges are ALWAYS strictly directed from `from_table` to `to_table` as defined in the catalog.
     Example: catalog specifies from_table = "orders" and to_table = "customers".
     The edge MUST be `(orders)-[:DATA_RELATION]->(customers)`.
   - NEVER reverse edge arrows.
   - NEVER use property joins like `WHERE o.customer_id = c.customer_id`.
   - NEVER use variable length relationships `[:DATA_RELATION*]`.

3. STRICT PREDICATE SCOPING (NO UNREQUESTED FILTERS):
   - Add WHERE filters ONLY for values, entities, names, cities, or products explicitly requested in the CURRENT question.
   - If the current question asks for an overall metric (e.g. "highest order amount", "total revenue", "average order amount") WITHOUT specifying an entity name or city, DO NOT add a WHERE clause.
   - NEVER add filters for names, values, or entities not mentioned in the CURRENT question.

4. MULTIPLE ENTITY FILTERING:
   - When filtering across multiple entities (e.g. customers from a city AND a product):
     MATCH (o:orders)-[:DATA_RELATION]->(c:customers), (o:orders)-[:DATA_RELATION]->(p:products)
     WHERE toLower(c.city) = 'visakhapatnam' AND toLower(p.product_name) = 'laptop'
     RETURN sum(toFloat(o.order_amount))
   - DO NOT use IN (MATCH...) subqueries! Use comma-separated MATCH patterns.

5. INTENT EXAMPLES & RETURN CLAUSES:
   - "show me all orders" -> MATCH (o:orders) RETURN o
   - "show me customers who placed orders" -> MATCH (o:orders)-[:DATA_RELATION]->(c:customers) RETURN c
   - "highest order amount" -> MATCH (o:orders) RETURN max(toFloat(o.order_amount))
   - "highest order amount for [Name]" -> MATCH (o:orders)-[:DATA_RELATION]->(c:customers) WHERE toLower(c.customer_name) = '[name]' RETURN max(toFloat(o.order_amount))
   - "total revenue" -> MATCH (o:orders) RETURN sum(toFloat(o.order_amount))
   - "average order amount" -> MATCH (o:orders) RETURN avg(toFloat(o.order_amount))
   - Always cast numeric string properties to float: `toFloat(...)`. Never place aggregate functions like max() inside WHERE clauses.

Cypher Query:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# ============================================================
# DETERMINISTIC QUERY REPAIR & NORMALIZATION ENGINE
# ============================================================

def fix_table_labels(query):
    """
    Ensure vertex labels match exact table names in semantic_catalog.json.
    Fixes singular/plural mismatches (e.g. :order -> :orders).
    """
    def repl_label(match):
        label = match.group(1)
        if label in valid_table_map:
            return f":{label}"
        if label + "s" in valid_table_map:
            return f":{label}s"
        if label.endswith("s") and label[:-1] in valid_table_map:
            return f":{label[:-1]}"
        return f":{label}"

    return re.sub(r":\s*(\w+)\b", repl_label, query)


def fix_mismatched_properties(query):
    """
    Detect properties used on wrong tables (e.g., o.product_name or {product_name: "Laptop"} on orders)
    and dynamically rewrite them into relationship traversals to the entity table possessing that column.
    """
    match = re.search(r"\((\w+):(\w+)\s*\{(\w+):\s*['\"]([^'\"]+)['\"]\Action*", query)
    if match:
        var, table, prop, val = match.groups()
        if table in valid_table_map:
            cols = {c["column"] for c in valid_table_map[table].get("columns", [])}
            if prop not in cols:
                for rel in relationships:
                    if rel["from_table"] == table:
                        target_tbl = rel["to_table"]
                        target_cols = {c["column"] for c in valid_table_map[target_tbl].get("columns", [])}
                        if prop in target_cols:
                            t_var = target_tbl[0]
                            return f"MATCH ({var}:{table})-[:DATA_RELATION]->({t_var}:{target_tbl}) WHERE toLower({t_var}.{prop}) = '{val.lower()}' RETURN {var}"
    return query


def fix_invalid_edge_traversals(query):
    """
    If LLM generates an edge (t1)-[:DATA_RELATION]->(t2) that does not exist in catalog,
    check if t1 possesses a direct column matching the filter attribute on t2.
    If so, collapse the un-cataloged edge traversal into a direct property filter.
    """
    edge_match = re.search(
        r"\((\w+):(\w+)\)\s*-\s*\[[^\]]*\]\s*->\s*\((\w+):(\w+)\)\s*WHERE\s+toLower\((\w+)\.(\w+)\)\s*=\s*'([^']+)'",
        query,
        re.IGNORECASE
    )
    if edge_match:
        v1, t1, v2, t2, f_var, f_col, val = edge_match.groups()
        if (t1, t2) not in valid_rel_pairs and (t2, t1) not in valid_rel_pairs:
            if t1 in valid_table_map:
                t1_cols = {c["column"] for c in valid_table_map[t1].get("columns", [])}
                direct_col = None
                for col in t1_cols:
                    if col in f_col or f_col.replace("_name", "") in col or col.replace("_name", "") in f_col:
                        direct_col = col
                        break
                if direct_col:
                    ret_m = re.search(r"RETURN\s+(.*)$", query, re.IGNORECASE)
                    ret_clause = ret_m.group(0) if ret_m else f"RETURN {v1}"
                    return f"MATCH ({v1}:{t1}) WHERE toLower({v1}.{direct_col}) = '{val.lower()}' {ret_clause}"
    return query


def fix_rel_direction(query):
    """
    Fix relationship direction to follow exact from_table -> to_table in semantic catalog.
    Replaces variable-length [:DATA_RELATION*] with single-step [:DATA_RELATION].
    """
    query = re.sub(r"\[\s*:\s*DATA_RELATION\*\d*.*\]", "[:DATA_RELATION]", query, flags=re.IGNORECASE)

    def replace_rel(match):
        v1, t1, edge, v2, t2 = match.groups()
        if (t1, t2) not in valid_rel_pairs and (t2, t1) in valid_rel_pairs:
            return f"({v2}:{t2})-[{edge}]->({v1}:{t1})"
        return match.group(0)

    rel_pattern = re.compile(
        r"\((\w+):(\w+)\)\s*-\s*\[([^\]]*)\]\s*->\s*\((\w+):(\w+)\)"
    )
    return rel_pattern.sub(replace_rel, query)


def resolve_human_readable_entities(query, question=""):
    """
    Dynamically resolve human-readable text values matched against ID columns.
    If an AI output puts a text value or ID into a foreign key column,
    this function uses semantic_catalog.json to traverse to the correct target
    table and filter against the human-readable descriptive column.
    """
    match = re.search(r"\((\w+):(\w+)\s*\{(\w+):\s*'([^']+)'\}\)", query)
    if match:
        var, table, col, val = match.groups()
        if col.endswith("_id"):
            target_table = None
            for rel in relationships:
                if rel["from_table"] == table and rel["from_column"] == col:
                    target_table = rel["to_table"]
                    break

            if target_table and target_table in valid_table_map:
                name_col = find_descriptive_name_column(target_table)
                if name_col:
                    target_var = target_table[0]
                    search_val = val
                    q_match = re.search(r"(?:named|by|for|in|category|customer|product)\s+([A-Za-z0-9_]+)", question, re.IGNORECASE)
                    if q_match:
                        search_val = q_match.group(1)

                    return f"MATCH ({var}:{table})-[:DATA_RELATION]->({target_var}:{target_table}) WHERE toLower({target_var}.{name_col}) = '{search_val.lower()}' RETURN {var}"

    match_where = re.search(r"WHERE\s+(\w+)\.(\w+)\s*=\s*'([^']+)'", query, re.IGNORECASE)
    if match_where:
        var, col, val = match_where.groups()
        if col.endswith("_id"):
            node_m = re.search(rf"\({var}:(\w+)\)", query)
            if node_m:
                table = node_m.group(1)
                target_table = None
                for rel in relationships:
                    if rel["from_table"] == table and rel["from_column"] == col:
                        target_table = rel["to_table"]
                        break
                if target_table and target_table in valid_table_map:
                    name_col = find_descriptive_name_column(target_table)
                    if name_col:
                        target_var = target_table[0]
                        ret_m = re.search(r"RETURN\s+(\w+)", query, re.IGNORECASE)
                        ret_var = ret_m.group(1) if ret_m else var
                        return f"MATCH ({var}:{table})-[:DATA_RELATION]->({target_var}:{target_table}) WHERE toLower({target_var}.{name_col}) = '{val.lower()}' RETURN {ret_var}"

    return query


def fix_return_clause(query, question=""):
    """
    Ensure the RETURN clause contains a single, clean expression matching the user's intent.
    Prevents RETURN c, o, p multi-column returns.
    """
    ret_match = re.search(r"RETURN\s+([A-Za-z0-9_,\s]+)$", query, re.IGNORECASE)
    if ret_match:
        exprs = [e.strip() for e in ret_match.group(1).split(",") if e.strip()]
        if len(exprs) > 1:
            q_lower = question.lower()
            best_target = exprs[-1]
            for var in exprs:
                node_m = re.search(rf"\({var}:(\w+)\)", query)
                if node_m:
                    tbl = node_m.group(1)
                    if tbl in q_lower or tbl[:-1] in q_lower:
                        best_target = var
                        break
            query = re.sub(r"RETURN\s+([A-Za-z0-9_,\s]+)$", f"RETURN {best_target}", query, flags=re.IGNORECASE)

    return query


def check_and_clean_untraced_predicates(query, question):
    """
    DETERMINISTIC SEMANTIC SANITY CHECK & CLEANUP:
    Extract literal values from WHERE predicates (e.g. 'kalyani', 'hyderabad', 'laptop').
    Verify whether each literal value can be semantically traced to words/stems in the current question.
    If a predicate literal value is NOT in the question (e.g. 'kalyani' when asking 'highest order amount'):
    - Strip that unmentioned predicate from the WHERE clause.
    - If no predicates remain in WHERE, strip the WHERE clause completely.
    - If a joined node table in MATCH (e.g. c:customers) was only used for the stripped predicate,
      simplify the MATCH pattern to remove the unneeded join.
    """
    if not question:
        return query

    question_lower = question.lower()
    where_match = re.search(r"\bWHERE\b\s+(.*?)(?=\bRETURN\b|\bORDER\b|\bLIMIT\b|$)", query, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return query

    where_clause = where_match.group(1).strip()
    literals = re.findall(r"['\"]([^'\"]+)['\"]", where_clause)

    untraced_literals = []
    words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", question_lower))

    for lit in literals:
        lit_lower = lit.lower()
        found = False
        for w in words:
            if (lit_lower in w or w in lit_lower or
                (lit_lower.endswith('s') and lit_lower[:-1] == w) or
                (w.endswith('s') and w[:-1] == lit_lower)):
                found = True
                break
        if not found:
            untraced_literals.append(lit)

    if not untraced_literals:
        return query

    # Filter out conditions containing untraced literals
    conditions = re.split(r"\s+\bAND\b\s+|\s+\bOR\b\s+", where_clause, flags=re.IGNORECASE)
    valid_conditions = []
    for cond in conditions:
        cond_has_untraced = any(lit.lower() in cond.lower() for lit in untraced_literals)
        if not cond_has_untraced:
            valid_conditions.append(cond.strip())

    ret_match = re.search(r"(\bRETURN\b.*)$", query, re.IGNORECASE | re.DOTALL)
    ret_clause = ret_match.group(1) if ret_match else ""

    if not valid_conditions:
        # WHERE clause is completely stripped. Simplify MATCH pattern if join was introduced solely for filter.
        match_part = query[:where_match.start()].strip()
        ret_var_match = re.search(r"RETURN\s+(?:max|min|avg|sum|count)?\(?\s*(?:toFloat|toInt)?\(?\s*(\w+)\.", ret_clause, re.IGNORECASE)
        if not ret_var_match:
            ret_var_match = re.search(r"RETURN\s+(\w+)\b", ret_clause, re.IGNORECASE)

        if ret_var_match:
            r_var = ret_var_match.group(1)
            node_m = re.search(rf"\({r_var}:(\w+)\)", match_part)
            if node_m:
                r_table = node_m.group(1)
                match_part = f"MATCH ({r_var}:{r_table})"

        cleaned_query = f"{match_part} {ret_clause}"
    else:
        new_where = " AND ".join(valid_conditions)
        match_part = query[:where_match.start()].strip()
        cleaned_query = f"{match_part} WHERE {new_where} {ret_clause}"

    return cleaned_query


# ============================================================
# PARSE AND NORMALIZE AI RESPONSE (NOISE & SQL STRIPPING)
# ============================================================

def parse_ai_response(text, question=""):

    text = text.strip()

    # Remove markdown code fences if added
    text = re.sub(r"```cypher", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Prepend MATCH if missing but RETURN is present
    if not re.search(r"\bMATCH\b", text, flags=re.IGNORECASE):
        if re.search(r"\bRETURN\b", text, flags=re.IGNORECASE):
            text = "MATCH " + text

    # Locate MATCH keyword if preamble text exists
    match_position = re.search(r"\bMATCH\b", text, flags=re.IGNORECASE)

    if not match_position:
        raise ValueError(
            "AI did not generate a Cypher MATCH query.\n\n" + text
        )

    text = text[match_position.start():].strip()

    # Strip any trailing explanatory text after RETURN clause
    ret_pos = re.search(r"\bRETURN\b", text, flags=re.IGNORECASE)
    if ret_pos:
        post_ret = text[ret_pos.start():]
        lines = post_ret.split("\n")
        first_statement_line = lines[0]
        text = text[:ret_pos.start()] + first_statement_line

    # Remove trailing semicolon
    text = text.rstrip(";").strip()

    # Apply dynamic post-processing driven strictly by semantic catalog
    text = fix_table_labels(text)
    text = fix_mismatched_properties(text)
    text = fix_invalid_edge_traversals(text)
    text = resolve_human_readable_entities(text, question)
    text = fix_rel_direction(text)
    text = fix_return_clause(text, question)
    text = check_and_clean_untraced_predicates(text, question)

    return {
        "query": text
    }


# ============================================================
# DETERMINISTIC CATALOG VALIDATION STAGE
# ============================================================

def validate_query(query, question=""):
    query_lower = query.lower()

    # 1. Read-only check: Block dangerous keywords and SQL syntax
    forbidden = [
        "select", "group by", "having", "delete", "detach", "drop",
        "truncate", "update", "insert", "create", "alter", "set",
        "remove", "merge", "load"
    ]
    for word in forbidden:
        if re.search(rf"\b{word}\b", query_lower):
            return False, f"Forbidden operation or SQL syntax detected: '{word.upper()}'"

    # 2. Reject SQL subqueries IN (MATCH...)
    if re.search(r"\bIN\s*\(\s*MATCH", query, re.IGNORECASE):
        return False, "SQL subquery syntax 'IN (MATCH...)' is not valid Cypher."

    # 3. Structure check
    if not re.search(r"\bmatch\b", query_lower):
        return False, "Generated query does not contain MATCH."
    if not re.search(r"\breturn\b", query_lower):
        return False, "Generated query does not contain RETURN."

    # 4. Reject variable length relationships
    if re.search(r"\[:\s*DATA_RELATION\*\d*.*\]", query, re.IGNORECASE):
        return False, "Variable-length relationships ([:DATA_RELATION*]) are not allowed."

    # 5. Reject multiple statements
    if len(query.strip().split(";")) > 1 and query.strip().split(";")[1].strip():
        return False, "Multiple statements are not allowed."

    # 6. Validate table labels against semantic catalog
    var_to_label = {}
    node_matches = re.findall(r"\b(\w+)\s*:\s*(\w+)\b", query)
    for var, label in node_matches:
        if label == "DATA_RELATION":
            continue
        if label not in valid_table_map:
            return False, f"Unknown table label referenced: '{label}'"
        var_to_label[var] = label

    # 7. Validate relationship edge types
    edge_labels = re.findall(r"\[(?:\w+)?\s*:\s*(\w+)\s*\]", query)
    for rel_label in edge_labels:
        if rel_label != "DATA_RELATION":
            return False, f"Unknown relationship type: '{rel_label}'"

    # 8. Validate relationship directions strictly against semantic catalog
    edge_matches = re.findall(r"\((\w+):(\w+)\)\s*-\s*\[[^\]]*\]\s*->\s*\((\w+):(\w+)\)", query)
    for v1, t1, v2, t2 in edge_matches:
        if (t1, t2) not in valid_rel_pairs:
            if (t2, t1) in valid_rel_pairs:
                return False, f"Reversed relationship direction detected: ({t1})->({t2}). Catalog specifies ({t2})->({t1})."
            else:
                return False, f"Invalid relationship edge: ({t1})->({t2}) does not exist in semantic catalog."

    rev_edge_matches = re.findall(r"\((\w+):(\w+)\)\s*<-\s*\[[^\]]*\]\s*-\s*\((\w+):(\w+)\)", query)
    for v1, t1, v2, t2 in rev_edge_matches:
        if (t2, t1) not in valid_rel_pairs:
            return False, f"Invalid relationship edge: ({t2})->({t1}) does not exist in semantic catalog."

    # 9. Validate property references against corresponding table columns
    prop_matches = re.findall(r"\b(\w+)\.(\w+)\b", query)
    for var, prop in prop_matches:
        if var in var_to_label:
            label = var_to_label[var]
            table_info = valid_table_map[label]
            valid_cols = {c["column"] for c in table_info.get("columns", [])}
            valid_cols.add("_table")
            if prop not in valid_cols:
                return False, f"Unknown column '{prop}' referenced for table '{label}'"

    # 10. Semantic Sanity Check: Ensure no untraced literal predicate values exist
    if question:
        where_match = re.search(r"\bWHERE\b\s+(.*?)(?=\bRETURN\b|\bORDER\b|\bLIMIT\b|$)", query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            literals = re.findall(r"['\"]([^'\"]+)['\"]", where_clause)
            question_lower = question.lower()
            words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", question_lower))
            for lit in literals:
                lit_lower = lit.lower()
                found = False
                for w in words:
                    if (lit_lower in w or w in lit_lower or
                        (lit_lower.endswith('s') and lit_lower[:-1] == w) or
                        (w.endswith('s') and w[:-1] == lit_lower)):
                        found = True
                        break
                if not found:
                    return False, f"Semantic Sanity Check Failure: Predicate value '{lit}' in query was not mentioned in the user question."

    return True, "Query passed deterministic catalog validation."


# ============================================================
# GENERIC RESULT FORMATTER
# ============================================================

def display_results(rows):
    print("\n================ RESULTS ================\n")

    if not rows:
        print("No results found.")
        return

    results = []
    seen = set()

    for row in rows:

        value = row[0]
        text = str(value)

        # Handle scalar results
        if "::vertex" not in text:
            results.append({"result": text})
            continue

        # Extract JSON from AGE vertex
        json_part = text.replace("::vertex", "").strip()

        try:
            data = json.loads(json_part)
            properties = data.get("properties", {})

            if not properties:
                continue

            properties_key = json.dumps(properties, sort_keys=True)
            if properties_key in seen:
                continue

            seen.add(properties_key)

            clean_properties = {}
            for key, val in properties.items():
                clean_properties[key.strip()] = val

            results.append(clean_properties)

        except Exception:
            results.append({"result": text})

    if not results:
        print("No results found.")
        return

    columns = []
    for result in results:
        for key in result.keys():
            if key not in columns:
                columns.append(key)

    display_columns = {}
    for column in columns:
        display_columns[column] = column.replace("_", " ").title()

    widths = {}
    for column in columns:
        header_length = len(display_columns[column])
        value_lengths = [
            len(str(result.get(column, "")))
            for result in results
        ]
        widths[column] = max(header_length, max(value_lengths, default=0))

    header = " | ".join(
        display_columns[column].ljust(widths[column])
        for column in columns
    )
    separator = "-+-".join("-" * widths[column] for column in columns)

    print(header)
    print(separator)

    for result in results:
        row_values = [
            str(result.get(column, "")).ljust(widths[column])
            for column in columns
        ]
        print(" | ".join(row_values))


# ============================================================
# EXECUTE AGE QUERY
# ============================================================

def execute_age_query(cypher_query):

    safe_graph_name = GRAPH_NAME.replace('"', '""')

    age_sql = """
    SELECT *
    FROM ag_catalog.cypher(
        '""" + safe_graph_name + """',
        $$
        """ + cypher_query + """
        $$
    ) AS result(
        result ag_catalog.agtype
    );
    """

    cur.execute(age_sql)
    return cur.fetchall()


# ============================================================
# MAIN INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":
    print("\n================ SEMANTIC → AGE ================\n")
    print(f"Semantic tables available : {len(tables)}")
    print(f"Relationships available   : {len(relationships)}")
    print("\nThe system is ready.")
    print("No database tables are hardcoded in this program.\n")

    while True:
        try:
            question = input("Ask a question (type 'exit' to stop): ")
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower().strip() == "exit":
            break

        if not question.strip():
            continue

        print("\nAnalyzing question with AI...")

        try:
            ai_response = generate_age_query(question)
            result = parse_ai_response(ai_response, question)
            cypher_query = result.get("query")

            if not cypher_query:
                print("AI did not generate a query.")
                continue

            print("\n================ GENERATED AGE QUERY ================\n")
            print(cypher_query)

            valid, message = validate_query(cypher_query, question)

            print("\n================ VALIDATION ================\n")
            print(message)

            if not valid:
                print("\nQuery was NOT executed.")
                continue

            print("\nExecuting query in Apache AGE...")
            rows = execute_age_query(cypher_query)
            display_results(rows)

        except Exception as e:
            conn.rollback()
            print("\nError:")
            print(e)

    cur.close()
    conn.close()
    print("\nSemantic → AGE process completed.")