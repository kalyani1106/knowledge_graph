import json
import psycopg

# ============================================================
# PostgreSQL CONNECTION
# ============================================================

HOST = "localhost"
PORT = 5455
DATABASE = "knowledge_graph_demo"
USER = "postgres"
PASSWORD = "postgres"

GRAPH_NAME = "ai_knowledge_graph"

# ============================================================
# CONNECT
# ============================================================

conn = psycopg.connect(
    host=HOST,
    port=PORT,
    dbname=DATABASE,
    user=USER,
    password=PASSWORD
)

cur = conn.cursor()

print("\n================ AI → AGE ================\n")

# ============================================================
# ENABLE AGE
# ============================================================

cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
cur.execute("LOAD 'age';")
cur.execute('SET search_path = ag_catalog, "$user", public;')

# ============================================================
# LOAD VALIDATED AI RELATIONSHIPS
# ============================================================

with open("validated_relationships.json", "r") as file:
    relationships = json.load(file)

print(f"Loaded {len(relationships)} validated relationships.")

# ============================================================
# CREATE TABLE VERTICES
# ============================================================

tables = set()

for relation in relationships:
    tables.add(relation["from_table"])
    tables.add(relation["to_table"])

print("\nTables detected:")

for table in sorted(tables):
    print(" -", table)

# ============================================================
# CREATE GRAPH VERTICES
# ============================================================

for table in tables:

    cypher_query = f"""
    CREATE (:Table {{name: '{table}'}})
    """

    query = f"""
    SELECT *
    FROM ag_catalog.cypher(
        '{GRAPH_NAME}',
        $$
        {cypher_query}
        $$
    ) AS (result agtype);
    """

    cur.execute(query)

conn.commit()

print("\nTable vertices created successfully.")

# ============================================================
# CREATE AI RELATIONSHIPS
# ============================================================

for relation in relationships:

    from_table = relation["from_table"]
    from_column = relation["from_column"]

    to_table = relation["to_table"]
    to_column = relation["to_column"]

    confidence = relation["confidence"]

    cypher_query = f"""
    MATCH (a:Table {{name: '{from_table}'}})
    MATCH (b:Table {{name: '{to_table}'}})
    CREATE (a)-[:RELATED_TO {{
        from_column: '{from_column}',
        to_column: '{to_column}',
        confidence: {confidence}
    }}]->(b)
    """

    query = f"""
    SELECT *
    FROM ag_catalog.cypher(
        '{GRAPH_NAME}',
        $$
        {cypher_query}
        $$
    ) AS (result agtype);
    """

    cur.execute(query)

conn.commit()

print("AI relationships created successfully.")

# ============================================================
# VERIFY GRAPH
# ============================================================

query = f"""
SELECT *
FROM ag_catalog.cypher(
    '{GRAPH_NAME}',
    $$
    MATCH (a:Table)-[r:RELATED_TO]->(b:Table)
    RETURN a, r, b
    $$
) AS (
    source agtype,
    relationship agtype,
    target agtype
);
"""

cur.execute(query)

rows = cur.fetchall()

print("\n================ CREATED GRAPH ================\n")

for row in rows:
    print(row)

# ============================================================
# CLOSE
# ============================================================

cur.close()
conn.close()

print("\nAI → Apache AGE process completed successfully.")