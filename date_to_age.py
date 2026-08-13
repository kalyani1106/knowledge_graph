import json
import psycopg

# ============================================================
# CONFIGURATION
# ============================================================

HOST = "localhost"
PORT = 5455
DATABASE = "knowledge_graph_demo"
USER = "postgres"
PASSWORD = "******"

GRAPH_NAME = "ai_data_graph"

RELATIONSHIP_FILE = "validated_relationships.json"


# ============================================================
# LOAD AI-VALIDATED RELATIONSHIPS
# ============================================================

with open(RELATIONSHIP_FILE, "r", encoding="utf-8") as f:
    relationships = json.load(f)

print("\n================ DATA → AGE ================\n")

print(
    f"AI relationships loaded: {len(relationships)}"
)

# ============================================================
# CONNECT TO POSTGRESQL
# ============================================================

conn = psycopg.connect(
    host=HOST,
    port=PORT,
    dbname=DATABASE,
    user=USER,
    password=PASSWORD
)

cur = conn.cursor()


# ============================================================
# LOAD AGE
# ============================================================

cur.execute("LOAD 'age';")

cur.execute(
    'SET search_path = ag_catalog, "$user", public;'
)


# ============================================================
# DISCOVER ALL DATABASE TABLES
# ============================================================

cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_type = 'BASE TABLE'
    ORDER BY table_name;
""")

tables = [
    row[0]
    for row in cur.fetchall()
]

print(f"\nTables detected: {len(tables)}")

for table in tables:
    print(f"  - {table}")


# ============================================================
# CREATE AGE GRAPH
# ============================================================

cur.execute("""
    SELECT 1
    FROM ag_catalog.ag_graph
    WHERE name = %s;
""", (GRAPH_NAME,))

if cur.fetchone() is None:

    cur.execute(
        "SELECT ag_catalog.create_graph(%s);",
        (GRAPH_NAME,)
    )

    print(f"\nGraph created: {GRAPH_NAME}")

else:

    print(f"\nGraph already exists: {GRAPH_NAME}")


# ============================================================
# CREATE TABLE VERTICES
# ============================================================

print("\nCreating table vertices...")

for table_name in tables:

    safe_table = table_name.replace("'", "''")

    query = f"""
    SELECT *
    FROM ag_catalog.cypher(
        '{GRAPH_NAME}',
        $$
        CREATE (t:Table {{name: '{safe_table}'}})
        RETURN t
        $$
    ) AS result(result ag_catalog.agtype);
    """

    try:

        cur.execute(query)
        cur.fetchall()

    except Exception:

        conn.rollback()


print("Table vertices created successfully.")


# ============================================================
# CREATE AI RELATIONSHIPS
# ============================================================

print("\nCreating AI-detected relationships...")

created_relationships = 0

for relationship in relationships:

    from_table = relationship["from_table"]
    from_column = relationship["from_column"]

    to_table = relationship["to_table"]
    to_column = relationship["to_column"]

    confidence = relationship.get(
        "confidence",
        0
    )

    safe_from_table = from_table.replace("'", "''")
    safe_to_table = to_table.replace("'", "''")
    safe_from_column = from_column.replace("'", "''")
    safe_to_column = to_column.replace("'", "''")

    query = f"""
    SELECT *
    FROM ag_catalog.cypher(
        '{GRAPH_NAME}',
        $$
        MATCH (a:Table), (b:Table)
        WHERE a.name = '{safe_from_table}'
          AND b.name = '{safe_to_table}'

        CREATE (a)-[:RELATED_TO {{
            from_column: '{safe_from_column}',
            to_column: '{safe_to_column}',
            confidence: {confidence}
        }}]->(b)

        RETURN a
        $$
    ) AS result(result ag_catalog.agtype);
    """

    try:

        cur.execute(query)
        cur.fetchall()

        created_relationships += 1

        print(
            f"  {from_table}.{from_column}"
            f" → "
            f"{to_table}.{to_column}"
        )

    except Exception as e:

        conn.rollback()

        print(
            f"  Failed: "
            f"{from_table} → {to_table}"
        )

        print(f"  Error: {e}")


# ============================================================
# COMMIT
# ============================================================

conn.commit()


print("\n================ COMPLETED ================\n")

print(
    f"Tables processed       : {len(tables)}"
)

print(
    f"AI relationships loaded: {len(relationships)}"
)

print(
    f"Relationships created  : {created_relationships}"
)

print(
    f"AGE graph              : {GRAPH_NAME}"
)


cur.close()
conn.close()