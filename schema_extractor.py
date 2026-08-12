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

conn = psycopg.connect(
    host=HOST,
    port=PORT,
    dbname=DATABASE,
    user=USER,
    password=PASSWORD
)

cur = conn.cursor()

# ============================================================
# GET TABLES AND COLUMNS
# ============================================================

query = """
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
"""

cur.execute(query)

rows = cur.fetchall()

# ============================================================
# ORGANIZE SCHEMA
# ============================================================

schema = {}

for table_name, column_name, data_type in rows:

    if table_name not in schema:
        schema[table_name] = []

    schema[table_name].append({
        "column": column_name,
        "type": data_type
    })

# ============================================================
# PRINT SCHEMA
# ============================================================

print("\n================ DATABASE SCHEMA ================\n")

for table_name, columns in schema.items():

    print(f"TABLE: {table_name}")

    for column in columns:
        print(
            f"    {column['column']} "
            f"({column['type']})"
        )

    print()

# ============================================================
# GET PRIMARY KEYS
# ============================================================

print("\n================ PRIMARY KEYS ================\n")

primary_key_query = """
SELECT
    tc.table_name,
    kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY'
AND tc.table_schema = 'public'
ORDER BY tc.table_name;
"""

cur.execute(primary_key_query)

primary_keys = cur.fetchall()

for table, column in primary_keys:
    print(f"{table}.{column}")

# ============================================================
# GET FOREIGN KEYS
# ============================================================

print("\n================ FOREIGN KEYS ================\n")

foreign_key_query = """
SELECT
    tc.table_name AS source_table,
    kcu.column_name AS source_column,
    ccu.table_name AS target_table,
    ccu.column_name AS target_column
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_schema = 'public'
ORDER BY tc.table_name;
"""

cur.execute(foreign_key_query)

foreign_keys = cur.fetchall()

for source_table, source_column, target_table, target_column in foreign_keys:

    print(
        f"{source_table}.{source_column}"
        f"  -->  "
        f"{target_table}.{target_column}"
    )

# ============================================================
# CREATE AI-READY SCHEMA JSON
# ============================================================

schema_data = {
    "tables": {}
}

for table_name, columns in schema.items():

    schema_data["tables"][table_name] = {
        "columns": columns,
        "primary_keys": [],
        "foreign_keys": []
    }

# Add primary keys
for table, column in primary_keys:

    if table in schema_data["tables"]:
        schema_data["tables"][table]["primary_keys"].append(column)

# Add foreign keys
for source_table, source_column, target_table, target_column in foreign_keys:

    if source_table in schema_data["tables"]:
        schema_data["tables"][source_table]["foreign_keys"].append({
            "column": source_column,
            "target_table": target_table,
            "target_column": target_column
        })

# Save JSON
with open("schema.json", "w", encoding="utf-8") as f:
    json.dump(schema_data, f, indent=4)

print("\nAI-ready schema saved to schema.json")

# ============================================================
# CLOSE CONNECTION
# ============================================================

cur.close()
conn.close()

print("\nSchema extraction completed successfully!")