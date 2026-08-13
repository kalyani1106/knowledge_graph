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

print("\n================ ROW DATA → AGE ================\n")

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
# DISCOVER ALL TABLES
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
# DISCOVER PRIMARY KEYS
# ============================================================

primary_keys = {}

cur.execute("""
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
""")

for table_name, column_name in cur.fetchall():
    primary_keys[table_name] = column_name


print("\nPrimary keys detected:")

for table, column in primary_keys.items():
    print(f"  {table}.{column}")


# ============================================================
# CREATE A UNIQUE LABEL FOR EACH TABLE
# ============================================================

def safe_label(table_name):

    # AGE labels cannot contain arbitrary characters.
    # Replace unsupported characters with underscore.

    return "".join(
        c if c.isalnum() or c == "_" else "_"
        for c in table_name
    )


# ============================================================
# CREATE ROW VERTICES
# ============================================================

print("\n================ CREATING ROW VERTICES ================\n")

row_counts = {}

for table_name in tables:

    # --------------------------------------------------------
    # Get columns dynamically
    # --------------------------------------------------------

    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position;
    """, (table_name,))

    columns = [
        row[0]
        for row in cur.fetchall()
    ]

    if not columns:
        continue


    # --------------------------------------------------------
    # Read rows dynamically
    # --------------------------------------------------------

    column_sql = ", ".join(
        '"' + column.replace('"', '""') + '"'
        for column in columns
    )

    table_sql = '"' + table_name.replace('"', '""') + '"'

    cur.execute(
        f"SELECT {column_sql} FROM {table_sql}"
    )

    rows = cur.fetchall()


    # --------------------------------------------------------
    # Create AGE vertices
    # --------------------------------------------------------

    count = 0

    for row in rows:

        properties = {}

        for column, value in zip(columns, row):

            if value is not None:

                # AGE properties are stored as strings
                # for this generic prototype.

                properties[column] = str(value)

        # Add source table information.

        properties["_table"] = table_name


        # ----------------------------------------------------
        # Create Cypher property string
        # ----------------------------------------------------

        cypher_properties = []

        for key, value in properties.items():

            safe_key = key.replace("'", "''")
            safe_value = value.replace("'", "''")

            cypher_properties.append(
                f"{safe_key}: '{safe_value}'"
            )

        property_string = ", ".join(
            cypher_properties
        )


        # ----------------------------------------------------
        # Create vertex
        # ----------------------------------------------------

        query = f"""
        SELECT *
        FROM ag_catalog.cypher(
            '{GRAPH_NAME}',
            $$
            CREATE (
                n:{safe_label(table_name)}
                {{{property_string}}}
            )
            RETURN n
            $$
        ) AS result(result ag_catalog.agtype);
        """

        try:

            cur.execute(query)
            cur.fetchall()

            count += 1

        except Exception as e:

            conn.rollback()

            print(
                f"Failed row in table {table_name}: {e}"
            )


    row_counts[table_name] = count

    print(
        f"{table_name}: {count} row vertices created"
    )


# ============================================================
# CREATE ROW-LEVEL RELATIONSHIPS
# ============================================================

print(
    "\n================ CREATING ROW RELATIONSHIPS ================\n"
)

relationship_count = 0


for relationship in relationships:

    from_table = relationship["from_table"]
    from_column = relationship["from_column"]

    to_table = relationship["to_table"]
    to_column = relationship["to_column"]


    # --------------------------------------------------------
    # Only process relationships for existing tables
    # --------------------------------------------------------

    if from_table not in tables:
        continue

    if to_table not in tables:
        continue


    safe_from_table = from_table.replace("'", "''")
    safe_to_table = to_table.replace("'", "''")
    safe_from_column = from_column.replace("'", "''")
    safe_to_column = to_column.replace("'", "''")


    from_label = safe_label(from_table)
    to_label = safe_label(to_table)


    # --------------------------------------------------------
    # Match rows using the AI-discovered relationship
    # --------------------------------------------------------

    query = f"""
    SELECT *
    FROM ag_catalog.cypher(
        '{GRAPH_NAME}',
        $$
        MATCH (a:{from_label}),
              (b:{to_label})

        WHERE a.{safe_from_column}
              = b.{safe_to_column}

        CREATE (a)-[:DATA_RELATION {{
            from_column: '{safe_from_column}',
            to_column: '{safe_to_column}'
        }}]->(b)

        RETURN a
        $$
    ) AS result(result ag_catalog.agtype);
    """


    try:

        cur.execute(query)

        results = cur.fetchall()

        relationship_count += len(results)

        print(
            f"{from_table}.{from_column}"
            f" → "
            f"{to_table}.{to_column}"
        )

        print(
            f"  Relationships created: {len(results)}"
        )

    except Exception as e:

        conn.rollback()

        print(
            f"Relationship failed: "
            f"{from_table} → {to_table}"
        )

        print(f"  Error: {e}")


# ============================================================
# COMMIT
# ============================================================

conn.commit()


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n================ COMPLETED ================\n"
)

print(
    f"Tables processed       : {len(tables)}"
)

print(
    f"AI relationships loaded: {len(relationships)}"
)

print(
    "Row vertices created:"
)

for table, count in row_counts.items():

    print(
        f"  {table}: {count}"
    )

print(
    f"\nRow relationships created: {relationship_count}"
)

print(
    f"AGE graph: {GRAPH_NAME}"
)


cur.close()
conn.close()