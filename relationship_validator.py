import json

# ============================================================
# LOAD SCHEMA
# ============================================================

with open("schema.json", "r") as file:
    schema = json.load(file)

tables = schema["tables"]

# ============================================================
# LOAD AI RELATIONSHIPS
# ============================================================

# We will save the AI output manually in this file for now.
# Later, we will automate this step.

ai_relationships = [
    {
        "from_table": "orders",
        "from_column": "customer_id",
        "to_table": "customers",
        "to_column": "customer_id",
        "relationship": "ONE-TO-MANY",
        "confidence": 0.99
    },
    {
        "from_table": "orders",
        "from_column": "product_id",
        "to_table": "products",
        "to_column": "product_id",
        "relationship": "ONE-TO-MANY",
        "confidence": 0.99
    },
    {
        "from_table": "orders",
        "from_column": "category",
        "to_table": "categories",
        "to_column": "category_name",
        "relationship": "MANY-TO-FEW",
        "confidence": 0.95
    }
]

# ============================================================
# VALIDATION FUNCTION
# ============================================================

def column_exists(table_name, column_name):

    if table_name not in tables:
        return False

    columns = tables[table_name]["columns"]

    for column in columns:
        if column["column"] == column_name:
            return True

    return False


# ============================================================
# VALIDATE AI RELATIONSHIPS
# ============================================================

valid_relationships = []
rejected_relationships = []

for relation in ai_relationships:

    from_table = relation["from_table"]
    from_column = relation["from_column"]

    to_table = relation["to_table"]
    to_column = relation["to_column"]

    # Check source table
    if from_table not in tables:
        relation["reason"] = "Source table does not exist"
        rejected_relationships.append(relation)
        continue

    # Check target table
    if to_table not in tables:
        relation["reason"] = "Target table does not exist"
        rejected_relationships.append(relation)
        continue

    # Check source column
    if not column_exists(from_table, from_column):
        relation["reason"] = (
            f"Column {from_column} does not exist in {from_table}"
        )
        rejected_relationships.append(relation)
        continue

    # Check target column
    if not column_exists(to_table, to_column):
        relation["reason"] = (
            f"Column {to_column} does not exist in {to_table}"
        )
        rejected_relationships.append(relation)
        continue

    # If everything exists
    valid_relationships.append(relation)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n================ VALID RELATIONSHIPS ================\n")

for relation in valid_relationships:

    print(
        f"{relation['from_table']}.{relation['from_column']}"
        f"  -->  "
        f"{relation['to_table']}.{relation['to_column']}"
    )

    print(
        f"Relationship : {relation['relationship']}"
    )

    print(
        f"Confidence   : {relation['confidence']}"
    )

    print()


print("\n================ REJECTED RELATIONSHIPS ================\n")

for relation in rejected_relationships:

    print(
        f"{relation['from_table']}.{relation['from_column']}"
        f"  -->  "
        f"{relation['to_table']}.{relation['to_column']}"
    )

    print(
        f"Reason : {relation['reason']}"
    )

    print()


# ============================================================
# SAVE VALID RELATIONSHIPS
# ============================================================

with open("validated_relationships.json", "w") as file:

    json.dump(
        valid_relationships,
        file,
        indent=4
    )

print("Validation completed successfully.")
print("Saved: validated_relationships.json")