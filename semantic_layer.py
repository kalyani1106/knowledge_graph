import json
import requests
import re

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

SCHEMA_FILE = "schema.json"
RELATIONSHIP_FILE = "validated_relationships.json"
OUTPUT_FILE = "semantic_catalog.json"

BATCH_SIZE = 20


# ============================================================
# LOAD FILES
# ============================================================

with open(SCHEMA_FILE, "r", encoding="utf-8") as file:
    schema = json.load(file)

with open(RELATIONSHIP_FILE, "r", encoding="utf-8") as file:
    relationships = json.load(file)


# ============================================================
# PREPARE SCHEMA
# ============================================================

tables = schema.get("tables", {})

schema_summary = []

for table_name, table_info in tables.items():

    columns = []

    for column in table_info.get("columns", []):

        columns.append({
            "name": column["column"],
            "type": column["type"]
        })

    schema_summary.append({
        "table": table_name,
        "columns": columns,
        "primary_keys": table_info.get("primary_keys", [])
    })


# ============================================================
# CALL LOCAL AI
# ============================================================

def analyze_batch(batch):

    prompt = f"""
You are building a semantic layer for a large database.

Analyze the following database schema.

For EVERY table, determine:

1. Business concept
2. Short description
3. Business meaning of each column

Do not invent tables.
Do not invent columns.

Return ONLY JSON in this exact format:

{{
    "tables": [
        {{
            "table": "table_name",
            "business_concept": "business concept",
            "description": "description",
            "columns": [
                {{
                    "column": "column_name",
                    "meaning": "business meaning"
                }}
            ]
        }}
    ]
}}

DATABASE SCHEMA:

{json.dumps(batch, indent=2)}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]


# ============================================================
# PARSE AI RESPONSE
# ============================================================

def parse_ai_response(text):

    text = text.strip()

    # Remove Markdown code fences if present
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    text = text.strip()

    try:
        data = json.loads(text)

    except json.JSONDecodeError:

        # Try to find JSON object inside response
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "No valid JSON object found in AI response."
            )

        data = json.loads(
            text[start:end + 1]
        )

    # Expected format:
    # {"tables": [...]}

    if isinstance(data, dict):

        if "tables" in data:
            return data["tables"]

    # In case model returns a list directly
    if isinstance(data, list):
        return data

    raise ValueError(
        "Unexpected AI response format."
    )


# ============================================================
# PROCESS BATCHES
# ============================================================

semantic_catalog = []

total_tables = len(schema_summary)

total_batches = (
    (total_tables + BATCH_SIZE - 1)
    // BATCH_SIZE
)

print("\n================ SEMANTIC LAYER ================\n")

print(
    f"Tables detected: {total_tables}"
)

print(
    f"Processing in batches of {BATCH_SIZE} tables...\n"
)


for start in range(
    0,
    total_tables,
    BATCH_SIZE
):

    batch = schema_summary[
        start:start + BATCH_SIZE
    ]

    batch_number = (
        start // BATCH_SIZE
    ) + 1

    print(
        f"Processing batch "
        f"{batch_number}/{total_batches}..."
    )

    try:

        ai_response = analyze_batch(batch)

        parsed_tables = parse_ai_response(
            ai_response
        )

        semantic_catalog.extend(
            parsed_tables
        )

        print(
            f"  AI analyzed "
            f"{len(parsed_tables)} tables."
        )

    except Exception as e:

        print(
            f"  Batch failed: {e}"
        )


# ============================================================
# SAVE SEMANTIC CATALOG
# ============================================================

output = {
    "tables": semantic_catalog,
    "relationships": relationships
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output,
        file,
        indent=4
    )


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n================ COMPLETED ================\n"
)

print(
    f"Tables detected : {total_tables}"
)

print(
    f"Tables analyzed : {len(semantic_catalog)}"
)

print(
    f"Relationships   : {len(relationships)}"
)

print(
    f"Saved           : {OUTPUT_FILE}"
)