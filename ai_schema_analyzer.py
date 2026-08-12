import json
import requests

# ============================================================
# LOAD DATABASE SCHEMA
# ============================================================

with open("schema.json", "r") as file:
    schema = json.load(file)

# ============================================================
# CREATE PROMPT FOR AI
# ============================================================

prompt = f"""
You are a database schema analysis expert.

Analyze the following PostgreSQL database schema.

Your task is to identify possible relationships between tables
by examining:

1. Primary keys
2. Foreign keys
3. Column names
4. Matching column names
5. Data types
6. Naming patterns

Do NOT invent relationships randomly.

Return ONLY valid JSON in this format:

{{
    "relationships": [
        {{
            "from_table": "table1",
            "from_column": "column1",
            "to_table": "table2",
            "to_column": "column2",
            "relationship": "LIKELY_RELATIONSHIP",
            "confidence": 0.95,
            "reason": "short explanation"
        }}
    ]
}}

DATABASE SCHEMA:

{json.dumps(schema, indent=2)}
"""

# ============================================================
# SEND TO OLLAMA
# ============================================================

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
)

# ============================================================
# DISPLAY RESULT
# ============================================================

if response.status_code != 200:
    print("Ollama error:")
    print(response.text)
    exit()

result = response.json()

print("\n================ AI ANALYSIS ================\n")

print(result["response"])