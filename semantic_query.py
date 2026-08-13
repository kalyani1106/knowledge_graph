import json
import requests

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

SEMANTIC_FILE = "semantic_catalog.json"


# ============================================================
# LOAD SEMANTIC CATALOG
# ============================================================

with open(SEMANTIC_FILE, "r", encoding="utf-8") as file:
    semantic_catalog = json.load(file)


tables = semantic_catalog.get("tables", [])
relationships = semantic_catalog.get("relationships", [])


# ============================================================
# PREPARE SEMANTIC CONTEXT
# ============================================================

semantic_context = {
    "tables": tables,
    "relationships": relationships
}


# ============================================================
# ASK LOCAL AI
# ============================================================

def understand_question(question):

    prompt = f"""
You are a semantic query analyzer for a database knowledge graph.

The database may contain hundreds or thousands of tables.

You must use ONLY the tables, columns and relationships
provided in the semantic catalog.

Do not invent tables.
Do not invent columns.
Do not invent relationships.

Analyze the user's question and identify the relevant
database concepts.

Return ONLY valid JSON in this format:

{{
    "question": "original question",
    "intent": "short description of what the user wants",
    "relevant_tables": [
        "table1",
        "table2"
    ],
    "relevant_columns": [
        {{
            "table": "table_name",
            "column": "column_name",
            "reason": "why this column is relevant"
        }}
    ],
    "relationships": [
        {{
            "from_table": "table_name",
            "from_column": "column_name",
            "to_table": "table_name",
            "to_column": "column_name"
        }}
    ]
}}

SEMANTIC CATALOG:

{json.dumps(semantic_context, indent=2)}

USER QUESTION:

{question}
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

def parse_response(text):

    text = text.strip()

    # Remove markdown code fences if Llama adds them
    if text.startswith("```"):
        lines = text.splitlines()

        lines = [
            line for line in lines
            if not line.strip().startswith("```")
        ]

        text = "\n".join(lines).strip()

    return json.loads(text)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n================ SEMANTIC QUERY LAYER ================\n")

print(
    f"Semantic tables available: {len(tables)}"
)

print(
    f"Relationships available: {len(relationships)}"
)

print("\nThe system is ready for natural-language questions.\n")


# ============================================================
# INTERACTIVE QUESTIONS
# ============================================================

while True:

    question = input(
        "Ask a question "
        "(type 'exit' to stop): "
    )

    if question.lower().strip() == "exit":
        break

    print("\nAnalyzing question...\n")

    try:

        ai_response = understand_question(question)

        result = parse_response(ai_response)

        print(
            "================ AI INTERPRETATION ================\n"
        )

        print(
            json.dumps(
                result,
                indent=4
            )
        )

    except Exception as e:

        print(
            "\nError while analyzing question:"
        )

        print(e)