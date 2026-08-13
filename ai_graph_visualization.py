import psycopg
import networkx as nx
import matplotlib.pyplot as plt

HOST = "localhost"
PORT = 5455
DATABASE = "knowledge_graph_demo"
USER = "postgres"
PASSWORD = "*******"

GRAPH_NAME = "ai_knowledge_graph"

conn = psycopg.connect(
    host=HOST,
    port=PORT,
    dbname=DATABASE,
    user=USER,
    password=PASSWORD
)

cur = conn.cursor()

print("\n================ AI GRAPH VISUALIZATION ================\n")

# ------------------------------------------------------------
# Get the AI relationships directly from the validated file
# ------------------------------------------------------------

import json

with open("validated_relationships.json", "r") as file:
    relationships = json.load(file)

# ------------------------------------------------------------
# Create NetworkX graph
# ------------------------------------------------------------

G = nx.DiGraph()

for relation in relationships:

    from_table = relation["from_table"]
    from_column = relation["from_column"]

    to_table = relation["to_table"]
    to_column = relation["to_column"]

    confidence = relation["confidence"]

    G.add_node(from_table)
    G.add_node(to_table)

    G.add_edge(
        from_table,
        to_table,
        relation=f"{from_column} → {to_column}",
        confidence=confidence
    )

# ------------------------------------------------------------
# Draw graph
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

pos = nx.spring_layout(
    G,
    seed=42,
    k=2.5
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=5000,
    node_color="skyblue",
    edgecolors="black",
    linewidths=2
)

nx.draw_networkx_edges(
    G,
    pos,
    width=2,
    arrows=True,
    arrowsize=25,
    arrowstyle="-|>"
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=14,
    font_weight="bold"
)

edge_labels = {
    (u, v):
    f"{d['relation']}\nconfidence: {d['confidence']}"
    for u, v, d in G.edges(data=True)
}

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=10,
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.8
    )
)

plt.title(
    "AI-Generated Knowledge Graph using Apache AGE",
    fontsize=20,
    fontweight="bold"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "ai_knowledge_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

cur.close()
conn.close()

print("\nAI knowledge graph visualization created successfully!")
print("Saved as: ai_knowledge_graph.png")
