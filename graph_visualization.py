import psycopg
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# POSTGRESQL CONNECTION
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
# IMPORTANT: LOAD APACHE AGE FOR THIS CONNECTION
# ============================================================

cur.execute("LOAD 'age';")

cur.execute(
    'SET search_path = ag_catalog, "$user", public;'
)

print("Apache AGE loaded successfully.")

# ============================================================
# GET KNOWLEDGE GRAPH DATA
#
# Customer -> Order -> Product -> Category
# ============================================================

query = """
SELECT *
FROM cypher(
    'ecommerce_graph',
    $$
    MATCH (c:Customer)-[:PLACED]->(o:Order)
          -[:CONTAINS]->(p:Product)
          -[:BELONGS_TO]->(cat:Category)

    RETURN
        c.customer_id,
        c.customer_name,
        o.order_id,
        p.product_id,
        p.product_name,
        cat.category_name

    ORDER BY o.order_id
    $$
) AS (
    customer_id agtype,
    customer_name agtype,
    order_id agtype,
    product_id agtype,
    product_name agtype,
    category_name agtype
);
"""

cur.execute(query)

rows = cur.fetchall()

print("Graph data retrieved successfully.")
print("Rows:", len(rows))

# ============================================================
# CLOSE DATABASE
# ============================================================

cur.close()
conn.close()

# ============================================================
# CLEAN AGE VALUES
# ============================================================

def clean(value):
    value = str(value)

    # Remove AGE string quotes
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]

    return value


# ============================================================
# CREATE NETWORKX GRAPH
# ============================================================

G = nx.DiGraph()

customers = {}
orders = {}
products = {}
categories = {}

for row in rows:

    customer_id = clean(row[0])
    customer_name = clean(row[1])

    order_id = clean(row[2])

    product_id = clean(row[3])
    product_name = clean(row[4])

    category_name = clean(row[5])

    # --------------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------------

    customers[customer_id] = customer_name

    G.add_node(
        customer_id,
        type="Customer",
        label=f"Customer\n{customer_id}\n{customer_name}"
    )

    # --------------------------------------------------------
    # ORDER
    # --------------------------------------------------------

    orders[order_id] = order_id

    G.add_node(
        order_id,
        type="Order",
        label=f"Order\n{order_id}"
    )

    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    products[product_id] = product_name

    G.add_node(
        product_id,
        type="Product",
        label=f"Product\n{product_id}\n{product_name}"
    )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    categories[category_name] = category_name

    category_id = "CAT_" + category_name

    G.add_node(
        category_id,
        type="Category",
        label=f"Category\n{category_name}"
    )

    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    G.add_edge(
        customer_id,
        order_id,
        relation="PLACED"
    )

    G.add_edge(
        order_id,
        product_id,
        relation="CONTAINS"
    )

    G.add_edge(
        product_id,
        category_id,
        relation="BELONGS_TO"
    )


# ============================================================
# POSITION NODES
# ============================================================

pos = {}

# ============================================================
# CUSTOMER COLUMN
# ============================================================

customer_list = list(customers.keys())

for i, customer_id in enumerate(customer_list):

    y = len(customer_list) - i

    pos[customer_id] = (0, y)


# ============================================================
# ORDER COLUMN
# ============================================================

order_list = list(orders.keys())

for i, order_id in enumerate(order_list):

    y = len(order_list) - i

    pos[order_id] = (4, y)


# ============================================================
# PRODUCT COLUMN
# ============================================================

product_list = list(products.keys())

for i, product_id in enumerate(product_list):

    y = len(product_list) - i

    pos[product_id] = (8, y)


# ============================================================
# CATEGORY COLUMN
# ============================================================

category_list = list(categories.keys())

category_y_positions = [9, 6.5, 4, 1.5]

for category_name, y in zip(
    category_list,
    category_y_positions
):

    category_id = "CAT_" + category_name

    pos[category_id] = (12, y)


# ============================================================
# DRAW GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(20, 11))

ax.set_title(
    "E-Commerce Knowledge Graph",
    fontsize=24,
    fontweight="bold",
    pad=25
)


# ============================================================
# NODE GROUPS
# ============================================================

customer_nodes = [
    n for n, d in G.nodes(data=True)
    if d["type"] == "Customer"
]

order_nodes = [
    n for n, d in G.nodes(data=True)
    if d["type"] == "Order"
]

product_nodes = [
    n for n, d in G.nodes(data=True)
    if d["type"] == "Product"
]

category_nodes = [
    n for n, d in G.nodes(data=True)
    if d["type"] == "Category"
]


# ============================================================
# DRAW CUSTOMER NODES
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=customer_nodes,
    node_color="skyblue",
    node_size=2300,
    node_shape="o",
    ax=ax
)


# ============================================================
# DRAW ORDER NODES
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=order_nodes,
    node_color="lightgreen",
    node_size=2200,
    node_shape="s",
    ax=ax
)


# ============================================================
# DRAW PRODUCT NODES
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=product_nodes,
    node_color="orange",
    node_size=2400,
    node_shape="o",
    ax=ax
)


# ============================================================
# DRAW CATEGORY NODES
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=category_nodes,
    node_color="plum",
    node_size=2600,
    node_shape="D",
    ax=ax
)


# ============================================================
# DRAW EDGES
# ============================================================

nx.draw_networkx_edges(
    G,
    pos,
    edge_color="black",
    width=1.5,
    arrows=True,
    arrowsize=15,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.0",
    ax=ax
)


# ============================================================
# NODE LABELS
# ============================================================

labels = {
    node: data["label"]
    for node, data in G.nodes(data=True)
}

nx.draw_networkx_labels(
    G,
    pos,
    labels=labels,
    font_size=9,
    font_weight="bold",
    ax=ax
)


# ============================================================
# EDGE LABELS
# ============================================================

edge_labels = {
    (u, v): data["relation"]
    for u, v, data in G.edges(data=True)
}

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8,
    label_pos=0.5,
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.8
    ),
    ax=ax
)


# ============================================================
# COLUMN HEADINGS
# ============================================================

ax.text(
    0,
    11,
    "CUSTOMERS",
    fontsize=17,
    fontweight="bold",
    ha="center"
)

ax.text(
    4,
    11,
    "ORDERS",
    fontsize=17,
    fontweight="bold",
    ha="center"
)

ax.text(
    8,
    11,
    "PRODUCTS",
    fontsize=17,
    fontweight="bold",
    ha="center"
)

ax.text(
    12,
    11,
    "CATEGORIES",
    fontsize=17,
    fontweight="bold",
    ha="center"
)


# ============================================================
# CLEAN UP
# ============================================================

ax.set_xlim(-1.5, 13.5)
ax.set_ylim(0, 11.5)

ax.axis("off")

plt.tight_layout()


# ============================================================
# SAVE IMAGE
# ============================================================

plt.savefig(
    "ecommerce_knowledge_graph.png",
    dpi=300,
    bbox_inches="tight"
)

print("Graph saved as ecommerce_knowledge_graph.png")

plt.show()