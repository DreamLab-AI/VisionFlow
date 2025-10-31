#!/usr/bin/env python3
import sqlite3
import json
import sys
from datetime import datetime

DB_PATH = "/app/data/knowledge_graph.db"

sample_data = {
    "nodes": [
        {
            "id": "person1",
            "type": "concept",
            "properties": {
                "name": "Alice Johnson",
                "age": 30,
                "email": "alice@example.com"
            }
        },
        {
            "id": "person2",
            "type": "concept",
            "properties": {
                "name": "Bob Smith",
                "age": 35,
                "email": "bob@example.com"
            }
        },
        {
            "id": "company1",
            "type": "page",
            "properties": {
                "name": "ACME Corporation",
                "industry": "Technology"
            }
        },
        {
            "id": "file1",
            "type": "page",
            "properties": {
                "name": "report.pdf",
                "fileSize": 1048576,
                "createdDate": "2025-01-15T10:30:00Z"
            }
        }
    ],
    "edges": [
        {"from": "person1", "to": "company1", "type": "link"},
        {"from": "person1", "to": "person2", "type": "related"},
        {"from": "person1", "to": "file1", "type": "reference"}
    ]
}

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Insert nodes and track mapping
id_map = {}
for i, node in enumerate(sample_data["nodes"], start=1):
    cur.execute("""
        INSERT INTO nodes (metadata_id, label, node_type, metadata, x, y, z)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        node["id"],
        node["properties"].get("name", node["id"]),
        node["type"],
        json.dumps(node["properties"]),
        float(i * 100),
        float(i * 50),
        0.0
    ))
    id_map[node["id"]] = cur.lastrowid

# Insert edges
for edge in sample_data["edges"]:
    edge_id = f"{edge['from']}-{edge['to']}"
    cur.execute("""
        INSERT INTO edges (id, source, target, edge_type, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (
        edge_id,
        id_map[edge["from"]],
        id_map[edge["to"]],
        edge["type"],
        "{}"
    ))

conn.commit()

# Verify
cur.execute("SELECT COUNT(*) FROM nodes")
node_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM edges")
edge_count = cur.fetchone()[0]

print(f"✅ Populated database: {node_count} nodes, {edge_count} edges")

conn.close()
