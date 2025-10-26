#!/usr/bin/env python3
import sys
import json

data = json.load(sys.stdin)
nodes = data.get("nodes", [])
edges = data.get("edges", [])
total = len(nodes)
with_public = sum(1 for n in nodes if n.get("metadata", {}).get("public") == "true")
without = total - with_public

print(f"=== GitHub Sync Filter Fix Results ===")
print(f"Total nodes: {total}")
print(f"Nodes WITH public=true: {with_public} ({with_public/total*100:.1f}%)")
print(f"Nodes WITHOUT public: {without} ({without/total*100:.1f}%)")
print(f"Total edges: {len(edges)}")
print(f"")
print(f"Expected: 100% of nodes should have public=true")
if without == 0:
    print(f"Status: ✅ SUCCESS")
else:
    print(f"Status: ❌ FAILED")
    print(f"\nSample nodes without public metadata:")
    for node in [n for n in nodes if n.get("metadata", {}).get("public") != "true"][:5]:
        label = node.get("label", "no-label")
        node_type = node.get("metadata", {}).get("type", "unknown")
        print(f"  - {label} (type: {node_type})")
