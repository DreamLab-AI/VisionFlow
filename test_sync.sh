#!/bin/bash
# Test script to verify GitHub sync data flow

echo "Testing GitHub sync data flow..."

# Test 1: Check database before sync
echo "Step 1: Checking database before sync..."
docker cp visionflow_container:/app/data/knowledge_graph.db /tmp/kg_before.db
NODES_BEFORE=$(sqlite3 /tmp/kg_before.db "SELECT COUNT(*) FROM nodes;")
EDGES_BEFORE=$(sqlite3 /tmp/kg_before.db "SELECT COUNT(*) FROM edges;")
echo "  Nodes before: $NODES_BEFORE"
echo "  Edges before: $EDGES_BEFORE"

# Test 2: Trigger sync via HTTP
echo "Step 2: Triggering GitHub sync via API..."
RESPONSE=$(curl -s -X POST http://localhost:4000/api/admin/sync -m 300)
echo "  Response: $RESPONSE"

# Test 3: Check database after sync
echo "Step 3: Checking database after sync..."
sleep 2
docker cp visionflow_container:/app/data/knowledge_graph.db /tmp/kg_after.db
NODES_AFTER=$(sqlite3 /tmp/kg_after.db "SELECT COUNT(*) FROM nodes;")
EDGES_AFTER=$(sqlite3 /tmp/kg_after.db "SELECT COUNT(*) FROM edges;")
echo "  Nodes after: $NODES_AFTER"
echo "  Edges after: $EDGES_AFTER"

# Test 4: Check client API
echo "Step 4: Checking client API response..."
CLIENT_RESPONSE=$(curl -s http://localhost:4000/api/graph/data | jq '.nodes | length')
echo "  Client sees nodes: $CLIENT_RESPONSE"

echo "Test complete!"
