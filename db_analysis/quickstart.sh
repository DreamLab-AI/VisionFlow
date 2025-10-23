#!/bin/bash
# VisionFlow Database Quick Start Script
# Initializes databases with mock credentials and builds knowledge graph

set -e  # Exit on error

echo "======================================================================"
echo "VisionFlow Database Quick Start"
echo "======================================================================"
echo ""

CONTAINER="visionflow_container"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER"; then
    echo "❌ Error: Container '$CONTAINER' is not running"
    echo "Start it with: docker start $CONTAINER"
    exit 1
fi

echo "✓ Container '$CONTAINER' is running"
echo ""

# Step 1: Add mock credentials
echo "Step 1/3: Adding mock credentials..."
docker exec -i "$CONTAINER" sqlite3 /app/data/settings.db << 'SQL'
INSERT OR IGNORE INTO api_keys (
    service_name,
    api_key_encrypted,
    key_name,
    key_description,
    scopes,
    is_active,
    created_at,
    updated_at
) VALUES
('nostr', 'wss://relay.damus.io', 'Mock Nostr Relay', 'Development relay', 'read,write', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('github', 'ghp_mock_dev_token', 'Mock GitHub', 'Development token', 'repo,read:org', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('ragflow', 'mock_ragflow_key', 'Mock RAGFlow', 'Development key', 'read,write', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('anthropic', 'sk-ant-mock-key', 'Mock Claude', 'Development key', 'messages', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
SQL

# Verify credentials
CRED_COUNT=$(docker exec "$CONTAINER" sqlite3 /app/data/settings.db "SELECT COUNT(*) FROM api_keys")
echo "  ✓ Added $CRED_COUNT mock credentials"
echo ""

# Step 2: Check if graph build endpoint exists
echo "Step 2/3: Checking VisionFlow API..."
if docker exec "$CONTAINER" curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "  ✓ VisionFlow API is responding"

    # Try to trigger graph build via API
    echo ""
    echo "  Attempting to build knowledge graph from markdown files..."

    # Check if rebuild endpoint exists
    REBUILD_RESPONSE=$(docker exec "$CONTAINER" curl -s -X POST \
        http://localhost:8080/api/graph/rebuild \
        -H "Content-Type: application/json" \
        -d '{"source": "markdown"}' 2>&1 || echo "FAILED")

    if [[ "$REBUILD_RESPONSE" != "FAILED" ]] && [[ "$REBUILD_RESPONSE" != *"404"* ]]; then
        echo "  ✓ Graph rebuild triggered"
        echo "    Response: $REBUILD_RESPONSE"
    else
        echo "  ⚠️  Graph rebuild endpoint not available"
        echo "    Please rebuild manually via VisionFlow UI"
    fi
else
    echo "  ⚠️  VisionFlow API not responding on port 8080"
    echo "    Graph build must be triggered manually"
fi
echo ""

# Step 3: Verify current state
echo "Step 3/3: Verifying database state..."

NODE_COUNT=$(docker exec "$CONTAINER" sqlite3 /app/data/knowledge_graph.db "SELECT COUNT(*) FROM nodes" 2>/dev/null || echo "0")
EDGE_COUNT=$(docker exec "$CONTAINER" sqlite3 /app/data/knowledge_graph.db "SELECT COUNT(*) FROM edges" 2>/dev/null || echo "0")

echo "  Nodes: $NODE_COUNT / 185 expected"
echo "  Edges: $EDGE_COUNT / 4014 expected"

if [ "$NODE_COUNT" -eq 0 ]; then
    echo ""
    echo "  ⚠️  Knowledge graph is still empty"
    echo "     Manual graph build required:"
    echo ""
    echo "     Option 1 - Via UI:"
    echo "       http://localhost:8080 -> Graph Management -> Rebuild"
    echo ""
    echo "     Option 2 - Via API:"
    echo "       curl -X POST http://localhost:8080/api/graph/rebuild \\"
    echo "         -H 'Content-Type: application/json' \\"
    echo "         -d '{\"source\": \"markdown\"}'"
else
    echo "  ✓ Knowledge graph has data!"
fi

echo ""
echo "======================================================================"
echo "Quick Start Complete!"
echo "======================================================================"
echo ""
echo "📊 Database Status:"
echo "  - Settings:   ✓ Configured ($CRED_COUNT API keys)"
if [ "$NODE_COUNT" -gt 0 ]; then
    echo "  - Graph:      ✓ Built ($NODE_COUNT nodes, $EDGE_COUNT edges)"
else
    echo "  - Graph:      ⚠️  Needs building (0 nodes)"
fi
echo "  - Ontology:   ⚠️  Empty (optional)"
echo ""
echo "🌐 Access VisionFlow:"
echo "  http://localhost:8080"
echo ""
echo "📚 Documentation:"
echo "  - Full Report:  /home/devuser/workspace/project/docs/VisionFlow_Database_Integrity_Report.md"
echo "  - Quick Guide:  /home/devuser/workspace/project/db_analysis/README.md"
echo "  - Summary:      /home/devuser/workspace/project/db_analysis/SUMMARY.txt"
echo ""
echo "🔧 Verification:"
echo "  python3 /home/devuser/workspace/project/db_analysis/analyze_databases.py"
echo ""
echo "======================================================================"
