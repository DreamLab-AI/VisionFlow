#!/bin/bash

echo "===== DATABASE-SPECIFIC TESTING ====="
echo ""

# Test if databases are accessible
echo "1. Database File Accessibility:"
docker exec visionflow_container ls -lh /app/backend/data/*.db 2>&1

echo ""
echo "2. Database Integrity Checks:"

# Check settings.db
echo "   settings.db:"
docker exec visionflow_container sqlite3 /app/backend/data/settings.db "PRAGMA integrity_check;" 2>&1

# Check knowledge_graph.db
echo "   knowledge_graph.db:"
docker exec visionflow_container sqlite3 /app/backend/data/knowledge_graph.db "PRAGMA integrity_check;" 2>&1

# Check ontology.db
echo "   ontology.db:"
docker exec visionflow_container sqlite3 /app/backend/data/ontology.db "PRAGMA integrity_check;" 2>&1

echo ""
echo "3. Table Structure Checks:"

echo "   settings.db tables:"
docker exec visionflow_container sqlite3 /app/backend/data/settings.db ".tables" 2>&1

echo "   knowledge_graph.db tables:"
docker exec visionflow_container sqlite3 /app/backend/data/knowledge_graph.db ".tables" 2>&1

echo "   ontology.db tables:"
docker exec visionflow_container sqlite3 /app/backend/data/ontology.db ".tables" 2>&1

echo ""
echo "4. Database Lock Status:"
docker exec visionflow_container lsof /app/backend/data/*.db 2>&1 || echo "No locks detected (or lsof not available)"

echo ""
echo "5. Backend Process Database Connections:"
docker exec visionflow_container sh -c 'lsof -p $(pgrep -f "node.*server.js") | grep .db' 2>&1 || echo "No database connections visible"

