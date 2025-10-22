#!/bin/bash
# Migration 001 Runner
# Simple shell script to execute database migration and validate results

set -e  # Exit on error

echo "🔄 Starting Database Migration 001..."
echo ""

# Configuration
DB_PATH="data/settings.db"
SQL_PATH="scripts/migrations/001_add_missing_settings.sql"

# Verify database exists
if [ ! -f "$DB_PATH" ]; then
    echo "❌ Error: Database not found at $DB_PATH"
    exit 1
fi

# Verify migration script exists
if [ ! -f "$SQL_PATH" ]; then
    echo "❌ Error: Migration script not found at $SQL_PATH"
    exit 1
fi

echo "📄 Migration script: $SQL_PATH"
echo "🔗 Database: $DB_PATH"
echo ""

# Get initial count
INITIAL_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM settings")
echo "📊 Initial settings count: $INITIAL_COUNT"
echo ""

# Execute migration
echo "⚡ Executing migration..."
if sqlite3 "$DB_PATH" < "$SQL_PATH" 2>&1; then
    echo "✅ Migration SQL executed successfully"
else
    echo "❌ Migration failed!"
    exit 1
fi
echo ""

# Get final count
FINAL_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM settings")
ADDED_COUNT=$((FINAL_COUNT - INITIAL_COUNT))

echo "📊 Final settings count: $FINAL_COUNT"
echo "➕ Settings added: $ADDED_COUNT"
echo ""

# Verify expected count
if [ "$ADDED_COUNT" -eq 73 ]; then
    echo "✅ SUCCESS: All 73 settings added correctly!"
else
    echo "⚠️  WARNING: Expected 73 settings, but added $ADDED_COUNT"
fi
echo ""

# Category breakdown
echo "📋 Category Breakdown:"
sqlite3 "$DB_PATH" << 'EOF'
SELECT
    CASE
        WHEN key LIKE 'analytics.%' THEN 'analytics'
        WHEN key LIKE 'dashboard.%' THEN 'dashboard'
        WHEN key LIKE 'performance.%' THEN 'performance'
        WHEN key LIKE 'gpu.%' THEN 'gpu'
        WHEN key LIKE 'effects.%' THEN 'effects'
        WHEN key LIKE 'dev.%' THEN 'developer'
        WHEN key LIKE 'agents.%' THEN 'agents'
        ELSE 'other'
    END as category,
    COUNT(*) as count
FROM settings
WHERE parent_key = 'app_full_settings'
GROUP BY category
ORDER BY category;
EOF
echo ""

# Check for duplicates
DUPLICATE_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM (SELECT key, COUNT(*) as cnt FROM settings GROUP BY key HAVING cnt > 1)")

if [ "$DUPLICATE_COUNT" -gt 0 ]; then
    echo "⚠️  WARNING: $DUPLICATE_COUNT duplicate keys found!"
    sqlite3 "$DB_PATH" "SELECT key, COUNT(*) as cnt FROM settings GROUP BY key HAVING cnt > 1"
else
    echo "✅ No duplicate keys found"
fi
echo ""

# Value type distribution
echo "📊 Value Type Distribution:"
sqlite3 "$DB_PATH" << 'EOF'
SELECT value_type, COUNT(*) as count
FROM settings
WHERE parent_key = 'app_full_settings'
GROUP BY value_type
ORDER BY value_type;
EOF
echo ""

echo "🎉 Migration 001 Complete!"
echo "📝 Documentation: docs/MIGRATION_001_RESULTS.md"
echo ""
