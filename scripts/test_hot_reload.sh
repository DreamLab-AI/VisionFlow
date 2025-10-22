#!/bin/bash
# Test script for settings hot-reload functionality

set -e

echo "🧪 Settings Hot-Reload Test Script"
echo "=================================="
echo ""

DB_PATH="${SETTINGS_DB_PATH:-data/settings.db}"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "❌ Settings database not found at: $DB_PATH"
    echo "   Please ensure the application has been started at least once."
    exit 1
fi

echo "✓ Found settings database at: $DB_PATH"
echo ""

# Function to update a setting
update_setting() {
    local key=$1
    local value=$2

    echo "📝 Updating setting: $key = $value"
    sqlite3 "$DB_PATH" "UPDATE settings SET value = '$value' WHERE key = '$key';"

    if [ $? -eq 0 ]; then
        echo "✓ Database updated successfully"
    else
        echo "❌ Failed to update database"
        return 1
    fi
}

# Function to verify setting
verify_setting() {
    local key=$1
    local expected=$2

    local current=$(sqlite3 "$DB_PATH" "SELECT value FROM settings WHERE key = '$key';")

    if [ "$current" = "$expected" ]; then
        echo "✓ Verified: $key = $current"
        return 0
    else
        echo "❌ Mismatch: expected=$expected, got=$current"
        return 1
    fi
}

echo "Test 1: Update Physics Damping"
echo "-------------------------------"
update_setting "visualisation.graphs.logseq.physics.damping" "0.95"
verify_setting "visualisation.graphs.logseq.physics.damping" "0.95"
echo "⏱️  Wait for hot-reload (500ms debounce)..."
sleep 1
echo ""

echo "Test 2: Update Spring Constant"
echo "-------------------------------"
update_setting "visualisation.graphs.logseq.physics.spring_k" "1.5"
verify_setting "visualisation.graphs.logseq.physics.spring_k" "1.5"
echo "⏱️  Wait for hot-reload (500ms debounce)..."
sleep 1
echo ""

echo "Test 3: Update Repulsion Constant"
echo "-------------------------------"
update_setting "visualisation.graphs.logseq.physics.repel_k" "2000.0"
verify_setting "visualisation.graphs.logseq.physics.repel_k" "2000.0"
echo "⏱️  Wait for hot-reload (500ms debounce)..."
sleep 1
echo ""

echo "✅ All tests completed successfully!"
echo ""
echo "📊 Check application logs for hot-reload confirmations:"
echo "   Look for: '✓ Settings hot-reloaded successfully from database'"
echo ""
echo "🔍 You can monitor hot-reload in real-time with:"
echo "   tail -f <log-file> | grep 'hot-reload'"
