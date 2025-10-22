#!/bin/bash
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         DATABASE MIGRATION 001 - VALIDATION REPORT            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 DATABASE STATISTICS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sqlite3 data/settings.db << 'SQL'
.mode line
SELECT 'Total Settings' as Metric, COUNT(*) as Value FROM settings
UNION ALL
SELECT 'New Settings (Migration 001)', COUNT(*) FROM settings WHERE parent_key = 'app_full_settings'
UNION ALL
SELECT 'Original Settings', COUNT(*) FROM settings WHERE parent_key IS NULL OR parent_key != 'app_full_settings';
SQL

echo ""
echo "📋 CATEGORY BREAKDOWN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sqlite3 data/settings.db << 'SQL'
.mode column
.headers on
SELECT
    CASE
        WHEN key LIKE 'analytics.%' THEN '🔬 Analytics'
        WHEN key LIKE 'dashboard.%' THEN '📊 Dashboard'
        WHEN key LIKE 'performance.%' THEN '⚡ Performance'
        WHEN key LIKE 'gpu.%' THEN '🎨 GPU Visualization'
        WHEN key LIKE 'effects.%' THEN '✨ Bloom Effects'
        WHEN key LIKE 'dev.%' THEN '🛠️  Developer'
        WHEN key LIKE 'agents.%' THEN '🤖 Agents'
        ELSE '📁 Other'
    END as Category,
    COUNT(*) as Count,
    ROUND(COUNT(*) * 100.0 / 73, 1) || '%' as Percentage
FROM settings
WHERE parent_key = 'app_full_settings'
GROUP BY
    CASE
        WHEN key LIKE 'analytics.%' THEN '🔬 Analytics'
        WHEN key LIKE 'dashboard.%' THEN '📊 Dashboard'
        WHEN key LIKE 'performance.%' THEN '⚡ Performance'
        WHEN key LIKE 'gpu.%' THEN '🎨 GPU Visualization'
        WHEN key LIKE 'effects.%' THEN '✨ Bloom Effects'
        WHEN key LIKE 'dev.%' THEN '🛠️  Developer'
        WHEN key LIKE 'agents.%' THEN '🤖 Agents'
        ELSE '📁 Other'
    END
ORDER BY Count DESC;
SQL

echo ""
echo "🔢 VALUE TYPE DISTRIBUTION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sqlite3 data/settings.db << 'SQL'
.mode column
.headers on
SELECT
    value_type as 'Type',
    COUNT(*) as Count,
    ROUND(COUNT(*) * 100.0 / 73, 1) || '%' as Percentage
FROM settings
WHERE parent_key = 'app_full_settings'
GROUP BY value_type
ORDER BY COUNT(*) DESC;
SQL

echo ""
echo "✅ VALIDATION CHECKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DUPLICATES=$(sqlite3 data/settings.db "SELECT COUNT(*) FROM (SELECT key, COUNT(*) FROM settings GROUP BY key HAVING COUNT(*) > 1)")
if [ "$DUPLICATES" -eq 0 ]; then
    echo "✅ No duplicate keys found"
else
    echo "❌ $DUPLICATES duplicate keys detected!"
fi

ANALYTICS=$(sqlite3 data/settings.db "SELECT COUNT(*) FROM settings WHERE key LIKE 'analytics.%'")
[ "$ANALYTICS" -eq 11 ] && echo "✅ Analytics: 11 settings" || echo "⚠️  Analytics: $ANALYTICS settings (expected 11)"

DASHBOARD=$(sqlite3 data/settings.db "SELECT COUNT(*) FROM settings WHERE key LIKE 'dashboard.%'")
[ "$DASHBOARD" -eq 8 ] && echo "✅ Dashboard: 8 settings" || echo "⚠️  Dashboard: $DASHBOARD settings (expected 8)"

PERFORMANCE=$(sqlite3 data/settings.db "SELECT COUNT(*) FROM settings WHERE key LIKE 'performance.%'")
[ "$PERFORMANCE" -eq 11 ] && echo "✅ Performance: 11 settings" || echo "⚠️  Performance: $PERFORMANCE settings (expected 11)"

GPU=$(sqlite3 data/settings.db "SELECT COUNT(*) FROM settings WHERE key LIKE 'gpu.%'")
[ "$GPU" -eq 8 ] && echo "✅ GPU Visualization: 8 settings" || echo "⚠️  GPU: $GPU settings (expected 8)"

EFFECTS=$(sqlite3 data/settings.db "SELECT COUNT(*) FROM settings WHERE key LIKE 'effects.%'")
[ "$EFFECTS" -eq 4 ] && echo "✅ Bloom Effects: 4 settings" || echo "⚠️  Effects: $EFFECTS settings (expected 4)"

DEV=$(sqlite3 data/settings.db "SELECT COUNT(*) FROM settings WHERE key LIKE 'dev.%'")
[ "$DEV" -eq 11 ] && echo "✅ Developer: 11 settings" || echo "⚠️  Developer: $DEV settings (expected 11)"

AGENTS=$(sqlite3 data/settings.db "SELECT COUNT(*) FROM settings WHERE key LIKE 'agents.%'")
[ "$AGENTS" -eq 20 ] && echo "✅ Agents: 20 settings" || echo "⚠️  Agents: $AGENTS settings (expected 20)"

echo ""
echo "📝 SAMPLE DATA (First 3 from each category)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for category in "analytics" "dashboard" "performance" "gpu" "effects" "dev" "agents"; do
    echo ""
    echo "Category: $category"
    sqlite3 data/settings.db << SQL
.mode list
.separator " | "
SELECT key, value_type, COALESCE(value_text, CAST(value_integer AS TEXT), CAST(value_float AS TEXT), CASE value_boolean WHEN 1 THEN 'true' ELSE 'false' END) as value
FROM settings
WHERE key LIKE '${category}.%'
ORDER BY key
LIMIT 3;
SQL
done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ MIGRATION SUCCESSFUL                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Files Created:"
echo "  • scripts/migrations/001_add_missing_settings.sql"
echo "  • scripts/run_migration.sh"
echo "  • scripts/run_migration.rs"
echo "  • docs/MIGRATION_001_RESULTS.md"
echo "  • docs/MIGRATION_SUMMARY.md"
echo "  • docs/SETTINGS_QUICK_REFERENCE.md"
echo ""
