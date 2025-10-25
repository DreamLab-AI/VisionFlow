#!/usr/bin/env python3
import json

# Create corrected analysis based on test results
results = {
    "working": ["health", "config"],
    "http_404_not_implemented": [
        "settings_system", "settings_visualisation", "settings_database", "settings_api",
        "graph_nodes", "graph_edges", "graph_stats",
        "ontology_individuals", "search_nodes", "search_semantic",
        "layout_force", "layout_hierarchical"
    ],
    "empty_reply_crash": ["settings_root", "ontology_classes", "ontology_properties"],
    "timeout": ["graph_data"],
    "summary": {
        "total_endpoints": 18,
        "working": 2,
        "not_implemented_404": 12,
        "crash_empty_reply": 3,
        "timeout": 1,
        "critical_findings": [
            "Config endpoint WORKS - this is a major discovery!",
            "12 endpoints return 404 - routes not implemented yet",
            "3 endpoints cause empty reply crash (exit 52)",
            "1 endpoint times out after 10 seconds",
            "Backend process count shows 0 - supervisord not in container or not configured"
        ]
    }
}

print("=" * 70)
print("CORRECTED ENDPOINT ANALYSIS")
print("=" * 70)
print()

print("✅ WORKING ENDPOINTS (2/18):")
print("   - /api/health: Basic health check (HTTP 200)")
print("   - /api/config: Configuration endpoint (HTTP 200) ⭐ WORKS!")
print()

print("❌ NOT IMPLEMENTED - 404 (12/18):")
for ep in results["http_404_not_implemented"]:
    print(f"   - {ep}")
print()

print("💥 EMPTY REPLY CRASH - Exit 52 (3/18):")
for ep in results["empty_reply_crash"]:
    print(f"   - {ep}: Connection closed by server")
print()

print("⏱️  TIMEOUT - Exit 28 (1/18):")
for ep in results["timeout"]:
    print(f"   - {ep}: 10 second timeout, backend hung")
print()

print("=" * 70)
print("CRITICAL INSIGHTS")
print("=" * 70)
print()
print("1. CONFIG ENDPOINT WORKS! 🎉")
print("   - This contradicts earlier hypothesis")
print("   - /api/config returns full JSON configuration")
print("   - Shows database access CAN work")
print()
print("2. MOST ENDPOINTS NOT IMPLEMENTED")
print("   - 12/18 endpoints return 404")
print("   - Backend routes exist but handlers incomplete")
print()
print("3. THREE SPECIFIC CRASHES:")
print("   - /api/settings (root)")
print("   - /api/ontology/classes")
print("   - /api/ontology/properties")
print("   Common: All try to access specific database tables")
print()
print("4. ONE TIMEOUT:")
print("   - /api/graph/data hangs for 10 seconds")
print("   - Backend doesn't crash, but doesn't respond")
print("   - Possible: Query too slow or infinite loop")
print()

with open('/home/devuser/workspace/project/tests/endpoint-analysis/corrected-analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Analysis saved to corrected-analysis.json")
