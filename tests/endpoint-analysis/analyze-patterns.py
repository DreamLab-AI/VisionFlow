#!/usr/bin/env python3
import json
import sys

# Load test results
with open('/home/devuser/workspace/project/tests/endpoint-analysis/endpoint-test-results.json', 'r') as f:
    data = json.load(f)

endpoints = data.get('endpoints', {})

# Categorize endpoints
working = []
crashing = []
timeout = []
error_5xx = []
connection_refused = []

for name, result in endpoints.items():
    if name == 'test_complete':
        continue
    
    http_code = result.get('http_code', '')
    exit_code = result.get('curl_exit_code', 0)
    backend_running = result.get('backend_running_after', 0)
    
    if http_code == '200':
        working.append(name)
    elif http_code == '000' or http_code == '':
        if exit_code == 7:
            connection_refused.append(name)
        elif exit_code == 28:
            timeout.append(name)
        else:
            crashing.append(name)
    elif http_code.startswith('5'):
        error_5xx.append(name)
    else:
        crashing.append(name)

print("=" * 60)
print("ENDPOINT PATTERN ANALYSIS")
print("=" * 60)
print()

print(f"✅ WORKING ({len(working)} endpoints):")
for ep in working:
    result = endpoints[ep]
    print(f"   - {ep}: {result['http_code']} in {result['time_total']}s")
print()

print(f"❌ CRASHING ({len(crashing)} endpoints):")
for ep in crashing:
    result = endpoints[ep]
    duration = result.get('actual_duration', 'N/A')
    backend = result.get('backend_running_after', '?')
    print(f"   - {ep}: HTTP {result['http_code']}, Exit {result['curl_exit_code']}, Duration {duration}s, Backend running: {backend}")
print()

print(f"⏱️  TIMEOUT ({len(timeout)} endpoints):")
for ep in timeout:
    result = endpoints[ep]
    print(f"   - {ep}: {result['http_code']}, Exit {result['curl_exit_code']}")
print()

print(f"🔌 CONNECTION REFUSED ({len(connection_refused)} endpoints):")
for ep in connection_refused:
    result = endpoints[ep]
    print(f"   - {ep}: Exit code {result['curl_exit_code']}")
print()

print(f"💥 5XX ERRORS ({len(error_5xx)} endpoints):")
for ep in error_5xx:
    result = endpoints[ep]
    print(f"   - {ep}: HTTP {result['http_code']}")
print()

# Pattern detection
print("=" * 60)
print("PATTERN CORRELATION ANALYSIS")
print("=" * 60)
print()

# Group by URL pattern
settings_endpoints = [ep for ep in endpoints.keys() if 'settings' in ep]
graph_endpoints = [ep for ep in endpoints.keys() if 'graph' in ep]
ontology_endpoints = [ep for ep in endpoints.keys() if 'ontology' in ep]
search_endpoints = [ep for ep in endpoints.keys() if 'search' in ep]
layout_endpoints = [ep for ep in endpoints.keys() if 'layout' in ep]

print("📊 ENDPOINT GROUPS:")
print(f"   Settings: {len(settings_endpoints)} endpoints")
for ep in settings_endpoints:
    status = "✅" if ep in working else "❌"
    print(f"      {status} {ep}")

print(f"   Graph: {len(graph_endpoints)} endpoints")
for ep in graph_endpoints:
    status = "✅" if ep in working else "❌"
    print(f"      {status} {ep}")

print(f"   Ontology: {len(ontology_endpoints)} endpoints")
for ep in ontology_endpoints:
    status = "✅" if ep in working else "❌"
    print(f"      {status} {ep}")

print(f"   Search: {len(search_endpoints)} endpoints")
for ep in search_endpoints:
    status = "✅" if ep in working else "❌"
    print(f"      {status} {ep}")

print(f"   Layout: {len(layout_endpoints)} endpoints")
for ep in layout_endpoints:
    status = "✅" if ep in working else "❌"
    print(f"      {status} {ep}")

print()
print("=" * 60)
print("KEY FINDINGS:")
print("=" * 60)

# Calculate failure rates by group
def failure_rate(group):
    total = len(group)
    if total == 0:
        return 0
    failed = len([ep for ep in group if ep not in working])
    return (failed / total) * 100

print(f"Settings failure rate: {failure_rate(settings_endpoints):.1f}%")
print(f"Graph failure rate: {failure_rate(graph_endpoints):.1f}%")
print(f"Ontology failure rate: {failure_rate(ontology_endpoints):.1f}%")
print(f"Search failure rate: {failure_rate(search_endpoints):.1f}%")
print(f"Layout failure rate: {failure_rate(layout_endpoints):.1f}%")

# Database correlation
print()
print("DATABASE CORRELATION:")
print("Settings endpoints (settings.db): ", "FAILING" if failure_rate(settings_endpoints) > 50 else "MOSTLY OK")
print("Graph endpoints (knowledge_graph.db): ", "FAILING" if failure_rate(graph_endpoints) > 50 else "MOSTLY OK")
print("Ontology endpoints (ontology.db): ", "FAILING" if failure_rate(ontology_endpoints) > 50 else "MOSTLY OK")

# Export analysis
analysis = {
    "working": working,
    "crashing": crashing,
    "timeout": timeout,
    "connection_refused": connection_refused,
    "error_5xx": error_5xx,
    "failure_rates": {
        "settings": failure_rate(settings_endpoints),
        "graph": failure_rate(graph_endpoints),
        "ontology": failure_rate(ontology_endpoints),
        "search": failure_rate(search_endpoints),
        "layout": failure_rate(layout_endpoints)
    }
}

with open('/home/devuser/workspace/project/tests/endpoint-analysis/pattern-analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print()
print("Analysis saved to pattern-analysis.json")
