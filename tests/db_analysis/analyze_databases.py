#!/usr/bin/env python3
"""
VisionFlow Database Analysis Script
Analyzes settings.db, knowledge_graph.db, and ontology.db for integrity and completeness
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any

def analyze_database(db_path: str) -> Dict[str, Any]:
    """Analyze a single SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    analysis = {
        'database': db_path,
        'tables': {},
        'total_records': 0,
        'issues': [],
        'credentials': []
    }

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()

    for table_name, in tables:
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        # Get record count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        # Get sample data (first 5 rows)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()

        analysis['tables'][table_name] = {
            'columns': [col[1] for col in columns],
            'record_count': count,
            'schema': columns,
            'sample': sample_data[:2]  # Just first 2 for brevity
        }

        analysis['total_records'] += count

    conn.close()
    return analysis

def analyze_settings_db(db_path: str) -> Dict[str, Any]:
    """Detailed analysis of settings.db"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    analysis = analyze_database(db_path)

    # Check for credential-related settings
    try:
        cursor.execute("""
            SELECT key, value FROM settings
            WHERE key LIKE '%api%'
               OR key LIKE '%token%'
               OR key LIKE '%credential%'
               OR key LIKE '%password%'
               OR key LIKE '%secret%'
               OR key LIKE '%nostr%'
               OR key LIKE '%github%'
               OR key LIKE '%ragflow%'
            ORDER BY key
        """)
        credentials = cursor.fetchall()
        analysis['credentials'] = credentials

        # Check for missing/empty credentials
        for key, value in credentials:
            if not value or value.strip() == '':
                analysis['issues'].append(f"Empty credential: {key}")
    except Exception as e:
        analysis['issues'].append(f"Error checking credentials: {e}")

    # Get all settings for overview
    try:
        cursor.execute("SELECT COUNT(*) FROM settings")
        total = cursor.fetchone()[0]
        analysis['total_settings'] = total

        cursor.execute("SELECT key FROM settings ORDER BY key")
        all_keys = [row[0] for row in cursor.fetchall()]
        analysis['all_setting_keys'] = all_keys
    except Exception as e:
        analysis['issues'].append(f"Error getting settings: {e}")

    conn.close()
    return analysis

def analyze_knowledge_graph_db(db_path: str) -> Dict[str, Any]:
    """Detailed analysis of knowledge_graph.db"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    analysis = analyze_database(db_path)

    # Expected: 185 nodes, 4014 edges
    expected_nodes = 185
    expected_edges = 4014

    try:
        cursor.execute("SELECT COUNT(*) FROM nodes")
        node_count = cursor.fetchone()[0]
        analysis['node_count'] = node_count

        if node_count != expected_nodes:
            analysis['issues'].append(
                f"Node count mismatch: expected {expected_nodes}, found {node_count}"
            )

        cursor.execute("SELECT COUNT(*) FROM edges")
        edge_count = cursor.fetchone()[0]
        analysis['edge_count'] = edge_count

        if edge_count != expected_edges:
            analysis['issues'].append(
                f"Edge count mismatch: expected {expected_edges}, found {edge_count}"
            )

        # Check for orphaned nodes (nodes with no edges)
        cursor.execute("""
            SELECT COUNT(*) FROM nodes
            WHERE id NOT IN (SELECT source FROM edges)
              AND id NOT IN (SELECT target FROM edges)
        """)
        orphaned = cursor.fetchone()[0]
        if orphaned > 0:
            analysis['orphaned_nodes'] = orphaned
            analysis['issues'].append(f"Found {orphaned} orphaned nodes (no connections)")

        # Get node type distribution
        cursor.execute("SELECT type, COUNT(*) as cnt FROM nodes GROUP BY type ORDER BY cnt DESC")
        node_types = cursor.fetchall()
        analysis['node_type_distribution'] = dict(node_types)

        # Get edge type distribution
        cursor.execute("SELECT type, COUNT(*) as cnt FROM edges GROUP BY type ORDER BY cnt DESC")
        edge_types = cursor.fetchall()
        analysis['edge_type_distribution'] = dict(edge_types)

    except Exception as e:
        analysis['issues'].append(f"Error analyzing graph: {e}")

    conn.close()
    return analysis

def analyze_ontology_db(db_path: str) -> Dict[str, Any]:
    """Detailed analysis of ontology.db"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    analysis = analyze_database(db_path)

    # Check for OWL/RDF specific data
    try:
        # Check for common ontology tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        expected_ontology_tables = ['classes', 'properties', 'individuals', 'triples']
        missing_tables = [t for t in expected_ontology_tables if t not in tables]

        if missing_tables:
            analysis['issues'].append(f"Missing expected ontology tables: {missing_tables}")

        # Get statistics for each table
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            if count == 0:
                analysis['issues'].append(f"Empty table: {table}")

    except Exception as e:
        analysis['issues'].append(f"Error analyzing ontology: {e}")

    conn.close()
    return analysis

def generate_report(analyses: Dict[str, Dict[str, Any]]) -> str:
    """Generate comprehensive report"""
    report = []
    report.append("=" * 80)
    report.append("VISIONFLOW DATABASE INTEGRITY REPORT")
    report.append("=" * 80)
    report.append("")

    # Settings Database
    report.append("### 1. SETTINGS DATABASE (settings.db)")
    report.append("-" * 80)
    settings_analysis = analyses['settings']
    report.append(f"Total Settings: {settings_analysis.get('total_settings', 0)}")
    report.append(f"Total Records: {settings_analysis['total_records']}")
    report.append(f"Tables: {len(settings_analysis['tables'])}")
    report.append("")

    report.append("Tables:")
    for table, info in settings_analysis['tables'].items():
        report.append(f"  - {table}: {info['record_count']} records")
        report.append(f"    Columns: {', '.join(info['columns'])}")
    report.append("")

    report.append("Credential-Related Settings:")
    if settings_analysis['credentials']:
        for key, value in settings_analysis['credentials']:
            masked_value = "***REDACTED***" if value and len(value) > 0 else "EMPTY/MISSING"
            report.append(f"  - {key}: {masked_value}")
    else:
        report.append("  No credential settings found")
    report.append("")

    if settings_analysis['issues']:
        report.append("⚠️  ISSUES:")
        for issue in settings_analysis['issues']:
            report.append(f"  - {issue}")
    else:
        report.append("✓ No issues detected")
    report.append("")

    # Knowledge Graph Database
    report.append("### 2. KNOWLEDGE GRAPH DATABASE (knowledge_graph.db)")
    report.append("-" * 80)
    kg_analysis = analyses['knowledge_graph']
    report.append(f"Node Count: {kg_analysis.get('node_count', 0)} (expected: 185)")
    report.append(f"Edge Count: {kg_analysis.get('edge_count', 0)} (expected: 4014)")
    report.append(f"Total Records: {kg_analysis['total_records']}")
    report.append("")

    if 'node_type_distribution' in kg_analysis:
        report.append("Node Type Distribution:")
        for node_type, count in kg_analysis['node_type_distribution'].items():
            report.append(f"  - {node_type}: {count}")
    report.append("")

    if 'edge_type_distribution' in kg_analysis:
        report.append("Edge Type Distribution:")
        for edge_type, count in kg_analysis['edge_type_distribution'].items():
            report.append(f"  - {edge_type}: {count}")
    report.append("")

    if kg_analysis['issues']:
        report.append("⚠️  ISSUES:")
        for issue in kg_analysis['issues']:
            report.append(f"  - {issue}")
    else:
        report.append("✓ No issues detected")
    report.append("")

    # Ontology Database
    report.append("### 3. ONTOLOGY DATABASE (ontology.db)")
    report.append("-" * 80)
    onto_analysis = analyses['ontology']
    report.append(f"Total Records: {onto_analysis['total_records']}")
    report.append(f"Tables: {len(onto_analysis['tables'])}")
    report.append("")

    report.append("Tables:")
    for table, info in onto_analysis['tables'].items():
        report.append(f"  - {table}: {info['record_count']} records")
        report.append(f"    Columns: {', '.join(info['columns'])}")
    report.append("")

    if onto_analysis['issues']:
        report.append("⚠️  ISSUES:")
        for issue in onto_analysis['issues']:
            report.append(f"  - {issue}")
    else:
        report.append("✓ No issues detected")
    report.append("")

    # Summary
    report.append("=" * 80)
    report.append("SUMMARY & RECOMMENDATIONS")
    report.append("=" * 80)

    all_issues = []
    all_issues.extend(settings_analysis['issues'])
    all_issues.extend(kg_analysis['issues'])
    all_issues.extend(onto_analysis['issues'])

    if all_issues:
        report.append(f"⚠️  Total Issues Found: {len(all_issues)}")
        report.append("")
        report.append("Recommendations:")

        if any('Empty credential' in issue for issue in all_issues):
            report.append("  1. Mock missing credentials (RAGFlow, Nostr, GitHub)")

        if any('count mismatch' in issue for issue in all_issues):
            report.append("  2. Investigate knowledge graph data loss")
            report.append("     - Check database integrity")
            report.append("     - Review import/migration logs")
            report.append("     - Consider re-importing from source")

        if any('orphaned nodes' in issue for issue in all_issues):
            report.append("  3. Clean up orphaned nodes or create missing edges")

        if any('Empty table' in issue for issue in all_issues):
            report.append("  4. Populate empty ontology tables or remove if unused")
    else:
        report.append("✓ All databases appear healthy!")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)

def main():
    """Main analysis function"""
    base_path = Path(__file__).parent

    analyses = {
        'settings': analyze_settings_db(str(base_path / 'settings.db')),
        'knowledge_graph': analyze_knowledge_graph_db(str(base_path / 'knowledge_graph.db')),
        'ontology': analyze_ontology_db(str(base_path / 'ontology.db'))
    }

    # Generate report
    report = generate_report(analyses)
    print(report)

    # Save detailed JSON output
    output_file = base_path / 'database_analysis_full.json'
    with open(output_file, 'w') as f:
        json.dump(analyses, f, indent=2, default=str)

    print(f"\nDetailed JSON analysis saved to: {output_file}")

if __name__ == '__main__':
    main()
