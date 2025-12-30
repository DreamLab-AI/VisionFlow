#!/usr/bin/env python3
"""
VisionFlow Strategic Analysis - Using Wardley Mapping Tools
"""

import sys
sys.path.insert(0, '/home/devuser/workspace/project/multi-agent-docker/skills/wardley-maps/tools')

from strategic_analyzer import StrategicAnalyzer, analyze_wardley_map

# Components for strategic analysis
components = [
    {'name': 'R&D Insight Discovery', 'visibility': 1.0, 'evolution': 0.50, 'type': 'user'},
    {'name': 'Web Interface', 'visibility': 0.95, 'evolution': 0.70, 'type': 'product'},
    {'name': 'XR Visualization', 'visibility': 0.85, 'evolution': 0.15, 'type': 'genesis'},
    {'name': 'Real-time Collaboration', 'visibility': 0.85, 'evolution': 0.65, 'type': 'product'},
    {'name': 'Semantic Physics Engine', 'visibility': 0.70, 'evolution': 0.25, 'type': 'custom'},
    {'name': 'Knowledge Graph Core', 'visibility': 0.65, 'evolution': 0.40, 'type': 'custom'},
    {'name': 'Multi-Agent Orchestration', 'visibility': 0.60, 'evolution': 0.30, 'type': 'custom'},
    {'name': 'Binary Protocol', 'visibility': 0.55, 'evolution': 0.50, 'type': 'product'},
    {'name': 'GraphRAG System', 'visibility': 0.55, 'evolution': 0.35, 'type': 'custom'},
    {'name': 'Solid/LDP Pods', 'visibility': 0.60, 'evolution': 0.20, 'type': 'genesis'},
    {'name': 'Domain Ontologies', 'visibility': 0.55, 'evolution': 0.30, 'type': 'custom'},
    {'name': 'Agent Memory', 'visibility': 0.50, 'evolution': 0.20, 'type': 'genesis'},
    {'name': 'Broadcast Optimizer', 'visibility': 0.45, 'evolution': 0.45, 'type': 'product'},
    {'name': 'LLM Services', 'visibility': 0.50, 'evolution': 0.45, 'type': 'product'},
    {'name': 'Task Orchestrator', 'visibility': 0.45, 'evolution': 0.55, 'type': 'product'},
    {'name': 'GPU Memory Manager', 'visibility': 0.35, 'evolution': 0.40, 'type': 'custom'},
    {'name': 'Neo4j Database', 'visibility': 0.30, 'evolution': 0.75, 'type': 'product'},
    {'name': 'CUDA Compute', 'visibility': 0.15, 'evolution': 0.85, 'type': 'commodity'},
    {'name': 'Actix Web', 'visibility': 0.20, 'evolution': 0.80, 'type': 'commodity'},
]

dependencies = [
    ('R&D Insight Discovery', 'Web Interface'),
    ('R&D Insight Discovery', 'XR Visualization'),
    ('Web Interface', 'Real-time Collaboration'),
    ('Web Interface', 'Knowledge Graph Core'),
    ('Web Interface', 'Binary Protocol'),
    ('XR Visualization', 'Semantic Physics Engine'),
    ('XR Visualization', 'Real-time Collaboration'),
    ('Semantic Physics Engine', 'CUDA Compute'),
    ('Semantic Physics Engine', 'Domain Ontologies'),
    ('Semantic Physics Engine', 'GPU Memory Manager'),
    ('Knowledge Graph Core', 'Neo4j Database'),
    ('Knowledge Graph Core', 'Domain Ontologies'),
    ('Knowledge Graph Core', 'Multi-Agent Orchestration'),
    ('Knowledge Graph Core', 'Solid/LDP Pods'),
    ('Multi-Agent Orchestration', 'Task Orchestrator'),
    ('Multi-Agent Orchestration', 'Agent Memory'),
    ('Multi-Agent Orchestration', 'LLM Services'),
    ('GraphRAG System', 'Knowledge Graph Core'),
    ('GraphRAG System', 'LLM Services'),
    ('LLM Services', 'CUDA Compute'),
]

# Run strategic analysis
try:
    analysis = analyze_wardley_map(components, dependencies)

    # Generate markdown report
    markdown_report = StrategicAnalyzer.export_analysis_to_markdown(analysis)

    # Save report
    with open('/home/devuser/workspace/project/docs/VISIONFLOW_STRATEGIC_REPORT.md', 'w') as f:
        f.write(markdown_report)

    print("Strategic Analysis Complete!")
    print(f"Total Components: {analysis.total_components}")
    print(f"Total Dependencies: {analysis.total_dependencies}")
    print(f"Insights Generated: {len(analysis.insights)}")
    print(f"\nCompetitive Advantages: {len(analysis.competitive_advantages)}")
    for adv in analysis.competitive_advantages[:3]:
        print(f"  - {adv}")
    print(f"\nVulnerabilities: {len(analysis.vulnerabilities)}")
    for vuln in analysis.vulnerabilities[:3]:
        print(f"  - {vuln}")
    print(f"\nOpportunities: {len(analysis.opportunities)}")
    for opp in analysis.opportunities[:3]:
        print(f"  - {opp}")
    print(f"\nReport saved to: /home/devuser/workspace/project/docs/VISIONFLOW_STRATEGIC_REPORT.md")

except Exception as e:
    print(f"Analysis error: {e}")
    import traceback
    traceback.print_exc()
