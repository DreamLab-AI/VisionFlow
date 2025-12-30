#!/usr/bin/env python3
"""
VisionFlow Comprehensive Wardley Map Generator
Based on swarm analysis of: Backend, Frontend, GPU/CUDA, Multi-Agent, Knowledge Graph
"""

import sys
import os
sys.path.insert(0, '/home/devuser/workspace/project/multi-agent-docker/skills/wardley-maps/tools')

from interactive_map_generator import InteractiveMapGenerator

# VisionFlow Components - Based on Swarm Analysis
components = [
    # User Needs Layer (Visibility: 1.0)
    {'name': 'R&D Insight Discovery', 'visibility': 1.0, 'evolution': 0.50,
     'category': 'User Need', 'description': 'Core user need for knowledge discovery'},

    # Interface Layer (Visibility: 0.85-0.95)
    {'name': 'Web Interface', 'visibility': 0.95, 'evolution': 0.70,
     'category': 'Product', 'description': 'React/Three.js frontend with 14 subsystems, 85K LOC'},
    {'name': 'XR Visualization', 'visibility': 0.85, 'evolution': 0.15,
     'category': 'Genesis', 'description': 'Quest 3/WebXR immersive knowledge work - first mover advantage'},
    {'name': 'Real-time Collaboration', 'visibility': 0.85, 'evolution': 0.65,
     'category': 'Product', 'description': 'Multi-WebSocket strategy with Vircadia integration'},

    # Application Layer (Visibility: 0.60-0.75)
    {'name': 'Semantic Physics Engine', 'visibility': 0.70, 'evolution': 0.25,
     'category': 'Custom', 'description': '13 CUDA kernels, OWL-to-force translation, 9 force types'},
    {'name': 'Knowledge Graph Core', 'visibility': 0.65, 'evolution': 0.40,
     'category': 'Custom', 'description': 'Neo4j + Whelk-rs OWL 2 EL reasoning (10-100x faster)'},
    {'name': 'Multi-Agent Orchestration', 'visibility': 0.60, 'evolution': 0.30,
     'category': 'Custom', 'description': '10 agent types, Actix actors, Queen/Worker topology'},
    {'name': 'Binary Protocol', 'visibility': 0.55, 'evolution': 0.50,
     'category': 'Product', 'description': '28-byte node format, 80% bandwidth reduction, 20Hz broadcast'},

    # Service Layer (Visibility: 0.45-0.60)
    {'name': 'GraphRAG System', 'visibility': 0.55, 'evolution': 0.35,
     'category': 'Custom', 'description': 'Microsoft GraphRAG with Leiden clustering, multi-hop reasoning'},
    {'name': 'Solid/LDP Pods', 'visibility': 0.60, 'evolution': 0.20,
     'category': 'Genesis', 'description': 'Decentralized data pods, privacy-first differentiation'},
    {'name': 'Domain Ontologies', 'visibility': 0.55, 'evolution': 0.30,
     'category': 'Custom', 'description': 'OWL 2 EL with SubClassOf attraction, DisjointWith repulsion'},
    {'name': 'Agent Memory', 'visibility': 0.50, 'evolution': 0.20,
     'category': 'Genesis', 'description': 'JSON-LD with PROV ontology: episodic, semantic, procedural'},
    {'name': 'Broadcast Optimizer', 'visibility': 0.45, 'evolution': 0.45,
     'category': 'Product', 'description': 'Delta compression, spatial culling, 58% bandwidth savings'},

    # Platform Layer (Visibility: 0.30-0.50)
    {'name': 'LLM Services', 'visibility': 0.50, 'evolution': 0.45,
     'category': 'Product', 'description': 'Z.AI/Anthropic API, worker pool, 30s timeout'},
    {'name': 'Task Orchestrator', 'visibility': 0.45, 'evolution': 0.55,
     'category': 'Product', 'description': 'Actix TaskOrchestratorActor, retry logic, HTTP integration'},
    {'name': 'GPU Memory Manager', 'visibility': 0.35, 'evolution': 0.40,
     'category': 'Custom', 'description': 'Pool-based allocation, leak detection, async transfers'},

    # Infrastructure Layer (Visibility: 0.10-0.35)
    {'name': 'Neo4j Database', 'visibility': 0.30, 'evolution': 0.75,
     'category': 'Product', 'description': 'Enterprise graph database, ACID, proven scale'},
    {'name': 'CUDA Compute', 'visibility': 0.15, 'evolution': 0.85,
     'category': 'Commodity', 'description': 'CUDA 12.0 via cudarc, RTX A6000/Quadro support'},
    {'name': 'Actix Web', 'visibility': 0.20, 'evolution': 0.80,
     'category': 'Commodity', 'description': 'Rust async web framework, WebSocket support'},
]

# Dependencies based on swarm analysis
dependencies = [
    # User Need connections
    ('R&D Insight Discovery', 'Web Interface'),
    ('R&D Insight Discovery', 'XR Visualization'),

    # Interface dependencies
    ('Web Interface', 'Real-time Collaboration'),
    ('Web Interface', 'Knowledge Graph Core'),
    ('Web Interface', 'Binary Protocol'),
    ('XR Visualization', 'Semantic Physics Engine'),
    ('XR Visualization', 'Real-time Collaboration'),
    ('Real-time Collaboration', 'Binary Protocol'),

    # Application dependencies
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
    ('Binary Protocol', 'Broadcast Optimizer'),

    # Service dependencies
    ('GraphRAG System', 'Knowledge Graph Core'),
    ('GraphRAG System', 'LLM Services'),
    ('GraphRAG System', 'Neo4j Database'),
    ('Domain Ontologies', 'Neo4j Database'),
    ('Broadcast Optimizer', 'Actix Web'),

    # Platform dependencies
    ('Task Orchestrator', 'Actix Web'),
    ('LLM Services', 'CUDA Compute'),
    ('GPU Memory Manager', 'CUDA Compute'),
]

# Strategic insights from swarm analysis
strategic_insights = {
    'competitive_advantages': [
        'Semantic Physics Engine',   # OWL-to-GPU unique
        'XR Visualization',          # Quest 3 first mover
        'Multi-Agent Orchestration', # Turbo Flow
        'Domain Ontologies',         # Whelk-rs 10-100x faster
    ],
    'vulnerabilities': [
        'CUDA Compute',              # Limits deployment options
        'Knowledge Graph Core',      # Dual database complexity
        'Binary Protocol',           # Protocol complexity
    ],
    'opportunities': [
        'GraphRAG System',           # Commoditize as API product
        'Solid/LDP Pods',            # Privacy-first differentiation
        'Agent Memory',              # Persistent learning patterns
    ],
    'threats': [
        'LLM Services',              # Vendor lock-in risk
        'CUDA Compute',              # Hardware dependency
    ]
}

# Generate the map
generator = InteractiveMapGenerator(width=1400, height=900)
html_map = generator.create_interactive_map(components, dependencies, strategic_insights)

# Save the map
output_path = '/home/devuser/workspace/project/docs/visionflow_wardley_map.html'
with open(output_path, 'w') as f:
    f.write(html_map)

print(f"VisionFlow Wardley Map generated: {output_path}")
print(f"Components: {len(components)}")
print(f"Dependencies: {len(dependencies)}")
print(f"Strategic advantages: {len(strategic_insights['competitive_advantages'])}")
print(f"Vulnerabilities: {len(strategic_insights['vulnerabilities'])}")
