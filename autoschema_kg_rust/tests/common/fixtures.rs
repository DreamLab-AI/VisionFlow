//! Test fixtures with sample data

use serde_json::json;
use std::collections::HashMap;

/// Sample CSV data for testing
pub const SAMPLE_CSV: &str = r#"id,name,email,department,salary
1,John Doe,john@example.com,Engineering,75000
2,Jane Smith,jane@example.com,Marketing,65000
3,Bob Johnson,bob@example.com,Sales,55000
4,Alice Brown,alice@example.com,HR,60000
5,Charlie Wilson,charlie@example.com,Engineering,80000
"#;

/// Sample JSON data for testing
pub fn sample_json_data() -> serde_json::Value {
    json!([
        {
            "id": "doc1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "metadata": {
                "author": "Dr. Smith",
                "category": "AI",
                "tags": ["ml", "ai", "tutorial"],
                "published": "2023-01-15T10:00:00Z"
            }
        },
        {
            "id": "doc2",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns.",
            "metadata": {
                "author": "Prof. Johnson",
                "category": "AI",
                "tags": ["deep-learning", "neural-networks", "ai"],
                "published": "2023-02-20T14:30:00Z"
            }
        },
        {
            "id": "doc3",
            "title": "Natural Language Processing",
            "content": "NLP combines computational linguistics with machine learning to help computers understand human language.",
            "metadata": {
                "author": "Dr. Brown",
                "category": "NLP",
                "tags": ["nlp", "linguistics", "text-processing"],
                "published": "2023-03-10T09:15:00Z"
            }
        }
    ])
}

/// Sample markdown content
pub const SAMPLE_MARKDOWN: &str = r#"# Knowledge Graph Construction

## Overview
This document describes the process of building knowledge graphs from unstructured data.

### Key Components
- **Entity Extraction**: Identifying named entities in text
- **Relationship Mining**: Finding connections between entities
- **Graph Storage**: Storing the graph in Neo4j

## Process Steps

1. **Data Ingestion**
   - Load documents from various sources
   - Parse different file formats

2. **Text Processing**
   - Clean and normalize text
   - Tokenize and segment

3. **Knowledge Extraction**
   - Extract entities using NER
   - Identify relationships
   - Resolve entity references

4. **Graph Construction**
   - Create nodes and edges
   - Store in graph database
   - Index for fast retrieval

## Example Entity Types
- Person
- Organization
- Location
- Concept
- Event

## Relationship Types
- WORKS_FOR
- LOCATED_IN
- RELATED_TO
- PART_OF
- HAPPENED_AT
"#;

/// Sample knowledge graph nodes
pub fn sample_kg_nodes() -> Vec<serde_json::Value> {
    vec![
        json!({
            "id": "person_1",
            "label": "Person",
            "properties": {
                "name": "John Smith",
                "age": 35,
                "occupation": "Data Scientist"
            }
        }),
        json!({
            "id": "org_1",
            "label": "Organization",
            "properties": {
                "name": "TechCorp Inc.",
                "industry": "Technology",
                "founded": 2010
            }
        }),
        json!({
            "id": "concept_1",
            "label": "Concept",
            "properties": {
                "name": "Machine Learning",
                "definition": "A subset of AI that enables computers to learn from data"
            }
        })
    ]
}

/// Sample knowledge graph relationships
pub fn sample_kg_relationships() -> Vec<serde_json::Value> {
    vec![
        json!({
            "id": "rel_1",
            "from": "person_1",
            "to": "org_1",
            "type": "WORKS_FOR",
            "properties": {
                "since": "2020-01-01",
                "position": "Senior Data Scientist"
            }
        }),
        json!({
            "id": "rel_2",
            "from": "person_1",
            "to": "concept_1",
            "type": "EXPERT_IN",
            "properties": {
                "experience_years": 5,
                "confidence": 0.9
            }
        })
    ]
}

/// Sample vector embeddings
pub fn sample_embeddings() -> HashMap<String, Vec<f32>> {
    let mut embeddings = HashMap::new();

    embeddings.insert("doc1".to_string(), vec![
        0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8
    ]);

    embeddings.insert("doc2".to_string(), vec![
        0.2, 0.1, -0.4, 0.3, -0.6, 0.5, -0.8, 0.7
    ]);

    embeddings.insert("doc3".to_string(), vec![
        0.3, 0.4, -0.1, 0.2, -0.7, 0.8, -0.5, 0.6
    ]);

    embeddings
}

/// Sample queries for testing retrieval
pub fn sample_queries() -> Vec<&'static str> {
    vec![
        "What is machine learning?",
        "Tell me about neural networks",
        "How does NLP work?",
        "Who works at TechCorp?",
        "What are the steps in knowledge graph construction?"
    ]
}

/// Sample configuration for testing
pub fn sample_config() -> serde_json::Value {
    json!({
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "retriever": {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "max_hops": 3
        },
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        },
        "vector_store": {
            "dimensions": 768,
            "index_type": "hnsw",
            "ef_construction": 200,
            "m": 16
        }
    })
}