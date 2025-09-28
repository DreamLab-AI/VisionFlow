//! Mock data generators for testing

use fake::{Fake, Faker};
use fake::faker::internet::en::SafeEmail;
use fake::faker::lorem::en::{Sentence, Paragraph};
use fake::faker::name::en::Name;
use serde_json::{Value, json};
use std::collections::HashMap;
use uuid::Uuid;

/// Generate random knowledge graph node
pub fn generate_kg_node() -> serde_json::Value {
    json!({
        "id": Uuid::new_v4().to_string(),
        "label": Name().fake::<String>(),
        "type": "Entity",
        "properties": {
            "name": Name().fake::<String>(),
            "description": Paragraph(3..5).fake::<String>(),
            "created_at": chrono::Utc::now().to_rfc3339(),
            "confidence": (0.5..1.0).fake::<f64>()
        }
    })
}

/// Generate random knowledge graph relationship
pub fn generate_kg_relationship(from_id: &str, to_id: &str) -> serde_json::Value {
    let relation_types = ["RELATED_TO", "CONTAINS", "PART_OF", "SIMILAR_TO", "DERIVED_FROM"];
    json!({
        "id": Uuid::new_v4().to_string(),
        "from": from_id,
        "to": to_id,
        "type": relation_types.choose(&mut rand::thread_rng()).unwrap(),
        "properties": {
            "weight": (0.1..1.0).fake::<f64>(),
            "created_at": chrono::Utc::now().to_rfc3339()
        }
    })
}

/// Generate sample CSV data
pub fn generate_csv_data(rows: usize) -> String {
    let mut csv = String::from("id,name,email,age,city\n");
    for i in 0..rows {
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            i,
            Name().fake::<String>(),
            SafeEmail().fake::<String>(),
            (18..80).fake::<u8>(),
            Name().fake::<String>()
        ));
    }
    csv
}

/// Generate sample JSON data
pub fn generate_json_data(count: usize) -> Vec<Value> {
    (0..count)
        .map(|_| json!({
            "id": Uuid::new_v4().to_string(),
            "title": Sentence(3..8).fake::<String>(),
            "content": Paragraph(5..10).fake::<String>(),
            "metadata": {
                "author": Name().fake::<String>(),
                "tags": (0..5).map(|_| Name().fake::<String>()).collect::<Vec<_>>(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }
        }))
        .collect()
}

/// Generate sample markdown content
pub fn generate_markdown_content() -> String {
    format!(
        "# {}\n\n{}\n\n## {}\n\n{}\n\n- {}\n- {}\n- {}",
        Sentence(2..4).fake::<String>(),
        Paragraph(3..5).fake::<String>(),
        Sentence(2..4).fake::<String>(),
        Paragraph(3..5).fake::<String>(),
        Sentence(1..3).fake::<String>(),
        Sentence(1..3).fake::<String>(),
        Sentence(1..3).fake::<String>()
    )
}

/// Generate vector embeddings for testing
pub fn generate_embedding(dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|_| (-1.0..1.0).fake::<f32>())
        .collect()
}

/// Generate batch of embeddings
pub fn generate_embeddings_batch(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| generate_embedding(dimensions))
        .collect()
}

/// Generate sample query for testing retrieval
pub fn generate_query() -> String {
    Sentence(5..15).fake::<String>()
}

/// Generate LLM response for testing
pub fn generate_llm_response() -> serde_json::Value {
    json!({
        "id": Uuid::new_v4().to_string(),
        "object": "text_completion",
        "created": chrono::Utc::now().timestamp(),
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": Paragraph(3..8).fake::<String>()
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": (50..200).fake::<u32>(),
            "completion_tokens": (100..500).fake::<u32>(),
            "total_tokens": (150..700).fake::<u32>()
        }
    })
}

/// Generate sample file paths for testing
pub fn generate_file_paths(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("test_file_{}.{}", i, ["txt", "json", "csv", "md"].choose(&mut rand::thread_rng()).unwrap()))
        .collect()
}

use rand::seq::SliceRandom;

/// Generate sample graph data for Neo4j testing
pub fn generate_neo4j_graph(node_count: usize, edge_ratio: f64) -> (Vec<Value>, Vec<Value>) {
    let nodes: Vec<Value> = (0..node_count)
        .map(|_| generate_kg_node())
        .collect();

    let mut edges = Vec::new();
    let edge_count = (node_count as f64 * edge_ratio) as usize;

    for _ in 0..edge_count {
        let from_idx = (0..node_count).fake::<usize>();
        let to_idx = (0..node_count).fake::<usize>();

        if from_idx != to_idx {
            let from_id = nodes[from_idx]["id"].as_str().unwrap();
            let to_id = nodes[to_idx]["id"].as_str().unwrap();
            edges.push(generate_kg_relationship(from_id, to_id));
        }
    }

    (nodes, edges)
}