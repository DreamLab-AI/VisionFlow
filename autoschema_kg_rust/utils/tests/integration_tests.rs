//! Integration tests for the utils crate

use utils::*;
use std::collections::HashMap;
use tempfile::{NamedTempFile, TempDir};
use std::io::Write;
use tokio_test;

#[test]
fn test_full_csv_to_graph_pipeline() {
    // Create test CSV
    let mut csv_file = NamedTempFile::new().unwrap();
    writeln!(csv_file, \"id,name,age,city\").unwrap();
    writeln!(csv_file, \"1,Alice,30,New York\").unwrap();
    writeln!(csv_file, \"2,Bob,25,Los Angeles\").unwrap();
    writeln!(csv_file, \"3,Charlie,35,Chicago\").unwrap();

    let graphml_file = NamedTempFile::new().unwrap();
    let csv_config = csv_processing::CsvConfig::default();

    // Convert CSV to GraphML
    csv_processing::csv_to_graphml(
        csv_file.path(),
        graphml_file.path(),
        &csv_config,
        Some(\"id\"),
        None,
    ).unwrap();

    // Verify GraphML was created and has content
    let content = std::fs::read_to_string(graphml_file.path()).unwrap();
    assert!(content.contains(\"<graphml\"));
    assert!(content.contains(\"Alice\"));
    assert!(content.contains(\"node\"));
}

#[test]
fn test_json_to_csv_to_hash_pipeline() {
    // Create test JSON
    let json_data = serde_json::json!([
        {\"id\": 1, \"name\": \"Alice\", \"details\": {\"age\": 30, \"city\": \"NYC\"}},
        {\"id\": 2, \"name\": \"Bob\", \"details\": {\"age\": 25, \"city\": \"LA\"}}
    ]);

    let mut json_file = NamedTempFile::new().unwrap();
    write!(json_file, \"{}\", serde_json::to_string(&json_data).unwrap()).unwrap();

    let csv_file = NamedTempFile::new().unwrap();
    let json_config = json_processing::JsonConfig::default();

    // Convert JSON to CSV
    let stats = json_processing::json_to_csv(
        json_file.path(),
        csv_file.path(),
        &json_config,
    ).unwrap();

    assert_eq!(stats.total_arrays, 1);
    assert!(stats.unique_paths > 0);

    // Read CSV and generate hashes for each row
    let csv_content = std::fs::read_to_string(csv_file.path()).unwrap();
    let lines: Vec<&str> = csv_content.lines().collect();

    let hash_config = hash_utils::HashConfig::default();
    let mut hashes = Vec::new();

    for line in &lines[1..] { // Skip header
        let hash_id = hash_utils::generate_hash_id(line, &hash_config).unwrap();
        hashes.push(hash_id);
    }

    assert_eq!(hashes.len(), 2);
    assert_ne!(hashes[0].hash, hashes[1].hash);
}

#[test]
fn test_markdown_to_structured_data_pipeline() {
    let markdown_content = r#\"---
title: Test Document
author: Test Author
---

# Main Title

## Section 1

This is the first section with some content.

```rust
fn hello() {
    println!(\"Hello, world!\");
}
```

## Section 2

This is the second section.

| Name | Value |
|------|-------|
| Item1 | 100   |
| Item2 | 200   |
\"#;

    let config = markdown_processing::MarkdownConfig::default();
    let document = markdown_processing::parse_markdown_content(markdown_content, &config).unwrap();

    // Verify parsing
    assert_eq!(document.title, Some(\"Main Title\".to_string()));
    assert_eq!(document.sections.len(), 2);
    assert_eq!(document.code_blocks.len(), 1);
    assert_eq!(document.tables.len(), 1);
    assert!(document.metadata.contains_key(\"title\"));

    // Convert to JSON
    let json_file = NamedTempFile::new().unwrap();
    markdown_processing::markdown_document_to_json_file(&document, json_file.path(), true).unwrap();

    // Verify JSON was created
    let json_content = std::fs::read_to_string(json_file.path()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    assert!(parsed[\"title\"].as_str().unwrap() == \"Main Title\");
}

#[test]
fn test_text_cleaning_pipeline() {
    let dirty_text = r#\"
    <div>Hello <b>World</b>!</div>

    Visit https://example.com or email us at test@example.com

    Call us at (555) 123-4567 for more info.

    \"Smart quotes\" and 'other quotes' need normalization.

    Multiple     spaces    and    tabs		should be cleaned.
    \"#;

    let mut config = text_cleaning::TextCleaningConfig::default();
    config.remove_html = true;
    config.remove_urls = true;
    config.remove_emails = true;
    config.remove_phone_numbers = true;
    config.normalize_whitespace = true;

    let (cleaned, stats) = text_cleaning::clean_text(dirty_text, &config).unwrap();

    assert!(stats.html_tags_removed > 0);
    assert!(stats.urls_removed > 0);
    assert!(stats.emails_removed > 0);
    assert!(stats.reduction_ratio > 0.0);
    assert!(!cleaned.contains(\"<div>\"));
    assert!(!cleaned.contains(\"https://\"));
    assert!(!cleaned.contains(\"@example.com\"));

    // Generate hash for cleaned text
    let hash_config = hash_utils::HashConfig::default();
    let hash_id = hash_utils::generate_hash_id(&cleaned, &hash_config).unwrap();
    assert!(!hash_id.hash.is_empty());
}

#[test]
fn test_large_file_streaming() {
    let temp_dir = TempDir::new().unwrap();
    let large_file_path = temp_dir.path().join(\"large_test.txt\");

    // Create a large test file
    {
        let mut file = std::fs::File::create(&large_file_path).unwrap();
        for i in 0..10000 {
            writeln!(file, \"Line {} with some content for testing streaming operations\", i).unwrap();
        }
    }

    let config = file_io::FileIOConfig {
        buffer_size: 4096,
        chunk_size: 8192,
        ..Default::default()
    };

    // Test streaming reader
    let mut reader = file_io::StreamingReader::new(&large_file_path, config.clone()).unwrap();
    let mut total_lines = 0;

    let stats = reader.read_lines_chunked(|lines| {
        total_lines += lines.len();
        // Process each chunk
        for line in lines {
            assert!(line.contains(\"Line\"));
        }
        Ok(())
    }).unwrap();

    assert_eq!(stats.lines_processed, 10000);
    assert_eq!(total_lines, 10000);

    // Test file info
    let file_info = file_io::get_file_info(&large_file_path).unwrap();
    assert!(file_info.size > 0);
    assert!(file_info.estimated_lines.unwrap_or(0) > 0);
}

#[test]
fn test_graph_analysis_pipeline() {
    let mut graph = graph_conversion::Graph::new(false);

    // Create a small graph
    for i in 0..10 {
        let mut attributes = indexmap::IndexMap::new();
        attributes.insert(\"type\".to_string(), \"person\".to_string());
        attributes.insert(\"age\".to_string(), (20 + i).to_string());

        graph.add_node(graph_conversion::GraphNode {
            id: format!(\"person_{}\", i),
            label: Some(format!(\"Person {}\", i)),
            attributes,
            node_type: Some(\"person\".to_string()),
        }).unwrap();
    }

    // Add some edges to create connections
    for i in 0..5 {
        let mut attributes = indexmap::IndexMap::new();
        attributes.insert(\"relationship\".to_string(), \"friend\".to_string());

        graph.add_edge(graph_conversion::GraphEdge {
            source: format!(\"person_{}\", i),
            target: format!(\"person_{}\", i + 1),
            label: Some(\"friend\".to_string()),
            weight: Some(1.0),
            attributes,
            edge_type: Some(\"friendship\".to_string()),
        }).unwrap();
    }

    // Analyze graph
    let stats = graph.calculate_stats();
    assert_eq!(stats.node_count, 10);
    assert_eq!(stats.edge_count, 5);
    assert!(stats.density > 0.0);

    // Convert to adjacency matrix
    let (node_ids, matrix) = graph_conversion::graph_to_adjacency_matrix(&graph);
    assert_eq!(node_ids.len(), 10);
    assert_eq!(matrix.len(), 10);
    assert_eq!(matrix[0].len(), 10);

    // Convert back to graph
    let reconstructed = graph_conversion::adjacency_matrix_to_graph(
        &node_ids,
        &matrix,
        false,
        0.5
    ).unwrap();
    assert_eq!(reconstructed.nodes.len(), 10);
    assert_eq!(reconstructed.edges.len(), 5);

    // Export to GraphML
    let graphml_file = NamedTempFile::new().unwrap();
    let config = graph_conversion::GraphConfig::default();
    graph_conversion::graph_to_graphml(&graph, graphml_file.path(), &config).unwrap();

    // Verify GraphML content
    let content = std::fs::read_to_string(graphml_file.path()).unwrap();
    assert!(content.contains(\"<graphml\"));
    assert!(content.contains(\"person_0\"));
    assert!(content.contains(\"friendship\"));
}

#[test]
fn test_hash_consistency_and_uniqueness() {
    let config = hash_utils::HashConfig::default();
    let test_data = vec![
        \"test string 1\",
        \"test string 2\",
        \"test string 1\", // Duplicate
        \"different string\",
    ];

    let mut hashes = Vec::new();
    let mut hash_to_original = HashMap::new();

    for (i, text) in test_data.iter().enumerate() {
        let hash_id = hash_utils::generate_hash_id(text, &config).unwrap();

        // Check consistency
        let hash_id2 = hash_utils::generate_hash_id(text, &config).unwrap();
        assert_eq!(hash_id.hash, hash_id2.hash);

        // Track for uniqueness check
        hash_to_original.entry(hash_id.hash.clone()).or_insert_with(Vec::new).push(i);
        hashes.push(hash_id);
    }

    // Verify duplicates have same hash
    assert_eq!(hashes[0].hash, hashes[2].hash); // \"test string 1\" appears twice

    // Verify different strings have different hashes
    assert_ne!(hashes[0].hash, hashes[1].hash);
    assert_ne!(hashes[0].hash, hashes[3].hash);
}

#[tokio::test]
async fn test_async_file_processing() {
    let temp_file_path = {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, \"async line 1\").unwrap();
        writeln!(temp_file, \"async line 2\").unwrap();
        writeln!(temp_file, \"async line 3\").unwrap();
        temp_file.path().to_path_buf()
    };

    // Keep temp file alive
    let _temp_file = std::fs::File::open(&temp_file_path).unwrap();

    let config = file_io::FileIOConfig::default();
    let processor = file_io::AsyncFileProcessor::new(config);

    let mut processed_lines = Vec::new();
    let stats = processor.process_lines_async(&temp_file_path, |line| {
        let mut lines = processed_lines.clone();
        async move {
            lines.push(line);
            processed_lines = lines;
            Ok(())
        }
    }).await.unwrap();

    assert_eq!(stats.lines_processed, 3);
}

#[test]
fn test_file_merging_and_concatenation() {
    let temp_dir = TempDir::new().unwrap();

    // Create multiple test files
    let file_paths: Vec<_> = (0..3).map(|i| {
        let file_path = temp_dir.path().join(format!(\"test_{}.txt\", i));
        let mut file = std::fs::File::create(&file_path).unwrap();
        for j in 0..100 {
            writeln!(file, \"File {} Line {}\", i, j).unwrap();
        }
        file_path
    }).collect();

    let output_path = temp_dir.path().join(\"merged.txt\");
    let config = file_io::FileIOConfig::default();
    let merger = file_io::FileMerger::new(config);

    // Test concatenation
    let stats = merger.concatenate_files(&file_paths, &output_path).unwrap();
    assert_eq!(stats.chunks_processed, 3);
    assert_eq!(stats.lines_processed, 300); // 3 files * 100 lines each

    // Verify merged content
    let content = std::fs::read_to_string(&output_path).unwrap();
    assert!(content.contains(\"File 0 Line 0\"));
    assert!(content.contains(\"File 2 Line 99\"));

    let line_count = content.lines().count();
    assert_eq!(line_count, 300);
}

#[test]
fn test_end_to_end_data_pipeline() {
    // This test demonstrates a complete data processing pipeline

    // 1. Start with JSON data
    let json_data = serde_json::json!([
        {\"id\": 1, \"name\": \"Alice Smith\", \"age\": 30, \"email\": \"alice@example.com\"},
        {\"id\": 2, \"name\": \"Bob Johnson\", \"age\": 25, \"email\": \"bob@example.com\"},
        {\"id\": 3, \"name\": \"Charlie Brown\", \"age\": 35, \"email\": \"charlie@example.com\"}
    ]);

    let temp_dir = TempDir::new().unwrap();
    let json_file = temp_dir.path().join(\"input.json\");
    std::fs::write(&json_file, serde_json::to_string_pretty(&json_data).unwrap()).unwrap();

    // 2. Convert JSON to CSV
    let csv_file = temp_dir.path().join(\"converted.csv\");
    let json_config = json_processing::JsonConfig::default();
    json_processing::json_to_csv(&json_file, &csv_file, &json_config).unwrap();

    // 3. Add numeric IDs to CSV
    let csv_with_ids = temp_dir.path().join(\"with_ids.csv\");
    let csv_config = csv_processing::CsvConfig::default();
    csv_processing::csv_add_numeric_id(
        &csv_file,
        &csv_with_ids,
        &csv_config,
        \"numeric_id\",
        1000
    ).unwrap();

    // 4. Clean text data from CSV
    let csv_content = std::fs::read_to_string(&csv_with_ids).unwrap();
    let mut text_config = text_cleaning::TextCleaningConfig::default();
    text_config.remove_emails = true;
    text_config.normalize_whitespace = true;

    let (cleaned_content, _) = text_cleaning::clean_text(&csv_content, &text_config).unwrap();

    // 5. Generate hashes for cleaned data
    let hash_config = hash_utils::HashConfig::default();
    let content_hash = hash_utils::generate_hash_id(&cleaned_content, &hash_config).unwrap();

    // 6. Convert to graph format
    let graphml_file = temp_dir.path().join(\"output.graphml\");
    csv_processing::csv_to_graphml(
        &csv_with_ids,
        &graphml_file,
        &csv_config,
        Some(\"numeric_id\"),
        None
    ).unwrap();

    // Verify the pipeline worked
    assert!(json_file.exists());
    assert!(csv_file.exists());
    assert!(csv_with_ids.exists());
    assert!(graphml_file.exists());
    assert!(!content_hash.hash.is_empty());

    // Verify GraphML contains expected data
    let graphml_content = std::fs::read_to_string(&graphml_file).unwrap();
    assert!(graphml_content.contains(\"<node\"));
    assert!(graphml_content.contains(\"1000\")); // First numeric ID
}