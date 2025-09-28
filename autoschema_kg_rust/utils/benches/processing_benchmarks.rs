use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use utils::*;
use std::collections::HashMap;
use tempfile::NamedTempFile;
use std::io::Write;

fn benchmark_csv_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group(\"csv_processing\");

    // Create test CSV data
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, \"id,name,age,city\").unwrap();
    for i in 0..10000 {
        writeln!(temp_file, \"{},User{},{},City{}\", i, i, 20 + (i % 50), i % 100).unwrap();
    }

    let config = csv_processing::CsvConfig::default();

    group.bench_function(\"analyze_csv\", |b| {
        b.iter(|| {
            csv_processing::analyze_csv(black_box(temp_file.path()), black_box(&config))
        })
    });

    // Benchmark CSV add numeric ID
    group.bench_function(\"csv_add_numeric_id\", |b| {
        b.iter(|| {
            let output_file = NamedTempFile::new().unwrap();
            csv_processing::csv_add_numeric_id(
                black_box(temp_file.path()),
                black_box(output_file.path()),
                black_box(&config),
                black_box(\"new_id\"),
                black_box(1)
            )
        })
    });

    group.finish();
}

fn benchmark_json_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group(\"json_processing\");

    // Create test JSON data
    let json_data = serde_json::json!([
        {\"id\": 1, \"name\": \"Alice\", \"age\": 30, \"nested\": {\"city\": \"NY\", \"zip\": \"10001\"}},
        {\"id\": 2, \"name\": \"Bob\", \"age\": 25, \"nested\": {\"city\": \"LA\", \"zip\": \"90001\"}},
    ]);

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(temp_file, \"{}\", serde_json::to_string(&json_data).unwrap()).unwrap();

    let config = json_processing::JsonConfig::default();

    group.bench_function(\"json_to_csv\", |b| {
        b.iter(|| {
            let output_file = NamedTempFile::new().unwrap();
            json_processing::json_to_csv(
                black_box(temp_file.path()),
                black_box(output_file.path()),
                black_box(&config)
            )
        })
    });

    // Benchmark JSON repair
    let malformed_json = r#\"{name: 'John', age: 30,}\"#;
    group.bench_function(\"json_repair\", |b| {
        b.iter(|| {
            json_processing::json_repair(black_box(malformed_json), black_box(&config))
        })
    });

    group.finish();
}

fn benchmark_hash_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group(\"hash_generation\");

    let config = hash_utils::HashConfig::default();
    let test_strings = vec![
        \"short\".to_string(),
        \"medium length string for testing\".to_string(),
        \"very long string that contains a lot of text for performance testing of hash generation functions\".repeat(10),
    ];

    for (i, test_string) in test_strings.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new(\"generate_hash_id\", i),
            test_string,
            |b, s| {
                b.iter(|| hash_utils::generate_hash_id(black_box(s), black_box(&config)))
            },
        );
    }

    // Benchmark batch generation
    let batch_inputs: Vec<String> = (0..1000).map(|i| format!(\"test_string_{}\", i)).collect();
    group.bench_function(\"batch_generate_hash_ids\", |b| {
        b.iter(|| {
            hash_utils::batch_generate_hash_ids(black_box(&batch_inputs), black_box(&config))
        })
    });

    // Benchmark structured hash
    let mut data = HashMap::new();
    data.insert(\"key1\".to_string(), \"value1\".to_string());
    data.insert(\"key2\".to_string(), \"value2\".to_string());
    data.insert(\"key3\".to_string(), \"value3\".to_string());

    group.bench_function(\"generate_structured_hash_id\", |b| {
        b.iter(|| {
            hash_utils::generate_structured_hash_id(black_box(&data), black_box(&config))
        })
    });

    group.finish();
}

fn benchmark_text_cleaning(c: &mut Criterion) {
    let mut group = c.benchmark_group(\"text_cleaning\");

    let test_texts = vec![
        \"Simple text\".to_string(),
        \"Text with <b>HTML</b> tags and https://example.com URLs\".to_string(),
        \"Complex text with HTML tags, URLs, emails like user@example.com, and lots of    whitespace\".to_string(),
        include_str!(\"../tests/sample_text.txt\").to_string(), // Large sample text
    ];

    let config = text_cleaning::TextCleaningConfig::default();

    for (i, text) in test_texts.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new(\"clean_text\", i),
            text,
            |b, t| {
                b.iter(|| text_cleaning::clean_text(black_box(t), black_box(&config)))
            },
        );
    }

    // Benchmark batch cleaning
    let batch_texts: Vec<String> = (0..100).map(|i| format!(\"Test text {} with some content\", i)).collect();
    group.bench_function(\"batch_clean_text\", |b| {
        b.iter(|| {
            text_cleaning::batch_clean_text(black_box(&batch_texts), black_box(&config))
        })
    });

    // Benchmark normalization
    let text_with_quotes = \"'Hello' "world" `test`\";
    group.bench_function(\"normalize_quotes\", |b| {
        b.iter(|| text_cleaning::normalize_quotes(black_box(text_with_quotes)))
    });

    group.finish();
}

fn benchmark_graph_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group(\"graph_operations\");

    // Create test graph
    let mut graph = graph_conversion::Graph::new(false);

    // Add nodes
    for i in 0..1000 {
        let mut attributes = indexmap::IndexMap::new();
        attributes.insert(\"label\".to_string(), format!(\"Node {}\", i));

        graph.add_node(graph_conversion::GraphNode {
            id: i.to_string(),
            label: Some(format!(\"Node {}\", i)),
            attributes,
            node_type: Some(\"test\".to_string()),
        }).unwrap();
    }

    // Add edges
    for i in 0..500 {
        let mut attributes = indexmap::IndexMap::new();
        attributes.insert(\"weight\".to_string(), \"1.0\".to_string());

        graph.add_edge(graph_conversion::GraphEdge {
            source: i.to_string(),
            target: (i + 1).to_string(),
            label: None,
            weight: Some(1.0),
            attributes,
            edge_type: Some(\"connects\".to_string()),
        }).unwrap();
    }

    group.bench_function(\"calculate_stats\", |b| {
        b.iter(|| graph.calculate_stats())
    });

    group.bench_function(\"adjacency_matrix_conversion\", |b| {
        b.iter(|| graph_conversion::graph_to_adjacency_matrix(black_box(&graph)))
    });

    let config = graph_conversion::GraphConfig::default();
    group.bench_function(\"graph_to_graphml\", |b| {
        b.iter(|| {
            let output_file = NamedTempFile::new().unwrap();
            graph_conversion::graph_to_graphml(
                black_box(&graph),
                black_box(output_file.path()),
                black_box(&config)
            )
        })
    });

    group.finish();
}

fn benchmark_file_io(c: &mut Criterion) {
    let mut group = c.benchmark_group(\"file_io\");

    // Create test file with multiple lines
    let mut temp_file = NamedTempFile::new().unwrap();
    for i in 0..10000 {
        writeln!(temp_file, \"Line {} with some content for testing file I/O performance\", i).unwrap();
    }

    let config = file_io::FileIOConfig::default();

    group.bench_function(\"streaming_reader_lines\", |b| {
        b.iter(|| {
            let mut reader = file_io::StreamingReader::new(black_box(temp_file.path()), config.clone()).unwrap();
            reader.read_lines_chunked(|_lines| Ok(())).unwrap()
        })
    });

    group.bench_function(\"get_file_info\", |b| {
        b.iter(|| file_io::get_file_info(black_box(temp_file.path())))
    });

    // Benchmark file merging
    let temp_files: Vec<_> = (0..5).map(|i| {
        let mut file = NamedTempFile::new().unwrap();
        for j in 0..1000 {
            writeln!(file, \"File {} Line {}\", i, j).unwrap();
        }
        file
    }).collect();

    let file_paths: Vec<_> = temp_files.iter().map(|f| f.path()).collect();

    group.bench_function(\"concatenate_files\", |b| {
        b.iter(|| {
            let output_file = NamedTempFile::new().unwrap();
            let merger = file_io::FileMerger::new(config.clone());
            merger.concatenate_files(black_box(&file_paths), black_box(output_file.path()))
        })
    });

    group.finish();
}

fn benchmark_markdown_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group(\"markdown_processing\");

    let markdown_content = r#\"# Main Title

## Introduction

This is a paragraph with some **bold** and *italic* text.

### Code Example

```rust
fn main() {
    println!(\"Hello, world!\");
}
```

## Table

| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |

## Links and Images

Check out [Rust](https://rust-lang.org) for more info.

![Logo](https://example.com/logo.png)
\"#;

    let config = markdown_processing::MarkdownConfig::default();

    group.bench_function(\"parse_markdown_content\", |b| {
        b.iter(|| {
            markdown_processing::parse_markdown_content(black_box(markdown_content), black_box(&config))
        })
    });

    // Create document for other benchmarks
    let document = markdown_processing::parse_markdown_content(markdown_content, &config).unwrap();

    group.bench_function(\"generate_toc\", |b| {
        b.iter(|| markdown_processing::generate_toc(black_box(&document), black_box(3)))
    });

    group.bench_function(\"analyze_code_blocks\", |b| {
        b.iter(|| markdown_processing::analyze_code_blocks(black_box(&document)))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_csv_processing,
    benchmark_json_processing,
    benchmark_hash_generation,
    benchmark_text_cleaning,
    benchmark_graph_operations,
    benchmark_file_io,
    benchmark_markdown_processing
);

criterion_main!(benches);