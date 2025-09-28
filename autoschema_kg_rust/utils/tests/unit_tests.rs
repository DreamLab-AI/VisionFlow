use utils::*;
use tokio_test;
use rstest::*;
use test_case::test_case;
use proptest::prelude::*;
use tempfile::TempDir;
use std::path::Path;

mod common;
use common::*;

#[cfg(test)]
mod csv_processing_tests {
    use super::*;

    #[test]
    fn test_csv_reader_creation() {
        let csv_data = "id,name,value\n1,test,42\n2,example,24";
        let reader = CsvProcessor::from_string(csv_data);
        assert!(reader.is_ok());
    }

    #[test]
    fn test_csv_parsing() {
        let csv_data = "id,name,age\n1,Alice,30\n2,Bob,25\n3,Charlie,35";
        let processor = CsvProcessor::from_string(csv_data).unwrap();

        let records = processor.parse_all().unwrap();
        assert_eq!(records.len(), 3);

        // Check first record
        assert_eq!(records[0].get("id"), Some("1"));
        assert_eq!(records[0].get("name"), Some("Alice"));
        assert_eq!(records[0].get("age"), Some("30"));
    }

    #[test]
    fn test_csv_with_custom_delimiter() {
        let csv_data = "id;name;value\n1;test;42\n2;example;24";
        let processor = CsvProcessor::from_string_with_delimiter(csv_data, b';');
        assert!(processor.is_ok());

        let records = processor.unwrap().parse_all().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_csv_error_handling() {
        let invalid_csv = "id,name\n1,Alice,extra_field";
        let processor = CsvProcessor::from_string(invalid_csv).unwrap();
        let result = processor.parse_all();

        // Should handle malformed CSV gracefully
        assert!(result.is_err() || result.unwrap().len() <= 1);
    }

    #[tokio::test]
    async fn test_csv_async_processing() {
        let csv_data = "id,name,value\n1,test,42\n2,example,24\n3,sample,66";
        let processor = CsvProcessor::from_string(csv_data).unwrap();

        let mut count = 0;
        let mut stream = processor.parse_stream().await.unwrap();

        while let Some(record) = stream.next().await {
            assert!(record.is_ok());
            count += 1;
        }

        assert_eq!(count, 3);
    }

    #[test]
    fn test_csv_to_json_conversion() {
        let csv_data = "id,name,active\n1,Alice,true\n2,Bob,false";
        let processor = CsvProcessor::from_string(csv_data).unwrap();

        let json_array = processor.to_json().unwrap();
        assert_eq!(json_array.as_array().unwrap().len(), 2);

        let first_record = &json_array[0];
        assert_eq!(first_record["id"], "1");
        assert_eq!(first_record["name"], "Alice");
        assert_eq!(first_record["active"], "true");
    }

    #[test]
    fn test_large_csv_processing() {
        // Generate large CSV data
        let mut csv_data = String::from("id,name,value\n");
        for i in 0..10000 {
            csv_data.push_str(&format!("{},name_{},{}\n", i, i, i * 2));
        }

        let processor = CsvProcessor::from_string(&csv_data).unwrap();
        let records = processor.parse_all().unwrap();

        assert_eq!(records.len(), 10000);
        assert_eq!(records[9999].get("id"), Some("9999"));
    }
}

#[cfg(test)]
mod json_processing_tests {
    use super::*;

    #[test]
    fn test_json_processor_creation() {
        let processor = JsonProcessor::new();
        assert!(processor.is_valid());
    }

    #[test]
    fn test_json_parsing() {
        let json_str = r#"{"name": "Alice", "age": 30, "active": true}"#;
        let processor = JsonProcessor::new();

        let parsed = processor.parse(json_str).unwrap();
        assert_eq!(parsed["name"], "Alice");
        assert_eq!(parsed["age"], 30);
        assert_eq!(parsed["active"], true);
    }

    #[test]
    fn test_json_array_processing() {
        let json_str = r#"[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]"#;
        let processor = JsonProcessor::new();

        let parsed = processor.parse(json_str).unwrap();
        let array = parsed.as_array().unwrap();

        assert_eq!(array.len(), 2);
        assert_eq!(array[0]["name"], "Alice");
        assert_eq!(array[1]["name"], "Bob");
    }

    #[test]
    fn test_json_nested_objects() {
        let json_str = r#"{
            "user": {
                "profile": {
                    "name": "Alice",
                    "preferences": {
                        "theme": "dark",
                        "notifications": true
                    }
                }
            }
        }"#;

        let processor = JsonProcessor::new();
        let parsed = processor.parse(json_str).unwrap();

        assert_eq!(parsed["user"]["profile"]["name"], "Alice");
        assert_eq!(parsed["user"]["profile"]["preferences"]["theme"], "dark");
    }

    #[test]
    fn test_json_flattening() {
        let nested_json = serde_json::json!({
            "user": {
                "name": "Alice",
                "address": {
                    "street": "123 Main St",
                    "city": "Springfield"
                }
            }
        });

        let processor = JsonProcessor::new();
        let flattened = processor.flatten(&nested_json, "_").unwrap();

        assert_eq!(flattened["user_name"], "Alice");
        assert_eq!(flattened["user_address_street"], "123 Main St");
        assert_eq!(flattened["user_address_city"], "Springfield");
    }

    #[test]
    fn test_json_schema_validation() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        });

        let valid_data = serde_json::json!({"name": "Alice", "age": 30});
        let invalid_data = serde_json::json!({"name": "Alice"}); // missing age

        let processor = JsonProcessor::new();
        assert!(processor.validate(&valid_data, &schema).is_ok());
        assert!(processor.validate(&invalid_data, &schema).is_err());
    }
}

#[cfg(test)]
mod markdown_processing_tests {
    use super::*;

    #[test]
    fn test_markdown_processor_creation() {
        let processor = MarkdownProcessor::new();
        assert!(processor.is_ready());
    }

    #[test]
    fn test_markdown_to_html() {
        let markdown = r#"
# Heading 1

This is a paragraph with **bold** and *italic* text.

## Heading 2

- Item 1
- Item 2
- Item 3

[Link](https://example.com)
"#;

        let processor = MarkdownProcessor::new();
        let html = processor.to_html(markdown).unwrap();

        assert!(html.contains("<h1>Heading 1</h1>"));
        assert!(html.contains("<strong>bold</strong>"));
        assert!(html.contains("<em>italic</em>"));
        assert!(html.contains("<ul>"));
        assert!(html.contains("<li>Item 1</li>"));
        assert!(html.contains("<a href=\"https://example.com\">Link</a>"));
    }

    #[test]
    fn test_markdown_extract_headers() {
        let markdown = r#"
# Main Title

Some content here.

## Section 1

More content.

### Subsection 1.1

Even more content.

## Section 2

Final content.
"#;

        let processor = MarkdownProcessor::new();
        let headers = processor.extract_headers(markdown).unwrap();

        assert_eq!(headers.len(), 4);
        assert_eq!(headers[0].level, 1);
        assert_eq!(headers[0].text, "Main Title");
        assert_eq!(headers[1].level, 2);
        assert_eq!(headers[1].text, "Section 1");
    }

    #[test]
    fn test_markdown_extract_links() {
        let markdown = r#"
Here are some links:
- [Google](https://google.com)
- [Example](https://example.com "Example Site")
- [Local](/path/to/page)

Also inline: Check out [this site](https://test.com) for more info.
"#;

        let processor = MarkdownProcessor::new();
        let links = processor.extract_links(markdown).unwrap();

        assert_eq!(links.len(), 4);
        assert!(links.iter().any(|l| l.url == "https://google.com"));
        assert!(links.iter().any(|l| l.url == "https://example.com"));
        assert!(links.iter().any(|l| l.text == "Local"));
    }

    #[test]
    fn test_markdown_to_plain_text() {
        let markdown = r#"
# Title

This is **bold** and *italic* text with [a link](https://example.com).

> This is a blockquote.

- List item 1
- List item 2
"#;

        let processor = MarkdownProcessor::new();
        let plain_text = processor.to_plain_text(markdown).unwrap();

        assert!(!plain_text.contains("#"));
        assert!(!plain_text.contains("**"));
        assert!(!plain_text.contains("*"));
        assert!(!plain_text.contains("["));
        assert!(!plain_text.contains("]"));
        assert!(plain_text.contains("Title"));
        assert!(plain_text.contains("bold"));
        assert!(plain_text.contains("italic"));
    }

    #[test]
    fn test_markdown_table_extraction() {
        let markdown = r#"
| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |
| Charlie | 35 | Chicago |
"#;

        let processor = MarkdownProcessor::new();
        let tables = processor.extract_tables(markdown).unwrap();

        assert_eq!(tables.len(), 1);
        let table = &tables[0];
        assert_eq!(table.headers, vec!["Name", "Age", "City"]);
        assert_eq!(table.rows.len(), 3);
        assert_eq!(table.rows[0], vec!["Alice", "30", "NYC"]);
    }
}

#[cfg(test)]
mod hash_utils_tests {
    use super::*;

    #[test]
    fn test_sha256_hash() {
        let input = "Hello, world!";
        let hash1 = hash_sha256(input);
        let hash2 = hash_sha256(input);

        // Same input should produce same hash
        assert_eq!(hash1, hash2);

        // Hash should be 64 characters (256 bits in hex)
        assert_eq!(hash1.len(), 64);

        // Different input should produce different hash
        let hash3 = hash_sha256("Different input");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_md5_hash() {
        let input = "test string";
        let hash = hash_md5(input);

        // MD5 hash should be 32 characters
        assert_eq!(hash.len(), 32);

        // Should be consistent
        assert_eq!(hash, hash_md5(input));
    }

    #[test]
    fn test_hash_file_content() {
        let content = "File content for hashing";
        let hash = hash_content(content, HashAlgorithm::Sha256);

        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test_case(""; "empty string")]
    #[test_case("a"; "single character")]
    #[test_case("Hello, world!"; "common phrase")]
    #[test_case("🚀 Unicode test 测试"; "unicode content")]
    fn test_hash_consistency(input: &str) {
        let hash1 = hash_sha256(input);
        let hash2 = hash_sha256(input);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_verify_hash() {
        let input = "content to verify";
        let hash = hash_sha256(input);

        assert!(verify_hash(input, &hash, HashAlgorithm::Sha256));
        assert!(!verify_hash("different content", &hash, HashAlgorithm::Sha256));
    }
}

#[cfg(test)]
mod file_io_tests {
    use super::*;

    #[test]
    fn test_file_reader_creation() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "test content").unwrap();

        let reader = FileReader::new(&file_path);
        assert!(reader.is_ok());
    }

    #[tokio::test]
    async fn test_async_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("async_test.txt");

        let content = "This is test content for async operations.";

        // Write file asynchronously
        let result = write_file_async(&file_path, content).await;
        assert!(result.is_ok());

        // Read file asynchronously
        let read_content = read_file_async(&file_path).await.unwrap();
        assert_eq!(read_content, content);
    }

    #[test]
    fn test_file_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("metadata_test.txt");
        let content = "Content for metadata testing";
        std::fs::write(&file_path, content).unwrap();

        let metadata = get_file_metadata(&file_path).unwrap();
        assert_eq!(metadata.size, content.len() as u64);
        assert!(metadata.is_file);
        assert!(!metadata.is_directory);
        assert!(metadata.created.is_some());
        assert!(metadata.modified.is_some());
    }

    #[test]
    fn test_directory_operations() {
        let temp_dir = TempDir::new().unwrap();
        let sub_dir = temp_dir.path().join("subdir");

        // Create directory
        assert!(create_directory(&sub_dir).is_ok());
        assert!(sub_dir.exists());
        assert!(sub_dir.is_dir());

        // List directory contents
        std::fs::write(sub_dir.join("file1.txt"), "content1").unwrap();
        std::fs::write(sub_dir.join("file2.txt"), "content2").unwrap();

        let entries = list_directory(&sub_dir).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().any(|e| e.file_name() == Some("file1.txt".as_ref())));
        assert!(entries.iter().any(|e| e.file_name() == Some("file2.txt".as_ref())));
    }

    #[test]
    fn test_file_copy_and_move() {
        let temp_dir = TempDir::new().unwrap();
        let source = temp_dir.path().join("source.txt");
        let copy_dest = temp_dir.path().join("copy.txt");
        let move_dest = temp_dir.path().join("moved.txt");

        let content = "Content to copy and move";
        std::fs::write(&source, content).unwrap();

        // Test copy
        assert!(copy_file(&source, &copy_dest).is_ok());
        assert!(copy_dest.exists());
        assert_eq!(std::fs::read_to_string(&copy_dest).unwrap(), content);

        // Test move
        assert!(move_file(&source, &move_dest).is_ok());
        assert!(!source.exists());
        assert!(move_dest.exists());
        assert_eq!(std::fs::read_to_string(&move_dest).unwrap(), content);
    }

    #[test]
    fn test_file_compression() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("compress_test.txt");
        let compressed_path = temp_dir.path().join("compressed.gz");

        let content = "This is content that will be compressed. ".repeat(100);
        std::fs::write(&file_path, &content).unwrap();

        // Compress file
        assert!(compress_file(&file_path, &compressed_path).is_ok());
        assert!(compressed_path.exists());

        // Compressed file should be smaller
        let original_size = std::fs::metadata(&file_path).unwrap().len();
        let compressed_size = std::fs::metadata(&compressed_path).unwrap().len();
        assert!(compressed_size < original_size);

        // Decompress and verify
        let decompressed_path = temp_dir.path().join("decompressed.txt");
        assert!(decompress_file(&compressed_path, &decompressed_path).is_ok());

        let decompressed_content = std::fs::read_to_string(&decompressed_path).unwrap();
        assert_eq!(decompressed_content, content);
    }
}

#[cfg(test)]
mod text_cleaning_tests {
    use super::*;

    #[test]
    fn test_text_cleaner_creation() {
        let cleaner = TextCleaner::new();
        assert!(cleaner.is_initialized());
    }

    #[test_case("Hello World", "hello world"; "basic lowercase")]
    #[test_case("HELLO WORLD", "hello world"; "all uppercase")]
    #[test_case("MiXeD cAsE", "mixed case"; "mixed case")]
    fn test_normalize_case(input: &str, expected: &str) {
        let cleaner = TextCleaner::new();
        let result = cleaner.normalize_case(input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_remove_extra_whitespace() {
        let cleaner = TextCleaner::new();

        let inputs_and_expected = vec![
            ("  hello   world  ", "hello world"),
            ("multiple\n\n\nlines", "multiple lines"),
            ("tabs\t\t\tand spaces", "tabs and spaces"),
            (" \t \n mixed \t \n ", "mixed"),
        ];

        for (input, expected) in inputs_and_expected {
            let result = cleaner.remove_extra_whitespace(input);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_remove_special_characters() {
        let cleaner = TextCleaner::new();

        let test_cases = vec![
            ("hello@world#test!", "hello world test"),
            ("price: $19.99", "price 19 99"),
            ("email@example.com", "email example com"),
            ("(555) 123-4567", "555 123 4567"),
        ];

        for (input, expected) in test_cases {
            let result = cleaner.remove_special_characters(input);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_remove_stop_words() {
        let cleaner = TextCleaner::new();

        let input = "this is a test sentence with some common stop words";
        let result = cleaner.remove_stop_words(input);

        // Should remove common stop words like "this", "is", "a", "with", "some"
        assert!(!result.contains(" is "));
        assert!(!result.contains(" a "));
        assert!(!result.contains(" the "));
        assert!(result.contains("test"));
        assert!(result.contains("sentence"));
    }

    #[test]
    fn test_extract_sentences() {
        let cleaner = TextCleaner::new();

        let text = "First sentence. Second sentence! Third sentence? Fourth sentence.";
        let sentences = cleaner.extract_sentences(text);

        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "First sentence");
        assert_eq!(sentences[1], "Second sentence");
        assert_eq!(sentences[2], "Third sentence");
        assert_eq!(sentences[3], "Fourth sentence");
    }

    #[test]
    fn test_extract_words() {
        let cleaner = TextCleaner::new();

        let text = "Hello, world! This is a test.";
        let words = cleaner.extract_words(text);

        let expected_words = vec!["Hello", "world", "This", "is", "a", "test"];
        assert_eq!(words, expected_words);
    }

    #[test]
    fn test_clean_pipeline() {
        let cleaner = TextCleaner::new();

        let dirty_text = "  HELLO,   WORLD!!!   This  is    A   TEST...  ";
        let cleaned = cleaner
            .normalize_case(dirty_text)
            .pipe(|s| cleaner.remove_special_characters(s))
            .pipe(|s| cleaner.remove_extra_whitespace(s));

        assert_eq!(cleaned, "hello world this is a test");
    }

    #[test]
    fn test_unicode_handling() {
        let cleaner = TextCleaner::new();

        let unicode_text = "Héllo wörld! 测试 🚀";
        let normalized = cleaner.normalize_unicode(unicode_text);

        // Should handle unicode characters properly
        assert!(normalized.len() > 0);
        assert!(normalized.contains("hello") || normalized.contains("Héllo"));
    }
}

#[cfg(test)]
mod graph_conversion_tests {
    use super::*;

    #[test]
    fn test_graph_converter_creation() {
        let converter = GraphConverter::new();
        assert!(converter.is_ready());
    }

    #[test]
    fn test_csv_to_graph_conversion() {
        let csv_data = "source,target,weight\nA,B,1.0\nB,C,2.0\nC,A,0.5";
        let converter = GraphConverter::new();

        let graph = converter.csv_to_graph(csv_data).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);

        // Check nodes exist
        assert!(graph.has_node("A"));
        assert!(graph.has_node("B"));
        assert!(graph.has_node("C"));

        // Check edges exist with correct weights
        assert_eq!(graph.edge_weight("A", "B"), Some(1.0));
        assert_eq!(graph.edge_weight("B", "C"), Some(2.0));
        assert_eq!(graph.edge_weight("C", "A"), Some(0.5));
    }

    #[test]
    fn test_json_to_graph_conversion() {
        let json_data = serde_json::json!({
            "nodes": [
                {"id": "1", "label": "Node 1"},
                {"id": "2", "label": "Node 2"},
                {"id": "3", "label": "Node 3"}
            ],
            "edges": [
                {"source": "1", "target": "2", "weight": 1.5},
                {"source": "2", "target": "3", "weight": 2.0}
            ]
        });

        let converter = GraphConverter::new();
        let graph = converter.json_to_graph(&json_data).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_graph_to_adjacency_matrix() {
        let converter = GraphConverter::new();
        let mut graph = Graph::new();

        graph.add_node("A");
        graph.add_node("B");
        graph.add_node("C");
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 2.0);

        let matrix = converter.to_adjacency_matrix(&graph).unwrap();
        assert_eq!(matrix.shape(), (3, 3));

        // Check matrix values
        assert_eq!(matrix[(0, 1)], 1.0); // A -> B
        assert_eq!(matrix[(1, 2)], 2.0); // B -> C
        assert_eq!(matrix[(0, 2)], 0.0); // A -> C (no direct edge)
    }

    #[test]
    fn test_graph_statistics() {
        let converter = GraphConverter::new();
        let csv_data = "source,target,weight\nA,B,1.0\nB,C,2.0\nC,D,1.5\nD,A,0.8\nB,D,1.2";

        let graph = converter.csv_to_graph(csv_data).unwrap();
        let stats = converter.calculate_statistics(&graph);

        assert_eq!(stats.node_count, 4);
        assert_eq!(stats.edge_count, 5);
        assert!(stats.density > 0.0);
        assert!(stats.avg_degree > 0.0);
    }
}

// Property-based tests
proptest! {
    #[test]
    fn test_hash_deterministic(input in ".*") {
        let hash1 = hash_sha256(&input);
        let hash2 = hash_sha256(&input);
        prop_assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_text_cleaning_properties(
        text in r"[a-zA-Z0-9\s\.\,\!\?]{1,100}"
    ) {
        let cleaner = TextCleaner::new();
        let cleaned = cleaner.remove_extra_whitespace(&text);

        // Cleaned text should not have multiple consecutive spaces
        prop_assert!(!cleaned.contains("  "));

        // Should not be longer than original
        prop_assert!(cleaned.len() <= text.len());
    }

    #[test]
    fn test_csv_processing_properties(
        data in prop::collection::vec(
            (1u32..1000, "[a-zA-Z]{1,20}", 0.0f64..1000.0),
            1..100
        )
    ) {
        let mut csv_content = String::from("id,name,value\n");
        for (id, name, value) in &data {
            csv_content.push_str(&format!("{},{},{}\n", id, name, value));
        }

        let processor = CsvProcessor::from_string(&csv_content);
        prop_assert!(processor.is_ok());

        if let Ok(proc) = processor {
            let records = proc.parse_all();
            prop_assert!(records.is_ok());

            if let Ok(recs) = records {
                prop_assert_eq!(recs.len(), data.len());
            }
        }
    }
}

// Benchmark placeholder - actual benchmarks would be in benches/
#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_csv_processing_performance() {
        let large_csv = generate_large_csv(10000);
        let start = Instant::now();

        let processor = CsvProcessor::from_string(&large_csv).unwrap();
        let _records = processor.parse_all().unwrap();

        let duration = start.elapsed();
        println!("CSV processing took: {:?}", duration);

        // Should complete within reasonable time (adjust as needed)
        assert!(duration.as_secs() < 5);
    }

    fn generate_large_csv(rows: usize) -> String {
        let mut csv = String::from("id,name,value,timestamp\n");
        for i in 0..rows {
            csv.push_str(&format!("{},name_{},{},{}\n", i, i, i as f64 * 1.1, i * 1000));
        }
        csv
    }
}