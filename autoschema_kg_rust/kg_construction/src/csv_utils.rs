//! CSV utilities for output generation and processing

use crate::{
    error::{KgConstructionError, Result},
    types::{ConceptNode, Edge, Node, NodeType},
};
use csv::{Writer, WriterBuilder};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// CSV output configuration
#[derive(Debug, Clone)]
pub struct CsvConfig {
    pub quote_all: bool,
    pub delimiter: u8,
    pub has_headers: bool,
    pub buffer_size: usize,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            quote_all: true,
            delimiter: b',',
            has_headers: true,
            buffer_size: 8192,
        }
    }
}

/// Thread-safe CSV writer for concurrent operations
pub struct ThreadSafeCsvWriter {
    writer: std::sync::Mutex<Writer<File>>,
}

impl ThreadSafeCsvWriter {
    pub fn new(file_path: impl AsRef<Path>, config: CsvConfig) -> Result<Self> {
        let file = File::create(file_path)?;
        let writer = WriterBuilder::new()
            .quote_style(if config.quote_all {
                csv::QuoteStyle::Always
            } else {
                csv::QuoteStyle::Necessary
            })
            .delimiter(config.delimiter)
            .has_headers(config.has_headers)
            .buffer_capacity(config.buffer_size)
            .from_writer(file);

        Ok(Self {
            writer: std::sync::Mutex::new(writer),
        })
    }

    pub fn write_record(&self, record: &[&str]) -> Result<()> {
        let mut writer = self.writer.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Lock poisoned: {}", e))
        })?;
        writer.write_record(record)?;
        writer.flush()?;
        Ok(())
    }

    pub fn write_headers(&self, headers: &[&str]) -> Result<()> {
        let mut writer = self.writer.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Lock poisoned: {}", e))
        })?;
        writer.write_record(headers)?;
        writer.flush()?;
        Ok(())
    }
}

/// Generate CSV output for concept nodes
pub fn write_concept_nodes_csv(
    output_file: impl AsRef<Path>,
    concept_nodes: &[ConceptNode],
    config: Option<CsvConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();
    let writer = ThreadSafeCsvWriter::new(output_file, config)?;

    // Write headers
    writer.write_headers(&["node", "conceptualized_node", "node_type"])?;

    // Write data
    for concept_node in concept_nodes {
        let conceptualized_str = concept_node.conceptualized_node.join(", ");
        writer.write_record(&[
            &concept_node.node,
            &conceptualized_str,
            &concept_node.node_type.to_string(),
        ])?;
    }

    Ok(())
}

/// Generate CSV output for nodes
pub fn write_nodes_csv(
    output_file: impl AsRef<Path>,
    nodes: &[Node],
    config: Option<CsvConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();
    let writer = ThreadSafeCsvWriter::new(output_file, config)?;

    // Write headers
    writer.write_headers(&["concept_id:ID", "name", ":LABEL"])?;

    // Write data
    for node in nodes {
        writer.write_record(&[&node.id, &node.name, &node.label])?;
    }

    Ok(())
}

/// Generate CSV output for edges
pub fn write_edges_csv(
    output_file: impl AsRef<Path>,
    edges: &[Edge],
    config: Option<CsvConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();
    let writer = ThreadSafeCsvWriter::new(output_file, config)?;

    // Write headers
    writer.write_headers(&[":START_ID", ":END_ID", "relation", ":TYPE"])?;

    // Write data
    for edge in edges {
        writer.write_record(&[&edge.start_id, &edge.end_id, &edge.relation, &edge.edge_type])?;
    }

    Ok(())
}

/// Generate CSV output for full concept triple edges
pub fn write_full_concept_triple_edges_csv(
    output_file: impl AsRef<Path>,
    edges: &[FullConceptTripleEdge],
    config: Option<CsvConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();
    let writer = ThreadSafeCsvWriter::new(output_file, config)?;

    // Write headers
    writer.write_headers(&[":START_ID", ":END_ID", "relation", "concepts", "synsets", ":TYPE"])?;

    // Write data
    for edge in edges {
        let concepts_str = format!("{:?}", edge.concepts); // Using Debug format for list
        let synsets_str = format!("{:?}", edge.synsets);
        writer.write_record(&[
            &edge.start_id,
            &edge.end_id,
            &edge.relation,
            &concepts_str,
            &synsets_str,
            &edge.edge_type,
        ])?;
    }

    Ok(())
}

/// Full concept triple edge structure
#[derive(Debug, Clone)]
pub struct FullConceptTripleEdge {
    pub start_id: String,
    pub end_id: String,
    pub relation: String,
    pub concepts: Vec<String>,
    pub synsets: Vec<String>,
    pub edge_type: String,
}

/// Buffered CSV writer for high-performance sequential writes
pub struct BufferedCsvWriter<W: Write> {
    writer: Writer<W>,
    buffer: Vec<Vec<String>>,
    buffer_size: usize,
}

impl<W: Write> BufferedCsvWriter<W> {
    pub fn new(writer: W, buffer_size: usize, config: CsvConfig) -> Self {
        let csv_writer = WriterBuilder::new()
            .quote_style(if config.quote_all {
                csv::QuoteStyle::Always
            } else {
                csv::QuoteStyle::Necessary
            })
            .delimiter(config.delimiter)
            .has_headers(config.has_headers)
            .from_writer(writer);

        Self {
            writer: csv_writer,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    pub fn write_headers(&mut self, headers: &[&str]) -> Result<()> {
        self.writer.write_record(headers)?;
        Ok(())
    }

    pub fn write_record(&mut self, record: Vec<String>) -> Result<()> {
        self.buffer.push(record);
        if self.buffer.len() >= self.buffer_size {
            self.flush_buffer()?;
        }
        Ok(())
    }

    pub fn flush_buffer(&mut self) -> Result<()> {
        for record in &self.buffer {
            self.writer.write_record(record)?;
        }
        self.buffer.clear();
        self.writer.flush()?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.flush_buffer()?;
        Ok(())
    }
}

/// Parse concepts from string representation (equivalent to Python ast.literal_eval)
pub fn parse_concepts(s: &str) -> Vec<String> {
    if s.is_empty() || s == "[]" {
        return Vec::new();
    }

    // Simple parsing for comma-separated values
    // In production, you might want more sophisticated parsing
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        inner
            .split(',')
            .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        s.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

/// Clean text for CSV output (equivalent to Python clean_text function)
pub fn clean_text_for_csv(text: &str) -> String {
    text.replace('\n', " ")
        .replace('\r', " ")
        .replace('\t', " ")
        .replace('\x0b', " ") // vertical tab
        .replace('\x0c', " ") // form feed
        .replace('\x08', " ") // backspace
        .replace('\x07', " ") // bell
        .replace('\x1b', " ") // escape
        .replace(';', ",")
        .replace('\x00', "") // NUL character
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Convert attributes to CSV-compatible types
pub fn convert_attribute(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Array(arr) => {
            let str_values: Vec<String> = arr
                .iter()
                .map(|v| convert_attribute(v))
                .collect();
            format!("[{}]", str_values.join(","))
        }
        serde_json::Value::Object(_) => format!("{}", value),
        serde_json::Value::Null => "".to_string(),
    }
}

/// Create directory if it doesn't exist
pub fn ensure_directory_exists(file_path: impl AsRef<Path>) -> Result<()> {
    if let Some(parent) = file_path.as_ref().parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Validate CSV file format and structure
pub fn validate_csv_file(
    file_path: impl AsRef<Path>,
    expected_headers: &[&str],
) -> Result<()> {
    let file = File::open(file_path)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let headers = reader.headers()?;

    if headers.len() != expected_headers.len() {
        return Err(KgConstructionError::ValidationError(
            format!(
                "Expected {} headers, found {}",
                expected_headers.len(),
                headers.len()
            ),
        ));
    }

    for (i, expected) in expected_headers.iter().enumerate() {
        if let Some(actual) = headers.get(i) {
            if actual != *expected {
                return Err(KgConstructionError::ValidationError(
                    format!(
                        "Header mismatch at position {}: expected '{}', found '{}'",
                        i, expected, actual
                    ),
                ));
            }
        }
    }

    Ok(())
}

/// Simple CSV processing for files
pub fn process_csv_simple<F>(
    input_file: impl AsRef<Path>,
    output_file: impl AsRef<Path>,
    processor: F,
) -> Result<()>
where
    F: Fn(csv::StringRecord) -> Result<Vec<String>>,
{
    let file = File::open(&input_file)?;
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);

    ensure_directory_exists(&output_file)?;
    let writer = ThreadSafeCsvWriter::new(output_file, CsvConfig::default())?;

    for result in reader.records() {
        let record = result?;
        match processor(record) {
            Ok(processed_row) => {
                let row_refs: Vec<&str> = processed_row.iter().map(|s| s.as_str()).collect();
                writer.write_record(&row_refs)?;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write as StdWrite;

    #[test]
    fn test_parse_concepts() {
        assert_eq!(parse_concepts(""), Vec::<String>::new());
        assert_eq!(parse_concepts("[]"), Vec::<String>::new());
        assert_eq!(parse_concepts("concept1, concept2"), vec!["concept1", "concept2"]);
        assert_eq!(
            parse_concepts("[\"concept1\", \"concept2\"]"),
            vec!["concept1", "concept2"]
        );
    }

    #[test]
    fn test_clean_text_for_csv() {
        let dirty = "Hello\nWorld\t\x00Test;More";
        let clean = clean_text_for_csv(dirty);
        assert_eq!(clean, "Hello World Test,More");
    }

    #[test]
    fn test_convert_attribute() {
        assert_eq!(convert_attribute(&serde_json::json!("test")), "test");
        assert_eq!(convert_attribute(&serde_json::json!(42)), "42");
        assert_eq!(convert_attribute(&serde_json::json!(true)), "true");
        assert_eq!(convert_attribute(&serde_json::json!(null)), "");
    }

    #[test]
    fn test_write_concept_nodes_csv() {
        let mut temp_file = NamedTempFile::new().unwrap();

        let concepts = vec![
            ConceptNode {
                node: "test_node".to_string(),
                conceptualized_node: vec!["concept1".to_string(), "concept2".to_string()],
                node_type: NodeType::Entity,
            }
        ];

        write_concept_nodes_csv(temp_file.path(), &concepts, None).unwrap();

        // Read back and verify
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("test_node"));
        assert!(content.contains("concept1, concept2"));
        assert!(content.contains("entity"));
    }
}