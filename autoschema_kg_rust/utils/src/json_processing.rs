//! JSON processing utilities with repair, conversion, and streaming capabilities

use crate::{Result, UtilsError};
use csv::{Writer, StringRecord};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, Map};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Configuration for JSON processing operations
#[derive(Debug, Clone)]
pub struct JsonConfig {
    pub pretty_print: bool,
    pub flatten_arrays: bool,
    pub max_depth: usize,
    pub include_null_values: bool,
    pub buffer_size: usize,
}

impl Default for JsonConfig {
    fn default() -> Self {
        Self {
            pretty_print: false,
            flatten_arrays: true,
            max_depth: 10,
            include_null_values: false,
            buffer_size: 8192,
        }
    }
}

/// Represents a flattened JSON path and value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonPath {
    pub path: String,
    pub value: Value,
    pub data_type: String,
}

/// Statistics about JSON processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonStats {
    pub total_objects: usize,
    pub total_arrays: usize,
    pub max_depth: usize,
    pub unique_paths: usize,
    pub file_size: u64,
}

/// Convert JSON file to CSV format
///
/// # Arguments
/// * `json_file` - Input JSON file path
/// * `csv_file` - Output CSV file path
/// * `config` - JSON processing configuration
pub fn json_to_csv<P: AsRef<Path>>(
    json_file: P,
    csv_file: P,
    config: &JsonConfig,
) -> Result<JsonStats> {
    let input_file = File::open(&json_file)?;
    let file_size = input_file.metadata()?.len();
    let reader = BufReader::new(input_file);

    // Parse JSON
    let json_value: Value = serde_json::from_reader(reader)?;

    let output_file = File::create(csv_file)?;
    let mut writer = Writer::from_writer(BufWriter::new(output_file));

    let mut stats = JsonStats {
        total_objects: 0,
        total_arrays: 0,
        max_depth: 0,
        unique_paths: 0,
        file_size,
    };

    match json_value {
        Value::Array(array) => {
            // Process array of objects
            let flattened_objects = array.iter()
                .map(|obj| flatten_json(obj, config, "", 0))
                .collect::<Vec<_>>();

            if !flattened_objects.is_empty() {
                // Collect all unique column names
                let mut all_columns = IndexMap::new();
                for obj in &flattened_objects {
                    for path in obj {
                        all_columns.insert(path.path.clone(), path.data_type.clone());
                    }
                }

                // Write headers
                let headers: Vec<String> = all_columns.keys().cloned().collect();
                writer.write_record(&headers)?;

                // Write data rows
                for obj in flattened_objects {
                    let mut record = StringRecord::new();
                    let obj_map: HashMap<String, String> = obj.iter()
                        .map(|p| (p.path.clone(), value_to_string(&p.value)))
                        .collect();

                    for header in &headers {
                        let value = obj_map.get(header).unwrap_or(&String::new());
                        record.push_field(value);
                    }
                    writer.write_record(&record)?;
                }

                stats.unique_paths = all_columns.len();
            }

            stats.total_arrays = 1;
        }
        Value::Object(_) => {
            // Process single object
            let flattened = flatten_json(&json_value, config, "", 0);

            if !flattened.is_empty() {
                // Write headers
                let headers: Vec<String> = flattened.iter().map(|p| p.path.clone()).collect();
                writer.write_record(&headers)?;

                // Write single row
                let mut record = StringRecord::new();
                for path in &flattened {
                    record.push_field(&value_to_string(&path.value));
                }
                writer.write_record(&record)?;

                stats.unique_paths = flattened.len();
            }

            stats.total_objects = 1;
        }
        _ => {
            return Err(UtilsError::Custom("JSON must be an object or array".to_string()));
        }
    }

    writer.flush()?;
    Ok(stats)
}

/// Convert JSON file to GraphML format
///
/// # Arguments
/// * `json_file` - Input JSON file path
/// * `graphml_file` - Output GraphML file path
/// * `config` - JSON processing configuration
/// * `node_id_field` - Field to use as node ID
pub fn json_to_graphml<P: AsRef<Path>>(
    json_file: P,
    graphml_file: P,
    config: &JsonConfig,
    node_id_field: Option<&str>,
) -> Result<()> {
    use quick_xml::events::{Event, BytesEnd, BytesStart, BytesText};
    use quick_xml::Writer;

    let input_file = File::open(json_file)?;
    let reader = BufReader::new(input_file);
    let json_value: Value = serde_json::from_reader(reader)?;

    let output_file = File::create(graphml_file)?;
    let mut xml_writer = Writer::new(BufWriter::new(output_file));

    // Write GraphML header
    xml_writer.write_event(Event::Decl(quick_xml::events::BytesDecl::new("1.0", Some("UTF-8"), None)))?;

    let mut graphml_start = BytesStart::new("graphml");
    graphml_start.push_attribute(("xmlns", "http://graphml.graphdrawing.org/xmlns"));
    xml_writer.write_event(Event::Start(graphml_start))?;

    // Start graph
    let mut graph_start = BytesStart::new("graph");
    graph_start.push_attribute(("id", "G"));
    graph_start.push_attribute(("edgedefault", "undirected"));
    xml_writer.write_event(Event::Start(graph_start))?;

    match json_value {
        Value::Array(array) => {
            for (idx, item) in array.iter().enumerate() {
                if let Value::Object(obj) = item {
                    let node_id = if let Some(id_field) = node_id_field {
                        obj.get(id_field)
                            .and_then(|v| v.as_str())
                            .unwrap_or(&format!("node_{}", idx))
                            .to_string()
                    } else {
                        format!("node_{}", idx)
                    };

                    write_json_node(&mut xml_writer, &node_id, obj)?;
                }
            }
        }
        Value::Object(obj) => {
            let node_id = node_id_field
                .and_then(|field| obj.get(field))
                .and_then(|v| v.as_str())
                .unwrap_or("root");

            write_json_node(&mut xml_writer, node_id, &obj)?;
        }
        _ => {
            return Err(UtilsError::Custom("JSON must be an object or array".to_string()));
        }
    }

    xml_writer.write_event(Event::End(BytesEnd::new("graph")))?;
    xml_writer.write_event(Event::End(BytesEnd::new("graphml")))?;

    Ok(())
}

/// Repair malformed JSON by attempting various fixes
///
/// # Arguments
/// * `input` - Malformed JSON string
/// * `config` - JSON processing configuration
///
/// # Returns
/// * `Result<Value>` - Repaired JSON value
pub fn json_repair(input: &str, config: &JsonConfig) -> Result<Value> {
    // First, try parsing as-is
    if let Ok(value) = serde_json::from_str::<Value>(input) {
        return Ok(value);
    }

    let mut repaired = input.to_string();

    // Common repairs
    repaired = fix_trailing_commas(&repaired);
    repaired = fix_unquoted_keys(&repaired);
    repaired = fix_single_quotes(&repaired);
    repaired = fix_missing_quotes(&repaired);
    repaired = fix_control_characters(&repaired);

    // Try parsing again
    if let Ok(value) = serde_json::from_str::<Value>(&repaired) {
        return Ok(value);
    }

    // More aggressive repairs
    repaired = fix_malformed_arrays(&repaired);
    repaired = fix_malformed_objects(&repaired);

    // Final attempt
    serde_json::from_str::<Value>(&repaired)
        .map_err(|e| UtilsError::Custom(format!("Could not repair JSON: {}", e)))
}

/// Validate JSON structure and return detailed analysis
pub fn validate_json<P: AsRef<Path>>(json_file: P) -> Result<JsonStats> {
    let input_file = File::open(&json_file)?;
    let file_size = input_file.metadata()?.len();
    let reader = BufReader::new(input_file);

    let json_value: Value = serde_json::from_reader(reader)?;

    let mut stats = JsonStats {
        total_objects: 0,
        total_arrays: 0,
        max_depth: 0,
        unique_paths: 0,
        file_size,
    };

    analyze_json_structure(&json_value, &mut stats, 0);

    Ok(stats)
}

/// Stream process large JSON files without loading everything into memory
pub fn stream_process_json<P: AsRef<Path>, F>(
    json_file: P,
    mut processor: F,
) -> Result<()>
where
    F: FnMut(&Value) -> Result<()>,
{
    use std::io::BufRead;

    let file = File::open(json_file)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<Value>(&line) {
            Ok(value) => processor(&value)?,
            Err(_) => {
                // Try to repair the line
                if let Ok(repaired) = json_repair(&line, &JsonConfig::default()) {
                    processor(&repaired)?;
                }
            }
        }
    }

    Ok(())
}

// Helper functions

fn flatten_json(value: &Value, config: &JsonConfig, prefix: &str, depth: usize) -> Vec<JsonPath> {
    if depth > config.max_depth {
        return vec![];
    }

    let mut result = Vec::new();

    match value {
        Value::Object(map) => {
            for (key, val) in map {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };

                match val {
                    Value::Object(_) | Value::Array(_) => {
                        result.extend(flatten_json(val, config, &new_prefix, depth + 1));
                    }
                    _ => {
                        if config.include_null_values || !val.is_null() {
                            result.push(JsonPath {
                                path: new_prefix,
                                value: val.clone(),
                                data_type: get_json_type(val),
                            });
                        }
                    }
                }
            }
        }
        Value::Array(arr) => {
            if config.flatten_arrays {
                for (idx, val) in arr.iter().enumerate() {
                    let new_prefix = format!("{}[{}]", prefix, idx);
                    result.extend(flatten_json(val, config, &new_prefix, depth + 1));
                }
            } else {
                result.push(JsonPath {
                    path: prefix.to_string(),
                    value: value.clone(),
                    data_type: "array".to_string(),
                });
            }
        }
        _ => {
            if config.include_null_values || !value.is_null() {
                result.push(JsonPath {
                    path: prefix.to_string(),
                    value: value.clone(),
                    data_type: get_json_type(value),
                });
            }
        }
    }

    result
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn get_json_type(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "boolean".to_string(),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer".to_string()
            } else {
                "float".to_string()
            }
        }
        Value::String(_) => "string".to_string(),
        Value::Array(_) => "array".to_string(),
        Value::Object(_) => "object".to_string(),
    }
}

fn write_json_node<W: Write>(
    writer: &mut quick_xml::Writer<W>,
    node_id: &str,
    obj: &Map<String, Value>,
) -> Result<()> {
    use quick_xml::events::{Event, BytesEnd, BytesStart, BytesText};

    let mut node_start = BytesStart::new("node");
    node_start.push_attribute(("id", node_id));
    writer.write_event(Event::Start(node_start))?;

    for (key, value) in obj {
        if !value.is_null() {
            let mut data_start = BytesStart::new("data");
            data_start.push_attribute(("key", key.as_str()));
            writer.write_event(Event::Start(data_start))?;
            writer.write_event(Event::Text(BytesText::new(&value_to_string(value))))?;
            writer.write_event(Event::End(BytesEnd::new("data")))?;
        }
    }

    writer.write_event(Event::End(BytesEnd::new("node")))?;
    Ok(())
}

fn analyze_json_structure(value: &Value, stats: &mut JsonStats, depth: usize) {
    stats.max_depth = stats.max_depth.max(depth);

    match value {
        Value::Object(obj) => {
            stats.total_objects += 1;
            for val in obj.values() {
                analyze_json_structure(val, stats, depth + 1);
            }
        }
        Value::Array(arr) => {
            stats.total_arrays += 1;
            for val in arr {
                analyze_json_structure(val, stats, depth + 1);
            }
        }
        _ => {}
    }
}

// JSON repair functions

fn fix_trailing_commas(input: &str) -> String {
    use regex::Regex;

    let re = Regex::new(r",\\s*([}\\]])").unwrap();
    re.replace_all(input, "$1").to_string()
}

fn fix_unquoted_keys(input: &str) -> String {
    use regex::Regex;

    let re = Regex::new(r"([{,]\\s*)([a-zA-Z_][a-zA-Z0-9_]*)\\s*:").unwrap();
    re.replace_all(input, "$1\\"$2\\":").to_string()
}

fn fix_single_quotes(input: &str) -> String {
    input.replace("'", "\\"")
}

fn fix_missing_quotes(input: &str) -> String {
    use regex::Regex;

    // Fix unquoted string values
    let re = Regex::new(r":\\s*([a-zA-Z][a-zA-Z0-9_]*)(\\s*[,}])").unwrap();
    re.replace_all(input, ": \\"$1\\"$2").to_string()
}

fn fix_control_characters(input: &str) -> String {
    input.chars()
        .filter(|&c| c >= ' ' || c == '\\t' || c == '\\n' || c == '\\r')
        .collect()
}

fn fix_malformed_arrays(input: &str) -> String {
    // Basic array structure fixes
    input.replace("[,", "[").replace(",]", "]")
}

fn fix_malformed_objects(input: &str) -> String {
    // Basic object structure fixes
    input.replace("{,", "{").replace(",}", "}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_json_repair() {
        let malformed = r#"{name: 'John', age: 30,}"#;
        let config = JsonConfig::default();
        let repaired = json_repair(malformed, &config).unwrap();

        if let Value::Object(obj) = repaired {
            assert_eq!(obj["name"], Value::String("John".to_string()));
            assert_eq!(obj["age"], Value::Number(serde_json::Number::from(30)));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_json_to_csv() {
        let json_data = r#"[
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "Los Angeles"}
        ]"#;

        let mut json_file = NamedTempFile::new().unwrap();
        write!(json_file, "{}", json_data).unwrap();

        let csv_file = NamedTempFile::new().unwrap();
        let config = JsonConfig::default();

        let stats = json_to_csv(json_file.path(), csv_file.path(), &config).unwrap();
        assert_eq!(stats.total_arrays, 1);
        assert_eq!(stats.unique_paths, 3);
    }

    #[test]
    fn test_flatten_json() {
        let json_value: Value = serde_json::from_str(r#"{
            "user": {
                "name": "Alice",
                "address": {
                    "street": "123 Main St",
                    "city": "New York"
                }
            }
        }"#).unwrap();

        let config = JsonConfig::default();
        let flattened = flatten_json(&json_value, &config, "", 0);

        assert_eq!(flattened.len(), 3);
        assert!(flattened.iter().any(|p| p.path == "user.name"));
        assert!(flattened.iter().any(|p| p.path == "user.address.street"));
        assert!(flattened.iter().any(|p| p.path == "user.address.city"));
    }
}