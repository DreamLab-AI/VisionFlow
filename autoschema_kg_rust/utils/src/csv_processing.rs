//! CSV processing utilities with memory-efficient streaming operations

use crate::{Result, UtilsError};
use csv::{Reader, Writer, StringRecord};
use indexmap::IndexMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Configuration for CSV processing operations
#[derive(Debug, Clone)]
pub struct CsvConfig {
    pub delimiter: u8,
    pub has_headers: bool,
    pub flexible: bool,
    pub buffer_size: usize,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_headers: true,
            flexible: true,
            buffer_size: 8192,
        }
    }
}

/// Represents a CSV column with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvColumn {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub unique_values: usize,
}

/// Statistics about a CSV file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvStats {
    pub row_count: usize,
    pub column_count: usize,
    pub columns: Vec<CsvColumn>,
    pub file_size: u64,
}

/// Merge multiple CSV files into a single output file
///
/// # Arguments
/// * `input_files` - Vector of input CSV file paths
/// * `output_file` - Output CSV file path
/// * `config` - CSV processing configuration
///
/// # Returns
/// * `Result<CsvStats>` - Statistics about the merged file
pub fn merge_csv<P: AsRef<Path>>(
    input_files: &[P],
    output_file: P,
    config: &CsvConfig,
) -> Result<CsvStats> {
    if input_files.is_empty() {
        return Err(UtilsError::Custom("No input files provided".to_string()));
    }

    let output_file = File::create(output_file)?;
    let mut writer = Writer::from_writer(BufWriter::new(output_file));

    let mut total_rows = 0;
    let mut headers_written = false;
    let mut column_stats: Vec<IndexMap<String, usize>> = Vec::new();

    for (file_index, input_path) in input_files.iter().enumerate() {
        let file = File::open(input_path)?;
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(config.delimiter)
            .has_headers(config.has_headers)
            .flexible(config.flexible)
            .buffer_capacity(config.buffer_size)
            .from_reader(BufReader::new(file));

        // Handle headers
        if config.has_headers {
            let headers = reader.headers()?.clone();

            if !headers_written {
                writer.write_record(&headers)?;
                headers_written = true;

                // Initialize column statistics
                column_stats = headers.iter()
                    .map(|_| IndexMap::new())
                    .collect();
            }
        }

        // Process records
        for result in reader.records() {
            let record = result?;
            writer.write_record(&record)?;

            // Update statistics
            for (col_idx, field) in record.iter().enumerate() {
                if col_idx < column_stats.len() {
                    *column_stats[col_idx].entry(field.to_string()).or_insert(0) += 1;
                }
            }

            total_rows += 1;
        }

        log::info!("Processed file {}: {}", file_index + 1, input_path.as_ref().display());
    }

    writer.flush()?;

    // Generate final statistics
    let columns = if config.has_headers && !column_stats.is_empty() {
        // Get headers from first file
        let first_file = File::open(&input_files[0])?;
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(config.delimiter)
            .has_headers(config.has_headers)
            .from_reader(BufReader::new(first_file));

        let headers = reader.headers()?.clone();

        headers.iter().enumerate().map(|(idx, name)| {
            let unique_values = column_stats.get(idx).map(|m| m.len()).unwrap_or(0);
            CsvColumn {
                name: name.to_string(),
                data_type: infer_data_type(&column_stats[idx]),
                nullable: column_stats[idx].contains_key(""),
                unique_values,
            }
        }).collect()
    } else {
        Vec::new()
    };

    Ok(CsvStats {
        row_count: total_rows,
        column_count: columns.len(),
        columns,
        file_size: std::fs::metadata(&output_file)?.len(),
    })
}

/// Convert CSV file to GraphML format
///
/// # Arguments
/// * `csv_file` - Input CSV file path
/// * `graphml_file` - Output GraphML file path
/// * `config` - CSV processing configuration
/// * `node_id_column` - Column to use as node ID (optional)
/// * `edge_columns` - Columns representing edges (source, target)
pub fn csv_to_graphml<P: AsRef<Path>>(
    csv_file: P,
    graphml_file: P,
    config: &CsvConfig,
    node_id_column: Option<&str>,
    edge_columns: Option<(&str, &str)>,
) -> Result<()> {
    use quick_xml::events::{Event, BytesEnd, BytesStart, BytesText};
    use quick_xml::Writer;

    let input_file = File::open(csv_file)?;
    let mut csv_reader = csv::ReaderBuilder::new()
        .delimiter(config.delimiter)
        .has_headers(config.has_headers)
        .flexible(config.flexible)
        .from_reader(BufReader::new(input_file));

    let output_file = File::create(graphml_file)?;
    let mut xml_writer = Writer::new(BufWriter::new(output_file));

    // Write GraphML header
    xml_writer.write_event(Event::Decl(quick_xml::events::BytesDecl::new("1.0", Some("UTF-8"), None)))?;

    let mut graphml_start = BytesStart::new("graphml");
    graphml_start.push_attribute(("xmlns", "http://graphml.graphdrawing.org/xmlns"));
    xml_writer.write_event(Event::Start(graphml_start))?;

    // Define attribute keys
    if config.has_headers {
        let headers = csv_reader.headers()?.clone();
        for (idx, header) in headers.iter().enumerate() {
            if Some(header) != node_id_column {
                let mut key_start = BytesStart::new("key");
                key_start.push_attribute(("id", format!("attr{}", idx).as_str()));
                key_start.push_attribute(("for", "node"));
                key_start.push_attribute(("attr.name", header));
                key_start.push_attribute(("attr.type", "string"));
                xml_writer.write_event(Event::Empty(key_start))?;
            }
        }
    }

    // Start graph
    let mut graph_start = BytesStart::new("graph");
    graph_start.push_attribute(("id", "G"));
    graph_start.push_attribute(("edgedefault", "undirected"));
    xml_writer.write_event(Event::Start(graph_start))?;

    let headers = if config.has_headers {
        Some(csv_reader.headers()?.clone())
    } else {
        None
    };

    // Process nodes
    for (row_idx, result) in csv_reader.records().enumerate() {
        let record = result?;

        let node_id = if let Some(id_col) = node_id_column {
            if let Some(headers) = &headers {
                if let Some(col_idx) = headers.iter().position(|h| h == id_col) {
                    record.get(col_idx).unwrap_or(&format!("node_{}", row_idx)).to_string()
                } else {
                    format!("node_{}", row_idx)
                }
            } else {
                format!("node_{}", row_idx)
            }
        } else {
            format!("node_{}", row_idx)
        };

        // Write node
        let mut node_start = BytesStart::new("node");
        node_start.push_attribute(("id", node_id.as_str()));
        xml_writer.write_event(Event::Start(node_start))?;

        // Write node attributes
        if let Some(headers) = &headers {
            for (col_idx, value) in record.iter().enumerate() {
                if let Some(header) = headers.get(col_idx) {
                    if Some(header) != node_id_column && !value.is_empty() {
                        let mut data_start = BytesStart::new("data");
                        data_start.push_attribute(("key", format!("attr{}", col_idx).as_str()));
                        xml_writer.write_event(Event::Start(data_start))?;
                        xml_writer.write_event(Event::Text(BytesText::new(value)))?;
                        xml_writer.write_event(Event::End(BytesEnd::new("data")))?;
                    }
                }
            }
        }

        xml_writer.write_event(Event::End(BytesEnd::new("node")))?;
    }

    // TODO: Handle edges if edge_columns is provided
    // This would require a second pass through the data

    xml_writer.write_event(Event::End(BytesEnd::new("graph")))?;
    xml_writer.write_event(Event::End(BytesEnd::new("graphml")))?;

    Ok(())
}

/// Add numeric ID column to CSV file
///
/// # Arguments
/// * `input_file` - Input CSV file path
/// * `output_file` - Output CSV file path
/// * `config` - CSV processing configuration
/// * `id_column_name` - Name for the new ID column
/// * `start_id` - Starting value for IDs
pub fn csv_add_numeric_id<P: AsRef<Path>>(
    input_file: P,
    output_file: P,
    config: &CsvConfig,
    id_column_name: &str,
    start_id: u64,
) -> Result<CsvStats> {
    let input = File::open(input_file)?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(config.delimiter)
        .has_headers(config.has_headers)
        .flexible(config.flexible)
        .from_reader(BufReader::new(input));

    let output = File::create(output_file)?;
    let mut writer = Writer::from_writer(BufWriter::new(output));

    let mut row_count = 0;
    let mut column_count = 0;

    // Handle headers
    if config.has_headers {
        let headers = reader.headers()?.clone();
        column_count = headers.len() + 1; // +1 for ID column

        let mut new_headers = StringRecord::new();
        new_headers.push_field(id_column_name);
        for field in headers.iter() {
            new_headers.push_field(field);
        }
        writer.write_record(&new_headers)?;
    }

    // Process records with IDs
    let mut current_id = start_id;
    for result in reader.records() {
        let record = result?;

        let mut new_record = StringRecord::new();
        new_record.push_field(&current_id.to_string());
        for field in record.iter() {
            new_record.push_field(field);
        }

        writer.write_record(&new_record)?;
        current_id += 1;
        row_count += 1;
    }

    writer.flush()?;

    Ok(CsvStats {
        row_count,
        column_count,
        columns: Vec::new(), // Could be populated if needed
        file_size: std::fs::metadata(&output_file)?.len(),
    })
}

/// Analyze CSV file and return statistics
pub fn analyze_csv<P: AsRef<Path>>(
    csv_file: P,
    config: &CsvConfig,
) -> Result<CsvStats> {
    let file = File::open(csv_file)?;
    let file_size = file.metadata()?.len();

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(config.delimiter)
        .has_headers(config.has_headers)
        .flexible(config.flexible)
        .from_reader(BufReader::new(file));

    let mut row_count = 0;
    let mut column_stats: Vec<IndexMap<String, usize>> = Vec::new();

    // Initialize column stats from headers
    if config.has_headers {
        let headers = reader.headers()?.clone();
        column_stats = headers.iter()
            .map(|_| IndexMap::new())
            .collect();
    }

    // Process all records
    for result in reader.records() {
        let record = result?;

        // Initialize column stats if no headers
        if column_stats.is_empty() {
            column_stats = record.iter()
                .map(|_| IndexMap::new())
                .collect();
        }

        // Update statistics
        for (col_idx, field) in record.iter().enumerate() {
            if col_idx < column_stats.len() {
                *column_stats[col_idx].entry(field.to_string()).or_insert(0) += 1;
            }
        }

        row_count += 1;
    }

    // Generate column information
    let columns = if config.has_headers {
        let file = File::open(csv_file)?;
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(config.delimiter)
            .has_headers(config.has_headers)
            .from_reader(BufReader::new(file));

        let headers = reader.headers()?.clone();

        headers.iter().enumerate().map(|(idx, name)| {
            let stats = column_stats.get(idx).unwrap_or(&IndexMap::new());
            CsvColumn {
                name: name.to_string(),
                data_type: infer_data_type(stats),
                nullable: stats.contains_key(""),
                unique_values: stats.len(),
            }
        }).collect()
    } else {
        column_stats.iter().enumerate().map(|(idx, stats)| {
            CsvColumn {
                name: format!("column_{}", idx),
                data_type: infer_data_type(stats),
                nullable: stats.contains_key(""),
                unique_values: stats.len(),
            }
        }).collect()
    };

    Ok(CsvStats {
        row_count,
        column_count: columns.len(),
        columns,
        file_size,
    })
}

/// Infer data type from column values
fn infer_data_type(values: &IndexMap<String, usize>) -> String {
    let mut is_integer = true;
    let mut is_float = true;
    let mut is_boolean = true;

    for value in values.keys() {
        if value.is_empty() {
            continue;
        }

        // Check integer
        if is_integer && value.parse::<i64>().is_err() {
            is_integer = false;
        }

        // Check float
        if is_float && value.parse::<f64>().is_err() {
            is_float = false;
        }

        // Check boolean
        if is_boolean {
            let lower = value.to_lowercase();
            if !matches!(lower.as_str(), "true" | "false" | "0" | "1" | "yes" | "no") {
                is_boolean = false;
            }
        }

        if !is_integer && !is_float && !is_boolean {
            break;
        }
    }

    if is_boolean {
        "boolean".to_string()
    } else if is_integer {
        "integer".to_string()
    } else if is_float {
        "float".to_string()
    } else {
        "string".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_csv_add_numeric_id() {
        let mut input_file = NamedTempFile::new().unwrap();
        writeln!(input_file, "name,age").unwrap();
        writeln!(input_file, "Alice,30").unwrap();
        writeln!(input_file, "Bob,25").unwrap();

        let output_file = NamedTempFile::new().unwrap();
        let config = CsvConfig::default();

        let stats = csv_add_numeric_id(
            input_file.path(),
            output_file.path(),
            &config,
            "id",
            1
        ).unwrap();

        assert_eq!(stats.row_count, 2);
        assert_eq!(stats.column_count, 3);
    }

    #[test]
    fn test_analyze_csv() {
        let mut input_file = NamedTempFile::new().unwrap();
        writeln!(input_file, "name,age,active").unwrap();
        writeln!(input_file, "Alice,30,true").unwrap();
        writeln!(input_file, "Bob,25,false").unwrap();

        let config = CsvConfig::default();
        let stats = analyze_csv(input_file.path(), &config).unwrap();

        assert_eq!(stats.row_count, 2);
        assert_eq!(stats.column_count, 3);
        assert_eq!(stats.columns[0].name, "name");
        assert_eq!(stats.columns[1].data_type, "integer");
        assert_eq!(stats.columns[2].data_type, "boolean");
    }
}