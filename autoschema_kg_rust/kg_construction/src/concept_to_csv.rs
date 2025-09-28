//! Concept to CSV conversion utilities - Rust implementation of Python concept_to_csv.py

use crate::{
    csv_utils::{
        parse_concepts, write_edges_csv, write_full_concept_triple_edges_csv, write_nodes_csv,
        ensure_directory_exists, CsvConfig, FullConceptTripleEdge,
    },
    error::{KgConstructionError, Result},
    graph_traversal::compute_hash_id,
    types::{ConceptMapping, Edge, Node},
};
use csv::{ReaderBuilder, StringRecord};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use uuid::Uuid;

/// Generate a random UUID
pub fn generate_uuid() -> String {
    Uuid::new_v4().to_string()
}

/// Compute hash ID with concept suffix (equivalent to Python function)
pub fn compute_concept_hash_id(text: &str) -> String {
    let text_with_suffix = format!("{}_concept", text);
    compute_hash_id(&text_with_suffix)
}

/// Main concept-to-CSV converter
pub struct ConceptToCsv {
    config: CsvConfig,
}

impl ConceptToCsv {
    pub fn new(config: Option<CsvConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }

    /// Convert all concept triples to CSV format
    /// (equivalent to Python all_concept_triples_csv_to_csv function)
    pub fn all_concept_triples_csv_to_csv(
        &self,
        node_file: impl AsRef<Path>,
        edge_file: impl AsRef<Path>,
        concepts_file: impl AsRef<Path>,
        output_node_file: impl AsRef<Path>,
        output_edge_file: impl AsRef<Path>,
        output_full_concept_triple_edges: impl AsRef<Path>,
    ) -> Result<()> {
        // Ensure all output directories exist
        ensure_directory_exists(&output_node_file)?;
        ensure_directory_exists(&output_edge_file)?;
        ensure_directory_exists(&output_full_concept_triple_edges)?;

        // Load concept mappings
        let concept_mapping = self.load_concept_mappings(&concepts_file)?;

        println!("Loading concepts done.");
        println!("Relation to concepts: {}", concept_mapping.relation_to_concepts.len());
        println!("Node to concepts: {}", concept_mapping.node_to_concepts.len());

        // Process triple nodes and create concept edges
        self.process_triple_nodes(&node_file, &output_edge_file, &concept_mapping)?;

        // Create concept nodes file
        self.create_concept_nodes_file(&output_node_file, &concept_mapping)?;

        // Process triple edges and create full concept triple edges
        self.process_triple_edges(
            &edge_file,
            &output_full_concept_triple_edges,
            &concept_mapping,
        )?;

        Ok(())
    }

    /// Load concept mappings from concepts file
    fn load_concept_mappings(
        &self,
        concepts_file: impl AsRef<Path>,
    ) -> Result<ConceptMapping> {
        let file = File::open(&concepts_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

        let mut concept_mapping = ConceptMapping::default();

        println!("Loading concepts...");
        for result in reader.records() {
            let record = result?;
            self.process_concept_record(&record, &mut concept_mapping)?;
        }

        Ok(concept_mapping)
    }

    /// Process a single concept record
    fn process_concept_record(
        &self,
        record: &StringRecord,
        concept_mapping: &mut ConceptMapping,
    ) -> Result<()> {
        if record.len() < 3 {
            return Err(KgConstructionError::ValidationError(
                "Concept record must have at least 3 fields".to_string(),
            ));
        }

        let node = record[0].to_string();
        let conceptualized_node = record[1].to_string();
        let node_type = record[2].to_string();

        let concepts: Vec<String> = conceptualized_node
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if node_type == "relation" {
            let existing_concepts = concept_mapping
                .relation_to_concepts
                .entry(node.clone())
                .or_default();
            existing_concepts.extend(concepts.clone());
            existing_concepts.sort();
            existing_concepts.dedup();
        } else {
            let existing_concepts = concept_mapping
                .node_to_concepts
                .entry(node.clone())
                .or_default();
            existing_concepts.extend(concepts.clone());
            existing_concepts.sort();
            existing_concepts.dedup();
        }

        // Add to all concepts set
        for concept in concepts {
            concept_mapping.all_concepts.insert(concept);
        }

        Ok(())
    }

    /// Process triple nodes and create concept edges
    fn process_triple_nodes(
        &self,
        node_file: impl AsRef<Path>,
        output_edge_file: impl AsRef<Path>,
        concept_mapping: &ConceptMapping,
    ) -> Result<()> {
        let file = File::open(&node_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

        let mut concept_edges = Vec::new();

        println!("Processing triple nodes...");
        for result in reader.records() {
            let record = result?;
            self.process_node_record(&record, &mut concept_edges, concept_mapping)?;
        }

        // Write concept edges to output file
        write_edges_csv(&output_edge_file, &concept_edges, Some(self.config.clone()))?;

        Ok(())
    }

    /// Process a single node record and create concept edges
    fn process_node_record(
        &self,
        record: &StringRecord,
        concept_edges: &mut Vec<Edge>,
        concept_mapping: &ConceptMapping,
    ) -> Result<()> {
        if record.is_empty() {
            return Ok(());
        }

        let node_name = record[0].to_string();
        let node_id = node_name.clone(); // Assuming node name is the ID

        // Add concepts from node_to_concepts mapping
        if let Some(concepts) = concept_mapping.node_to_concepts.get(&node_name) {
            for concept in concepts {
                let concept_id = compute_concept_hash_id(concept);
                concept_edges.push(Edge {
                    start_id: node_id.clone(),
                    end_id: concept_id,
                    relation: "has_concept".to_string(),
                    edge_type: "Concept".to_string(),
                });
            }
        }

        // Add concepts from the concepts column (if exists)
        if record.len() > 2 {
            let concepts_from_record = parse_concepts(&record[2]);
            for concept in concepts_from_record {
                let concept_id = compute_concept_hash_id(&concept);
                concept_edges.push(Edge {
                    start_id: node_id.clone(),
                    end_id: concept_id,
                    relation: "has_concept".to_string(),
                    edge_type: "Concept".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Create concept nodes file
    fn create_concept_nodes_file(
        &self,
        output_node_file: impl AsRef<Path>,
        concept_mapping: &ConceptMapping,
    ) -> Result<()> {
        let mut concept_nodes = Vec::new();

        println!("Processing concept nodes...");
        for concept in &concept_mapping.all_concepts {
            let concept_id = compute_concept_hash_id(concept);
            concept_nodes.push(Node {
                id: concept_id,
                name: concept.clone(),
                label: "Concept".to_string(),
                node_type: None,
                concepts: Vec::new(),
                synsets: Vec::new(),
            });
        }

        write_nodes_csv(&output_node_file, &concept_nodes, Some(self.config.clone()))?;

        Ok(())
    }

    /// Process triple edges and create full concept triple edges
    fn process_triple_edges(
        &self,
        edge_file: impl AsRef<Path>,
        output_full_concept_triple_edges: impl AsRef<Path>,
        concept_mapping: &ConceptMapping,
    ) -> Result<()> {
        let file = File::open(&edge_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

        let mut full_edges = Vec::new();

        println!("Processing triple edges...");
        for result in reader.records() {
            let record = result?;
            self.process_edge_record(&record, &mut full_edges, concept_mapping)?;
        }

        write_full_concept_triple_edges_csv(
            &output_full_concept_triple_edges,
            &full_edges,
            Some(self.config.clone()),
        )?;

        Ok(())
    }

    /// Process a single edge record
    fn process_edge_record(
        &self,
        record: &StringRecord,
        full_edges: &mut Vec<FullConceptTripleEdge>,
        concept_mapping: &ConceptMapping,
    ) -> Result<()> {
        if record.len() < 6 {
            return Err(KgConstructionError::ValidationError(
                "Edge record must have at least 6 fields".to_string(),
            ));
        }

        let src_id = record[0].to_string();
        let end_id = record[1].to_string();
        let relation = record[2].to_string();
        let concepts_str = record[3].to_string();
        let synsets_str = record[4].to_string();

        let mut original_concepts = parse_concepts(&concepts_str);
        let synsets = parse_concepts(&synsets_str);

        // Add relation concepts if available
        if let Some(relation_concepts) = concept_mapping.relation_to_concepts.get(&relation) {
            for concept in relation_concepts {
                if !original_concepts.contains(concept) {
                    original_concepts.push(concept.clone());
                }
            }
            original_concepts.sort();
            original_concepts.dedup();
        }

        full_edges.push(FullConceptTripleEdge {
            start_id: src_id,
            end_id,
            relation,
            concepts: original_concepts,
            synsets,
            edge_type: "Relation".to_string(),
        });

        Ok(())
    }
}

/// Parallel concept processing for large datasets
pub struct ParallelConceptProcessor {
    converter: ConceptToCsv,
    max_workers: usize,
}

impl ParallelConceptProcessor {
    pub fn new(config: Option<CsvConfig>, max_workers: usize) -> Self {
        Self {
            converter: ConceptToCsv::new(config),
            max_workers,
        }
    }

    /// Process concepts in parallel chunks
    pub fn process_concepts_parallel(
        &self,
        concepts_file: impl AsRef<Path>,
        chunk_size: usize,
    ) -> Result<ConceptMapping> {
        use std::sync::{mpsc, Arc, Mutex};
        use std::thread;

        let file = File::open(&concepts_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

        // Read all records first
        let mut all_records = Vec::new();
        for result in reader.records() {
            all_records.push(result?);
        }

        // Split into chunks
        let chunks: Vec<_> = all_records.chunks(chunk_size).collect();
        let (tx, rx) = mpsc::channel();

        // Process chunks in parallel
        for chunk in chunks {
            let chunk_records = chunk.to_vec();
            let tx = tx.clone();

            thread::spawn(move || {
                let mut local_mapping = ConceptMapping::default();
                let converter = ConceptToCsv::new(None);

                for record in chunk_records {
                    if let Err(e) = converter.process_concept_record(&record, &mut local_mapping) {
                        eprintln!("Error processing record: {}", e);
                    }
                }

                if tx.send(local_mapping).is_err() {
                    eprintln!("Failed to send results from worker thread");
                }
            });
        }

        drop(tx); // Close the sender

        // Collect results from all workers
        let mut final_mapping = ConceptMapping::default();
        for received_mapping in rx {
            // Merge mappings
            for (node, concepts) in received_mapping.node_to_concepts {
                let existing = final_mapping.node_to_concepts.entry(node).or_default();
                existing.extend(concepts);
                existing.sort();
                existing.dedup();
            }

            for (relation, concepts) in received_mapping.relation_to_concepts {
                let existing = final_mapping.relation_to_concepts.entry(relation).or_default();
                existing.extend(concepts);
                existing.sort();
                existing.dedup();
            }

            final_mapping.all_concepts.extend(received_mapping.all_concepts);
        }

        Ok(final_mapping)
    }

    /// Batch process multiple concept files
    pub fn batch_process_concept_files(
        &self,
        input_files: Vec<impl AsRef<Path>>,
        output_directory: impl AsRef<Path>,
    ) -> Result<()> {
        use std::sync::Arc;
        use std::thread;

        let converter = Arc::new(self.converter.clone());
        let mut handles = Vec::new();

        for (i, input_file) in input_files.into_iter().enumerate() {
            let converter: Arc<ConceptToCsv> = Arc::clone(&converter);
            let output_dir = output_directory.as_ref().to_path_buf();

            let handle = thread::spawn(move || -> Result<()> {
                let node_file = input_file.as_ref();
                let edge_file = input_file.as_ref(); // Assuming same base name
                let concepts_file = input_file.as_ref();

                let output_node_file = output_dir.join(format!("concept_nodes_{}.csv", i));
                let output_edge_file = output_dir.join(format!("concept_edges_{}.csv", i));
                let output_full_edges = output_dir.join(format!("full_concept_edges_{}.csv", i));

                converter.all_concept_triples_csv_to_csv(
                    node_file,
                    edge_file,
                    concepts_file,
                    output_node_file,
                    output_edge_file,
                    output_full_edges,
                )?;

                Ok(())
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|e| {
                KgConstructionError::ThreadingError(format!("Thread panicked: {:?}", e))
            })??;
        }

        Ok(())
    }
}

/// Concept validation utilities
pub struct ConceptValidator;

impl ConceptValidator {
    /// Validate concept mapping consistency
    pub fn validate_concept_mapping(mapping: &ConceptMapping) -> Result<()> {
        // Check for empty concepts
        for (node, concepts) in &mapping.node_to_concepts {
            if concepts.is_empty() {
                return Err(KgConstructionError::ValidationError(
                    format!("Node '{}' has no concepts", node),
                ));
            }

            for concept in concepts {
                if concept.trim().is_empty() {
                    return Err(KgConstructionError::ValidationError(
                        format!("Node '{}' has empty concept", node),
                    ));
                }
            }
        }

        for (relation, concepts) in &mapping.relation_to_concepts {
            if concepts.is_empty() {
                return Err(KgConstructionError::ValidationError(
                    format!("Relation '{}' has no concepts", relation),
                ));
            }
        }

        // Check consistency between all_concepts and individual mappings
        let mut all_mapped_concepts = HashSet::new();
        for concepts in mapping.node_to_concepts.values() {
            all_mapped_concepts.extend(concepts.iter().cloned());
        }
        for concepts in mapping.relation_to_concepts.values() {
            all_mapped_concepts.extend(concepts.iter().cloned());
        }

        if all_mapped_concepts != mapping.all_concepts {
            return Err(KgConstructionError::ValidationError(
                "Inconsistency between all_concepts and individual mappings".to_string(),
            ));
        }

        Ok(())
    }

    /// Check for duplicate concepts across different nodes
    pub fn find_duplicate_concepts(mapping: &ConceptMapping) -> HashMap<String, Vec<String>> {
        let mut concept_to_nodes: HashMap<String, Vec<String>> = HashMap::new();

        for (node, concepts) in &mapping.node_to_concepts {
            for concept in concepts {
                concept_to_nodes
                    .entry(concept.clone())
                    .or_default()
                    .push(node.clone());
            }
        }

        // Return only concepts that appear in multiple nodes
        concept_to_nodes
            .into_iter()
            .filter(|(_, nodes)| nodes.len() > 1)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    fn create_test_concepts_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "node,conceptualized_node,node_type").unwrap();
        writeln!(file, "entity1,\"concept1, concept2\",entity").unwrap();
        writeln!(file, "event1,\"action, occurrence\",event").unwrap();
        writeln!(file, "relation1,\"connection, link\",relation").unwrap();
        file
    }

    fn create_test_nodes_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "name:ID,type,concepts,synsets,:LABEL").unwrap();
        writeln!(file, "entity1,Entity,\"[concept1]\",\"[]\",Entity").unwrap();
        writeln!(file, "event1,Event,\"[action]\",\"[]\",Event").unwrap();
        file
    }

    fn create_test_edges_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, ":START_ID,:END_ID,relation,concepts,synsets,:TYPE").unwrap();
        writeln!(file, "entity1,event1,participates,\"[participation]\",\"[]\",Relation").unwrap();
        file
    }

    #[test]
    fn test_concept_to_csv_conversion() {
        let temp_dir = TempDir::new().unwrap();
        let concepts_file = create_test_concepts_file();
        let nodes_file = create_test_nodes_file();
        let edges_file = create_test_edges_file();

        let converter = ConceptToCsv::new(None);

        let output_nodes = temp_dir.path().join("concept_nodes.csv");
        let output_edges = temp_dir.path().join("concept_edges.csv");
        let output_full_edges = temp_dir.path().join("full_concept_edges.csv");

        let result = converter.all_concept_triples_csv_to_csv(
            nodes_file.path(),
            edges_file.path(),
            concepts_file.path(),
            &output_nodes,
            &output_edges,
            &output_full_edges,
        );

        assert!(result.is_ok());
        assert!(output_nodes.exists());
        assert!(output_edges.exists());
        assert!(output_full_edges.exists());
    }

    #[test]
    fn test_concept_mapping_loading() {
        let concepts_file = create_test_concepts_file();
        let converter = ConceptToCsv::new(None);

        let mapping = converter.load_concept_mappings(concepts_file.path()).unwrap();

        assert!(!mapping.node_to_concepts.is_empty());
        assert!(!mapping.relation_to_concepts.is_empty());
        assert!(!mapping.all_concepts.is_empty());

        assert!(mapping.node_to_concepts.contains_key("entity1"));
        assert!(mapping.relation_to_concepts.contains_key("relation1"));
    }

    #[test]
    fn test_compute_concept_hash_id() {
        let hash1 = compute_concept_hash_id("test_concept");
        let hash2 = compute_concept_hash_id("test_concept");
        assert_eq!(hash1, hash2);

        let hash3 = compute_concept_hash_id("different_concept");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_concept_validator() {
        let mut mapping = ConceptMapping::default();
        mapping.node_to_concepts.insert(
            "test_node".to_string(),
            vec!["concept1".to_string(), "concept2".to_string()],
        );
        mapping.all_concepts.insert("concept1".to_string());
        mapping.all_concepts.insert("concept2".to_string());

        let result = ConceptValidator::validate_concept_mapping(&mapping);
        assert!(result.is_ok());

        // Test with empty concepts
        mapping.node_to_concepts.insert("empty_node".to_string(), vec![]);
        let result = ConceptValidator::validate_concept_mapping(&mapping);
        assert!(result.is_err());
    }
}