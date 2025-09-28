//! Batch processing functionality for concept generation

use crate::{
    error::{KgConstructionError, Result},
    types::{BatchedData, NodeType},
};
use std::collections::HashSet;

/// Build batched data from a list of sessions
pub fn build_batch_data(sessions: Vec<String>, batch_size: usize) -> Vec<Vec<String>> {
    sessions
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Build batched events from a list of nodes with types
pub fn build_batched_events(
    all_node_list: &[(String, String)],
    batch_size: usize,
) -> Vec<Vec<String>> {
    let event_nodes: Vec<String> = all_node_list
        .iter()
        .filter(|(_, node_type)| node_type.to_lowercase() == "event")
        .map(|(node, _)| node.clone())
        .collect();

    event_nodes
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Build batched entities from a list of nodes with types
pub fn build_batched_entities(
    all_node_list: &[(String, String)],
    batch_size: usize,
) -> Vec<Vec<String>> {
    let entity_nodes: Vec<String> = all_node_list
        .iter()
        .filter(|(_, node_type)| node_type.to_lowercase() == "entity")
        .map(|(node, _)| node.clone())
        .collect();

    entity_nodes
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Build batched relations from a list of nodes with types
pub fn build_batched_relations(
    all_node_list: &[(String, String)],
    batch_size: usize,
) -> Vec<Vec<String>> {
    let relations: Vec<String> = all_node_list
        .iter()
        .filter(|(_, node_type)| node_type.to_lowercase() == "relation")
        .map(|(node, _)| node.clone())
        .collect();

    relations
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Build comprehensive batched data structure
pub fn build_comprehensive_batched_data(
    all_node_list: &[(String, String)],
    batch_size: usize,
) -> Result<BatchedData> {
    let events = build_batched_events(all_node_list, batch_size);
    let entities = build_batched_entities(all_node_list, batch_size);
    let relations = build_batched_relations(all_node_list, batch_size);

    Ok(BatchedData {
        events,
        entities,
        relations,
    })
}

/// Create all batches with type information
pub fn create_all_batches(batched_data: &BatchedData) -> Vec<(NodeType, Vec<String>)> {
    let mut all_batches = Vec::new();

    // Add event batches
    for batch in &batched_data.events {
        all_batches.push((NodeType::Event, batch.clone()));
    }

    // Add entity batches
    for batch in &batched_data.entities {
        all_batches.push((NodeType::Entity, batch.clone()));
    }

    // Add relation batches
    for batch in &batched_data.relations {
        all_batches.push((NodeType::Relation, batch.clone()));
    }

    all_batches
}

/// Validate batch data integrity
pub fn validate_batch_data(all_node_list: &[(String, String)]) -> Result<()> {
    let valid_types: HashSet<&str> = ["entity", "event", "relation"].iter().cloned().collect();

    for (node, node_type) in all_node_list {
        if node.trim().is_empty() {
            return Err(KgConstructionError::ValidationError(
                "Empty node name found".to_string(),
            ));
        }

        if !valid_types.contains(node_type.to_lowercase().as_str()) {
            return Err(KgConstructionError::ValidationError(
                format!("Invalid node type '{}' for node '{}'", node_type, node),
            ));
        }
    }

    Ok(())
}

/// Calculate optimal batch size based on available memory and node count
pub fn calculate_optimal_batch_size(
    total_nodes: usize,
    available_memory_mb: usize,
    estimated_node_size_bytes: usize,
) -> usize {
    const MIN_BATCH_SIZE: usize = 1;
    const MAX_BATCH_SIZE: usize = 1000;
    const SAFETY_FACTOR: f64 = 0.8; // Use 80% of available memory

    let available_bytes = (available_memory_mb * 1024 * 1024) as f64 * SAFETY_FACTOR;
    let calculated_batch_size = (available_bytes / estimated_node_size_bytes as f64) as usize;

    // Ensure batch size is within reasonable bounds
    calculated_batch_size
        .max(MIN_BATCH_SIZE)
        .min(MAX_BATCH_SIZE)
        .min(total_nodes)
}

/// Parallel batch processing with thread pool
use std::sync::{Arc, Mutex};
use std::thread;

pub fn process_batches_parallel<F, T>(
    batches: Vec<(NodeType, Vec<String>)>,
    max_workers: usize,
    processor: F,
) -> Result<Vec<T>>
where
    F: Fn(NodeType, Vec<String>) -> Result<T> + Send + Sync + 'static,
    T: Send + 'static,
{
    let processor = Arc::new(processor);
    let results = Arc::new(Mutex::new(Vec::new()));
    let batch_queue = Arc::new(Mutex::new(batches.into_iter().enumerate().collect::<Vec<_>>()));

    let mut handles = Vec::new();

    for _ in 0..max_workers {
        let processor = Arc::clone(&processor);
        let results = Arc::clone(&results);
        let batch_queue = Arc::clone(&batch_queue);

        let handle = thread::spawn(move || -> Result<()> {
            loop {
                let batch_item = {
                    let mut queue = batch_queue.lock().map_err(|e| {
                        KgConstructionError::ThreadingError(format!("Lock poisoned: {}", e))
                    })?;
                    queue.pop()
                };

                match batch_item {
                    Some((index, (node_type, batch))) => {
                        match processor(node_type, batch) {
                            Ok(result) => {
                                let mut results = results.lock().map_err(|e| {
                                    KgConstructionError::ThreadingError(format!("Lock poisoned: {}", e))
                                })?;
                                results.push((index, result));
                            }
                            Err(e) => {
                                eprintln!("Error processing batch {}: {}", index, e);
                                return Err(e);
                            }
                        }
                    }
                    None => break, // No more batches to process
                }
            }
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

    // Extract results and sort by original index
    let mut results = Arc::try_unwrap(results)
        .map_err(|_| KgConstructionError::ThreadingError("Failed to unwrap results".to_string()))?
        .into_inner()
        .map_err(|e| KgConstructionError::ThreadingError(format!("Lock poisoned: {}", e)))?;

    results.sort_by_key(|(index, _)| *index);
    Ok(results.into_iter().map(|(_, result)| result).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_batch_data() {
        let sessions = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let batches = build_batch_data(sessions, 2);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec!["a", "b"]);
        assert_eq!(batches[1], vec!["c", "d"]);
    }

    #[test]
    fn test_build_batched_events() {
        let node_list = vec![
            ("node1".to_string(), "event".to_string()),
            ("node2".to_string(), "entity".to_string()),
            ("node3".to_string(), "event".to_string()),
        ];
        let batches = build_batched_events(&node_list, 2);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], vec!["node1", "node3"]);
    }

    #[test]
    fn test_validate_batch_data() {
        let valid_data = vec![
            ("node1".to_string(), "event".to_string()),
            ("node2".to_string(), "entity".to_string()),
        ];
        assert!(validate_batch_data(&valid_data).is_ok());

        let invalid_data = vec![
            ("node1".to_string(), "invalid".to_string()),
        ];
        assert!(validate_batch_data(&invalid_data).is_err());
    }

    #[test]
    fn test_calculate_optimal_batch_size() {
        let batch_size = calculate_optimal_batch_size(1000, 100, 1000);
        assert!(batch_size > 0);
        assert!(batch_size <= 1000);
    }
}