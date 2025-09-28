//! Data loading and sharding functionality

use crate::{
    error::{KgConstructionError, Result},
    types::ShardConfig,
};
use csv::ReaderBuilder;
use rand::{seq::SliceRandom, thread_rng};
use std::fs::File;
use std::path::Path;

/// Load data with sharding for distributed processing
pub fn load_data_with_shard(
    input_file: impl AsRef<Path>,
    shard_config: ShardConfig,
) -> Result<Vec<Vec<String>>> {
    let file = File::open(&input_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    // Read all records
    let mut data: Vec<Vec<String>> = Vec::new();
    for result in reader.records() {
        let record = result?;
        data.push(record.iter().map(|field| field.to_string()).collect());
    }

    // Shuffle data if requested
    if shard_config.shuffle_data {
        let mut rng = thread_rng();
        data.shuffle(&mut rng);
    }

    // Calculate shard boundaries
    let total_lines = data.len();
    let lines_per_shard = (total_lines + shard_config.num_shards - 1) / shard_config.num_shards;
    let start_idx = shard_config.shard_idx * lines_per_shard;
    let end_idx = ((shard_config.shard_idx + 1) * lines_per_shard).min(total_lines);

    if start_idx >= total_lines {
        return Ok(Vec::new());
    }

    Ok(data[start_idx..end_idx].to_vec())
}

/// Load node data with type information from CSV
pub fn load_node_data_with_types(
    input_file: impl AsRef<Path>,
    shard_config: Option<ShardConfig>,
) -> Result<Vec<(String, String)>> {
    let file = File::open(&input_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut node_list: Vec<(String, String)> = Vec::new();

    for result in reader.records() {
        let record = result?;
        if record.len() >= 2 {
            let node = record[0].to_string();
            let node_type = record[1].to_string();
            node_list.push((node, node_type));
        }
    }

    // Apply sharding if configured
    if let Some(shard_config) = shard_config {
        if shard_config.shuffle_data {
            let mut rng = thread_rng();
            node_list.shuffle(&mut rng);
        }

        let total_lines = node_list.len();
        let lines_per_shard = (total_lines + shard_config.num_shards - 1) / shard_config.num_shards;
        let start_idx = shard_config.shard_idx * lines_per_shard;
        let end_idx = ((shard_config.shard_idx + 1) * lines_per_shard).min(total_lines);

        if start_idx < total_lines {
            node_list = node_list[start_idx..end_idx].to_vec();
        } else {
            node_list.clear();
        }
    }

    Ok(node_list)
}

/// Memory-efficient streaming data loader for large files
pub struct StreamingDataLoader {
    reader: csv::Reader<File>,
    shard_config: Option<ShardConfig>,
    current_index: usize,
    shard_start: usize,
    shard_end: usize,
}

impl StreamingDataLoader {
    pub fn new(
        input_file: impl AsRef<Path>,
        shard_config: Option<ShardConfig>,
    ) -> Result<Self> {
        let file = File::open(&input_file)?;
        let reader = ReaderBuilder::new().has_headers(true).from_reader(file);

        let (shard_start, shard_end) = if let Some(ref config) = shard_config {
            // For streaming, we need to estimate total lines
            // This is a simplified approach - in production, you might want to
            // do a preliminary scan or use file position estimation
            let estimated_lines = 1000000; // Default estimate
            let lines_per_shard = (estimated_lines + config.num_shards - 1) / config.num_shards;
            let start = config.shard_idx * lines_per_shard;
            let end = ((config.shard_idx + 1) * lines_per_shard).min(estimated_lines);
            (start, end)
        } else {
            (0, usize::MAX)
        };

        Ok(Self {
            reader,
            shard_config,
            current_index: 0,
            shard_start,
            shard_end,
        })
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<Vec<String>>> {
        let mut batch = Vec::new();
        let mut count = 0;

        while count < batch_size {
            // Skip records until we reach the shard start
            if self.current_index < self.shard_start {
                match self.reader.records().next() {
                    Some(Ok(_)) => {
                        self.current_index += 1;
                        continue;
                    }
                    Some(Err(e)) => return Err(e.into()),
                    None => break,
                }
            }

            // Stop if we've reached the shard end
            if self.current_index >= self.shard_end {
                break;
            }

            match self.reader.records().next() {
                Some(Ok(record)) => {
                    batch.push(record.iter().map(|field| field.to_string()).collect());
                    count += 1;
                    self.current_index += 1;
                }
                Some(Err(e)) => return Err(e.into()),
                None => break,
            }
        }

        Ok(batch)
    }

    pub fn has_more(&self) -> bool {
        self.current_index < self.shard_end
    }
}

/// Parallel data loading across multiple files
pub fn load_multiple_files_parallel(
    file_paths: Vec<impl AsRef<Path>>,
    max_workers: usize,
) -> Result<Vec<Vec<Vec<String>>>> {
    use std::sync::mpsc;
    use std::thread;

    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();

    // Distribute files across workers
    let files_per_worker = (file_paths.len() + max_workers - 1) / max_workers;

    for worker_id in 0..max_workers {
        let start_idx = worker_id * files_per_worker;
        let end_idx = ((worker_id + 1) * files_per_worker).min(file_paths.len());

        if start_idx >= file_paths.len() {
            break;
        }

        let worker_files: Vec<_> = file_paths[start_idx..end_idx]
            .iter()
            .map(|p| p.as_ref().to_owned())
            .collect();
        let tx = tx.clone();

        let handle = thread::spawn(move || {
            let mut worker_results = Vec::new();

            for file_path in worker_files {
                match load_data_with_shard(&file_path, ShardConfig::default()) {
                    Ok(data) => worker_results.push(data),
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                }
            }

            let _ = tx.send(Ok((worker_id, worker_results)));
        });

        handles.push(handle);
    }

    // Close the sender
    drop(tx);

    // Collect results
    let mut all_results = vec![Vec::new(); max_workers];
    for received in rx {
        match received {
            Ok((worker_id, results)) => {
                all_results[worker_id] = results;
            }
            Err(e) => return Err(e),
        }
    }

    // Wait for all threads
    for handle in handles {
        handle.join().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Thread panicked: {:?}", e))
        })?;
    }

    // Flatten results
    Ok(all_results.into_iter().flatten().collect())
}

/// Data validation utilities
pub fn validate_csv_structure(input_file: impl AsRef<Path>, expected_columns: usize) -> Result<()> {
    let file = File::open(&input_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    // Check headers
    let headers = reader.headers()?;
    if headers.len() != expected_columns {
        return Err(KgConstructionError::ValidationError(
            format!(
                "Expected {} columns, found {} in file {:?}",
                expected_columns,
                headers.len(),
                input_file.as_ref()
            ),
        ));
    }

    // Validate a sample of records
    let mut sample_count = 0;
    const SAMPLE_SIZE: usize = 100;

    for result in reader.records() {
        let record = result?;
        if record.len() != expected_columns {
            return Err(KgConstructionError::ValidationError(
                format!(
                    "Row {} has {} columns, expected {}",
                    sample_count + 1,
                    record.len(),
                    expected_columns
                ),
            ));
        }

        sample_count += 1;
        if sample_count >= SAMPLE_SIZE {
            break;
        }
    }

    Ok(())
}

/// Estimate file size and optimal processing parameters
pub fn estimate_processing_parameters(
    input_file: impl AsRef<Path>,
) -> Result<(usize, usize, usize)> {
    let metadata = std::fs::metadata(&input_file)?;
    let file_size_bytes = metadata.len() as usize;

    // Estimate number of records (assuming average 100 bytes per record)
    let estimated_records = file_size_bytes / 100;

    // Calculate optimal batch size (target 10MB per batch)
    let target_batch_size_bytes = 10 * 1024 * 1024; // 10MB
    let optimal_batch_size = (target_batch_size_bytes / 100).max(1).min(1000);

    // Estimate processing time (assuming 1ms per record)
    let estimated_time_seconds = estimated_records / 1000;

    Ok((estimated_records, optimal_batch_size, estimated_time_seconds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "node,type").unwrap();
        writeln!(file, "entity1,entity").unwrap();
        writeln!(file, "event1,event").unwrap();
        writeln!(file, "relation1,relation").unwrap();
        file
    }

    #[test]
    fn test_load_data_with_shard() {
        let file = create_test_csv();
        let shard_config = ShardConfig {
            shard_idx: 0,
            num_shards: 2,
            shuffle_data: false,
        };

        let result = load_data_with_shard(file.path(), shard_config).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_load_node_data_with_types() {
        let file = create_test_csv();
        let result = load_node_data_with_types(file.path(), None).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], ("entity1".to_string(), "entity".to_string()));
        assert_eq!(result[1], ("event1".to_string(), "event".to_string()));
        assert_eq!(result[2], ("relation1".to_string(), "relation".to_string()));
    }

    #[test]
    fn test_validate_csv_structure() {
        let file = create_test_csv();
        let result = validate_csv_structure(file.path(), 2);
        assert!(result.is_ok());

        let result = validate_csv_structure(file.path(), 3);
        assert!(result.is_err());
    }
}