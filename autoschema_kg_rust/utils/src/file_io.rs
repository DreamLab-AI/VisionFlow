//! Efficient file I/O utilities with streaming support for large datasets

use crate::{Result, UtilsError};
use bytes::{Bytes, BytesMut};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tokio::fs as async_fs;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader as AsyncBufReader};

/// Configuration for file I/O operations
#[derive(Debug, Clone)]
pub struct FileIOConfig {
    pub buffer_size: usize,
    pub chunk_size: usize,
    pub enable_compression: bool,
    pub compression_level: u32,
    pub enable_memory_mapping: bool,
    pub max_memory_usage: usize,
    pub enable_async: bool,
}

impl Default for FileIOConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024,      // 64KB
            chunk_size: 1024 * 1024,     // 1MB
            enable_compression: false,
            compression_level: 6,
            enable_memory_mapping: false,
            max_memory_usage: 100 * 1024 * 1024, // 100MB
            enable_async: false,
        }
    }
}

/// File processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStats {
    pub file_size: u64,
    pub processed_bytes: u64,
    pub lines_processed: usize,
    pub chunks_processed: usize,
    pub compression_ratio: Option<f64>,
    pub processing_time_ms: u64,
}

/// Streaming file reader for large files
pub struct StreamingReader {
    reader: Box<dyn BufRead>,
    config: FileIOConfig,
    stats: FileStats,
}

impl StreamingReader {
    /// Create a new streaming reader
    pub fn new<P: AsRef<Path>>(file_path: P, config: FileIOConfig) -> Result<Self> {
        let file = File::open(&file_path)?;
        let file_size = file.metadata()?.len();

        let reader: Box<dyn BufRead> = if config.enable_compression {
            Box::new(BufReader::with_capacity(
                config.buffer_size,
                GzDecoder::new(file),
            ))
        } else {
            Box::new(BufReader::with_capacity(config.buffer_size, file))
        };

        Ok(Self {
            reader,
            config,
            stats: FileStats {
                file_size,
                processed_bytes: 0,
                lines_processed: 0,
                chunks_processed: 0,
                compression_ratio: None,
                processing_time_ms: 0,
            },
        })
    }

    /// Read lines in chunks with a callback
    pub fn read_lines_chunked<F>(&mut self, mut callback: F) -> Result<FileStats>
    where
        F: FnMut(&[String]) -> Result<()>,
    {
        let start_time = std::time::Instant::now();
        let mut lines_buffer = Vec::with_capacity(1000);
        let mut total_size = 0;

        for line_result in self.reader.lines() {
            let line = line_result?;
            total_size += line.len() + 1; // +1 for newline
            lines_buffer.push(line);

            if lines_buffer.len() >= 1000 {
                callback(&lines_buffer)?;
                self.stats.lines_processed += lines_buffer.len();
                self.stats.chunks_processed += 1;
                lines_buffer.clear();
            }
        }

        // Process remaining lines
        if !lines_buffer.is_empty() {
            callback(&lines_buffer)?;
            self.stats.lines_processed += lines_buffer.len();
            self.stats.chunks_processed += 1;
        }

        self.stats.processed_bytes = total_size as u64;
        self.stats.processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(self.stats.clone())
    }

    /// Read data in binary chunks
    pub fn read_chunks<F>(&mut self, mut callback: F) -> Result<FileStats>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        let start_time = std::time::Instant::now();
        let mut buffer = vec![0; self.config.chunk_size];

        loop {
            let bytes_read = self.reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            callback(&buffer[..bytes_read])?;
            self.stats.processed_bytes += bytes_read as u64;
            self.stats.chunks_processed += 1;
        }

        self.stats.processing_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(self.stats.clone())
    }
}

/// Streaming file writer for large datasets
pub struct StreamingWriter {
    writer: Box<dyn Write>,
    config: FileIOConfig,
    buffer: Vec<u8>,
    stats: FileStats,
}

impl StreamingWriter {
    /// Create a new streaming writer
    pub fn new<P: AsRef<Path>>(file_path: P, config: FileIOConfig) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_path)?;

        let writer: Box<dyn Write> = if config.enable_compression {
            Box::new(BufWriter::with_capacity(
                config.buffer_size,
                GzEncoder::new(file, Compression::new(config.compression_level)),
            ))
        } else {
            Box::new(BufWriter::with_capacity(config.buffer_size, file))
        };

        Ok(Self {
            writer,
            config: config.clone(),
            buffer: Vec::with_capacity(config.buffer_size),
            stats: FileStats {
                file_size: 0,
                processed_bytes: 0,
                lines_processed: 0,
                chunks_processed: 0,
                compression_ratio: None,
                processing_time_ms: 0,
            },
        })
    }

    /// Write lines with automatic buffering
    pub fn write_lines(&mut self, lines: &[String]) -> Result<()> {
        for line in lines {
            self.write_line(line)?;
        }
        Ok(())
    }

    /// Write a single line
    pub fn write_line(&mut self, line: &str) -> Result<()> {
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\\n")?;
        self.stats.processed_bytes += line.len() as u64 + 1;
        self.stats.lines_processed += 1;
        Ok(())
    }

    /// Write binary data with chunking
    pub fn write_chunk(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data)?;
        self.stats.processed_bytes += data.len() as u64;
        self.stats.chunks_processed += 1;
        Ok(())
    }

    /// Finish writing and get statistics
    pub fn finish(mut self) -> Result<FileStats> {
        self.writer.flush()?;
        Ok(self.stats)
    }
}

/// Parallel file processor using Rayon
pub struct ParallelFileProcessor {
    config: FileIOConfig,
}

impl ParallelFileProcessor {
    pub fn new(config: FileIOConfig) -> Self {
        Self { config }
    }

    /// Process multiple files in parallel
    pub fn process_files_parallel<P, F>(
        &self,
        file_paths: &[P],
        processor: F,
    ) -> Result<Vec<FileStats>>
    where
        P: AsRef<Path> + Sync,
        F: Fn(&Path) -> Result<FileStats> + Sync,
    {
        use rayon::prelude::*;

        file_paths
            .par_iter()
            .map(|path| processor(path.as_ref()))
            .collect()
    }

    /// Split large file into chunks for parallel processing
    pub fn split_file_for_parallel<P: AsRef<Path>>(
        &self,
        input_file: P,
        chunk_size: u64,
        output_dir: P,
    ) -> Result<Vec<PathBuf>> {
        let input_path = input_file.as_ref();
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        let mut file = File::open(input_path)?;
        let file_size = file.metadata()?.len();
        let mut chunk_paths = Vec::new();

        let mut chunk_index = 0;
        let mut bytes_processed = 0;

        while bytes_processed < file_size {
            let chunk_file_name = format!("chunk_{:04}.dat", chunk_index);
            let chunk_path = output_dir.join(chunk_file_name);

            let mut chunk_file = File::create(&chunk_path)?;
            let mut buffer = vec![0; self.config.buffer_size];
            let mut chunk_bytes = 0;

            while chunk_bytes < chunk_size && bytes_processed < file_size {
                let to_read = std::cmp::min(
                    buffer.len(),
                    (chunk_size - chunk_bytes) as usize,
                );
                let bytes_read = file.read(&mut buffer[..to_read])?;

                if bytes_read == 0 {
                    break;
                }

                chunk_file.write_all(&buffer[..bytes_read])?;
                chunk_bytes += bytes_read as u64;
                bytes_processed += bytes_read as u64;
            }

            chunk_file.flush()?;
            chunk_paths.push(chunk_path);
            chunk_index += 1;
        }

        Ok(chunk_paths)
    }
}

/// Memory-efficient file merger
pub struct FileMerger {
    config: FileIOConfig,
}

impl FileMerger {
    pub fn new(config: FileIOConfig) -> Self {
        Self { config }
    }

    /// Merge multiple sorted files into one
    pub fn merge_sorted_files<P: AsRef<Path>>(
        &self,
        input_files: &[P],
        output_file: P,
        compare_fn: fn(&str, &str) -> std::cmp::Ordering,
    ) -> Result<FileStats> {
        let start_time = std::time::Instant::now();

        // Open all input files
        let mut readers: Vec<_> = input_files
            .iter()
            .map(|path| {
                let file = File::open(path)?;
                Ok(BufReader::new(file).lines())
            })
            .collect::<Result<Vec<_>>>()?;

        // Create output writer
        let output_file = File::create(output_file)?;
        let mut writer = BufWriter::new(output_file);

        // Priority queue for merge
        let mut heap = std::collections::BinaryHeap::new();

        // Initialize heap with first line from each file
        for (index, reader) in readers.iter_mut().enumerate() {
            if let Some(Ok(line)) = reader.next() {
                heap.push(std::cmp::Reverse((line, index)));
            }
        }

        let mut lines_written = 0;
        let mut bytes_written = 0;

        // Merge process
        while let Some(std::cmp::Reverse((line, file_index))) = heap.pop() {
            writeln!(writer, "{}", line)?;
            lines_written += 1;
            bytes_written += line.len() + 1;

            // Get next line from the same file
            if let Some(Ok(next_line)) = readers[file_index].next() {
                heap.push(std::cmp::Reverse((next_line, file_index)));
            }
        }

        writer.flush()?;

        Ok(FileStats {
            file_size: bytes_written as u64,
            processed_bytes: bytes_written as u64,
            lines_processed: lines_written,
            chunks_processed: input_files.len(),
            compression_ratio: None,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Simple file concatenation
    pub fn concatenate_files<P: AsRef<Path>>(
        &self,
        input_files: &[P],
        output_file: P,
    ) -> Result<FileStats> {
        let start_time = std::time::Instant::now();
        let output = File::create(output_file)?;
        let mut writer = BufWriter::with_capacity(self.config.buffer_size, output);

        let mut total_bytes = 0;
        let mut total_lines = 0;

        for input_path in input_files {
            let input = File::open(input_path)?;
            let mut reader = BufReader::with_capacity(self.config.buffer_size, input);

            let mut buffer = vec![0; self.config.chunk_size];
            loop {
                let bytes_read = reader.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }

                writer.write_all(&buffer[..bytes_read])?;
                total_bytes += bytes_read;

                // Count newlines for line statistics
                total_lines += buffer[..bytes_read].iter().filter(|&&b| b == b'\\n').count();
            }
        }

        writer.flush()?;

        Ok(FileStats {
            file_size: total_bytes as u64,
            processed_bytes: total_bytes as u64,
            lines_processed: total_lines,
            chunks_processed: input_files.len(),
            compression_ratio: None,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }
}

/// Async file operations for non-blocking I/O
pub struct AsyncFileProcessor {
    config: FileIOConfig,
}

impl AsyncFileProcessor {
    pub fn new(config: FileIOConfig) -> Self {
        Self { config }
    }

    /// Async line processing
    pub async fn process_lines_async<P, F, Fut>(
        &self,
        file_path: P,
        processor: F,
    ) -> Result<FileStats>
    where
        P: AsRef<Path>,
        F: Fn(String) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let start_time = std::time::Instant::now();
        let file = async_fs::File::open(file_path).await?;
        let reader = AsyncBufReader::with_capacity(self.config.buffer_size, file);
        let mut lines = reader.lines();

        let mut stats = FileStats {
            file_size: 0,
            processed_bytes: 0,
            lines_processed: 0,
            chunks_processed: 0,
            compression_ratio: None,
            processing_time_ms: 0,
        };

        while let Some(line) = lines.next_line().await? {
            stats.processed_bytes += line.len() as u64 + 1;
            stats.lines_processed += 1;

            processor(line).await?;
        }

        stats.processing_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(stats)
    }

    /// Async chunk processing
    pub async fn process_chunks_async<P, F, Fut>(
        &self,
        file_path: P,
        processor: F,
    ) -> Result<FileStats>
    where
        P: AsRef<Path>,
        F: Fn(Vec<u8>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let start_time = std::time::Instant::now();
        let mut file = async_fs::File::open(file_path).await?;
        let mut buffer = vec![0; self.config.chunk_size];

        let mut stats = FileStats {
            file_size: 0,
            processed_bytes: 0,
            lines_processed: 0,
            chunks_processed: 0,
            compression_ratio: None,
            processing_time_ms: 0,
        };

        loop {
            let bytes_read = file.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }

            stats.processed_bytes += bytes_read as u64;
            stats.chunks_processed += 1;

            processor(buffer[..bytes_read].to_vec()).await?;
        }

        stats.processing_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(stats)
    }
}

/// Utility functions for common file operations

/// Get file information and statistics
pub fn get_file_info<P: AsRef<Path>>(file_path: P) -> Result<FileInfo> {
    let metadata = std::fs::metadata(&file_path)?;
    let path = file_path.as_ref();

    Ok(FileInfo {
        path: path.to_path_buf(),
        size: metadata.len(),
        is_compressed: is_compressed_file(path),
        estimated_lines: estimate_line_count(path)?,
        mime_type: detect_mime_type(path),
    })
}

/// File information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: PathBuf,
    pub size: u64,
    pub is_compressed: bool,
    pub estimated_lines: Option<usize>,
    pub mime_type: String,
}

fn is_compressed_file(path: &Path) -> bool {
    if let Some(extension) = path.extension() {
        matches!(
            extension.to_str().unwrap_or("").to_lowercase().as_str(),
            "gz" | "bz2" | "xz" | "zip" | "7z"
        )
    } else {
        false
    }
}

fn estimate_line_count(path: &Path) -> Result<Option<usize>> {
    // Sample-based line count estimation
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();

    if file_size == 0 {
        return Ok(Some(0));
    }

    let sample_size = std::cmp::min(64 * 1024, file_size); // 64KB sample
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0; sample_size as usize];
    let bytes_read = reader.read(&mut buffer)?;

    if bytes_read == 0 {
        return Ok(Some(0));
    }

    let newlines_in_sample = buffer[..bytes_read].iter().filter(|&&b| b == b'\\n').count();

    if newlines_in_sample == 0 {
        return Ok(None); // Binary file or single line
    }

    let estimated_lines = (file_size as f64 / bytes_read as f64 * newlines_in_sample as f64) as usize;
    Ok(Some(estimated_lines))
}

fn detect_mime_type(path: &Path) -> String {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => "text/csv".to_string(),
        Some("json") => "application/json".to_string(),
        Some("xml") => "application/xml".to_string(),
        Some("txt") => "text/plain".to_string(),
        Some("md") => "text/markdown".to_string(),
        Some("html") => "text/html".to_string(),
        Some("gz") => "application/gzip".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_streaming_reader() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "line1").unwrap();
        writeln!(temp_file, "line2").unwrap();
        writeln!(temp_file, "line3").unwrap();

        let config = FileIOConfig::default();
        let mut reader = StreamingReader::new(temp_file.path(), config).unwrap();

        let mut collected_lines = Vec::new();
        let stats = reader.read_lines_chunked(|lines| {
            collected_lines.extend_from_slice(lines);
            Ok(())
        }).unwrap();

        assert_eq!(collected_lines.len(), 3);
        assert_eq!(stats.lines_processed, 3);
    }

    #[test]
    fn test_streaming_writer() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = FileIOConfig::default();
        let mut writer = StreamingWriter::new(temp_file.path(), config).unwrap();

        let lines = vec!["line1".to_string(), "line2".to_string()];
        writer.write_lines(&lines).unwrap();
        let stats = writer.finish().unwrap();

        assert_eq!(stats.lines_processed, 2);
    }

    #[test]
    fn test_file_info() {
        let temp_file = NamedTempFile::new().unwrap();
        let info = get_file_info(temp_file.path()).unwrap();

        assert_eq!(info.size, 0);
        assert!(!info.is_compressed);
    }

    #[tokio::test]
    async fn test_async_processing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "async line 1").unwrap();
        writeln!(temp_file, "async line 2").unwrap();

        let config = FileIOConfig::default();
        let processor = AsyncFileProcessor::new(config);

        let mut line_count = 0;
        let stats = processor.process_lines_async(temp_file.path(), |_line| async {
            line_count += 1;
            Ok(())
        }).await.unwrap();

        assert_eq!(stats.lines_processed, 2);
    }
}