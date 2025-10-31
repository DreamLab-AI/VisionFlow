// src/services/streaming_sync_service.rs
//! Streaming GitHub Sync Service with Swarm-Based Parallel Processing
//!
//! This service provides fault-tolerant, high-performance GitHub synchronization
//! with the following key features:
//!
//! ## Architecture
//! - **No Batch Accumulation**: Parse → Save immediately using incremental methods
//! - **Swarm Workers**: 4-8 concurrent workers for parallel file processing
//! - **Progress Tracking**: Real-time metrics and progress reporting via channels
//! - **Fault Tolerance**: Continue on errors, don't fail entire sync
//! - **Concurrent-Safe**: Handle concurrent database writes safely with semaphores
//!
//! ## Usage
//! ```rust
//! let service = StreamingSyncService::new(
//!     content_api,
//!     kg_repo,
//!     onto_repo,
//!     Some(8), // max_workers
//! );
//!
//! // Create progress receiver
//! let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
//! service.set_progress_channel(progress_tx);
//!
//! // Start sync in background
//! let sync_handle = tokio::spawn(async move {
//!     service.sync_graphs_streaming().await
//! });
//!
//! // Monitor progress
//! while let Some(progress) = progress_rx.recv().await {
//!     println!("Progress: {}/{} files", progress.files_processed, progress.files_total);
//! }
//!
//! let stats = sync_handle.await??;
//! ```

use crate::repositories::{UnifiedGraphRepository, UnifiedOntologyRepository};
use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use crate::ports::ontology_repository::OntologyRepository;
use crate::services::github::content_enhanced::EnhancedContentAPI;
use crate::services::github::types::GitHubFileBasicMetadata;
use crate::services::parsers::{KnowledgeGraphParser, OntologyParser};
use log::{debug, error, info, warn};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;

/// Default number of concurrent workers in the swarm
const DEFAULT_MAX_WORKERS: usize = 8;

/// Default maximum concurrent database writes
const DEFAULT_MAX_DB_WRITES: usize = 4;

/// File type detection result
#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    KnowledgeGraph, // Contains "public:: true"
    Ontology,       // Contains "- ### OntologyBlock"
    Skip,           // Neither marker found
}

/// Progress information reported during sync
#[derive(Debug, Clone)]
pub struct SyncProgress {
    pub files_total: usize,
    pub files_processed: usize,
    pub files_succeeded: usize,
    pub files_failed: usize,
    pub current_file: String,
    pub errors: Vec<String>,
    pub kg_nodes_saved: usize,
    pub kg_edges_saved: usize,
    pub onto_classes_saved: usize,
    pub onto_properties_saved: usize,
    pub onto_axioms_saved: usize,
}

impl SyncProgress {
    pub fn new(total: usize) -> Self {
        Self {
            files_total: total,
            files_processed: 0,
            files_succeeded: 0,
            files_failed: 0,
            current_file: String::new(),
            errors: Vec::new(),
            kg_nodes_saved: 0,
            kg_edges_saved: 0,
            onto_classes_saved: 0,
            onto_properties_saved: 0,
            onto_axioms_saved: 0,
        }
    }
}

/// Final sync statistics returned after completion
#[derive(Debug, Clone)]
pub struct SyncStatistics {
    pub total_files: usize,
    pub kg_files_processed: usize,
    pub ontology_files_processed: usize,
    pub skipped_files: usize,
    pub failed_files: usize,
    pub errors: Vec<String>,
    pub duration: Duration,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub total_classes: usize,
    pub total_properties: usize,
    pub total_axioms: usize,
}

/// Result of processing a single file
#[derive(Debug, Clone)]
enum FileProcessResult {
    KnowledgeGraph {
        file_name: String,
        nodes: usize,
        edges: usize,
        public_page_name: Option<String>,
    },
    Ontology {
        file_name: String,
        classes: usize,
        properties: usize,
        axioms: usize,
    },
    Skipped {
        file_name: String,
        reason: String,
    },
    Error {
        file_name: String,
        error: String,
    },
}

/// Streaming GitHub Sync Service with swarm-based parallel processing
pub struct StreamingSyncService {
    content_api: Arc<EnhancedContentAPI>,
    kg_parser: Arc<KnowledgeGraphParser>,
    onto_parser: Arc<OntologyParser>,
    kg_repo: Arc<UnifiedGraphRepository>,
    onto_repo: Arc<UnifiedOntologyRepository>,
    max_workers: usize,
    max_db_writes: usize,
    progress_tx: Option<mpsc::UnboundedSender<SyncProgress>>,
}

impl StreamingSyncService {
    /// Create new StreamingSyncService with optional custom worker count
    pub fn new(
        content_api: Arc<EnhancedContentAPI>,
        kg_repo: Arc<UnifiedGraphRepository>,
        onto_repo: Arc<UnifiedOntologyRepository>,
        max_workers: Option<usize>,
    ) -> Self {
        let max_workers = max_workers.unwrap_or(DEFAULT_MAX_WORKERS);
        info!(
            "Initializing StreamingSyncService with {} workers",
            max_workers
        );

        Self {
            content_api,
            kg_parser: Arc::new(KnowledgeGraphParser::new()),
            onto_parser: Arc::new(OntologyParser::new()),
            kg_repo,
            onto_repo,
            max_workers,
            max_db_writes: DEFAULT_MAX_DB_WRITES,
            progress_tx: None,
        }
    }

    /// Set progress channel for real-time progress updates
    pub fn set_progress_channel(&mut self, tx: mpsc::UnboundedSender<SyncProgress>) {
        self.progress_tx = Some(tx);
    }

    /// Synchronize all graphs from GitHub repository using streaming approach
    ///
    /// This method:
    /// 1. Fetches all markdown files from GitHub
    /// 2. Splits files into chunks for worker swarm
    /// 3. Spawns concurrent workers using JoinSet
    /// 4. Each worker: fetch → parse → save immediately (no accumulation)
    /// 5. Collects results and aggregates statistics
    /// 6. Returns partial success statistics
    pub async fn sync_graphs_streaming(&self) -> Result<SyncStatistics, String> {
        info!("🚀 Starting streaming GitHub sync with {} workers", self.max_workers);
        let start_time = Instant::now();

        // Fetch all markdown files
        let files = match self.fetch_all_markdown_files().await {
            Ok(files) => {
                info!("📁 Found {} markdown files in repository", files.len());
                files
            }
            Err(e) => {
                let error_msg = format!("Failed to fetch files from GitHub: {}", e);
                error!("{}", error_msg);
                return Err(error_msg);
            }
        };

        if files.is_empty() {
            warn!("No files to process");
            return Ok(SyncStatistics {
                total_files: 0,
                kg_files_processed: 0,
                ontology_files_processed: 0,
                skipped_files: 0,
                failed_files: 0,
                errors: Vec::new(),
                duration: start_time.elapsed(),
                total_nodes: 0,
                total_edges: 0,
                total_classes: 0,
                total_properties: 0,
                total_axioms: 0,
            });
        }

        // Initialize progress tracking
        let total_files = files.len();
        let mut progress = SyncProgress::new(total_files);

        // Send initial progress
        if let Some(tx) = &self.progress_tx {
            let _ = tx.send(progress.clone());
        }

        // Create semaphore to limit concurrent database writes
        let db_semaphore = Arc::new(Semaphore::new(self.max_db_writes));

        // Create shared progress tracker
        let (result_tx, mut result_rx) = mpsc::unbounded_channel();

        // Spawn worker swarm using JoinSet
        let mut worker_set = JoinSet::new();

        // Split files into chunks for workers
        let chunk_size = (files.len() + self.max_workers - 1) / self.max_workers;

        info!(
            "🐝 Spawning {} workers with ~{} files each",
            self.max_workers, chunk_size
        );

        for (worker_id, chunk) in files.chunks(chunk_size).enumerate() {
            let worker_files = chunk.to_vec();
            let content_api = Arc::clone(&self.content_api);
            let kg_parser = Arc::clone(&self.kg_parser);
            let onto_parser = Arc::clone(&self.onto_parser);
            let kg_repo = Arc::clone(&self.kg_repo);
            let onto_repo = Arc::clone(&self.onto_repo);
            let db_semaphore = Arc::clone(&db_semaphore);
            let result_tx = result_tx.clone();

            worker_set.spawn(async move {
                Self::worker_process_files(
                    worker_id,
                    worker_files,
                    content_api,
                    kg_parser,
                    onto_parser,
                    kg_repo,
                    onto_repo,
                    db_semaphore,
                    result_tx,
                )
                .await
            });
        }

        // Drop the original sender so the channel closes when all workers are done
        drop(result_tx);

        // Collect results from workers in real-time
        let mut public_page_names = HashSet::new();
        let mut stats = SyncStatistics {
            total_files,
            kg_files_processed: 0,
            ontology_files_processed: 0,
            skipped_files: 0,
            failed_files: 0,
            errors: Vec::new(),
            duration: Duration::from_secs(0),
            total_nodes: 0,
            total_edges: 0,
            total_classes: 0,
            total_properties: 0,
            total_axioms: 0,
        };

        // Process results as they come in
        while let Some(result) = result_rx.recv().await {
            match result {
                FileProcessResult::KnowledgeGraph {
                    file_name,
                    nodes,
                    edges,
                    public_page_name,
                } => {
                    stats.kg_files_processed += 1;
                    stats.total_nodes += nodes;
                    stats.total_edges += edges;
                    if let Some(name) = public_page_name {
                        public_page_names.insert(name);
                    }
                    progress.files_processed += 1;
                    progress.files_succeeded += 1;
                    progress.current_file = file_name.clone();
                    progress.kg_nodes_saved = stats.total_nodes;
                    progress.kg_edges_saved = stats.total_edges;
                    debug!("✅ KG file {}: {} nodes, {} edges", file_name, nodes, edges);
                }
                FileProcessResult::Ontology {
                    file_name,
                    classes,
                    properties,
                    axioms,
                } => {
                    stats.ontology_files_processed += 1;
                    stats.total_classes += classes;
                    stats.total_properties += properties;
                    stats.total_axioms += axioms;
                    progress.files_processed += 1;
                    progress.files_succeeded += 1;
                    progress.current_file = file_name.clone();
                    progress.onto_classes_saved = stats.total_classes;
                    progress.onto_properties_saved = stats.total_properties;
                    progress.onto_axioms_saved = stats.total_axioms;
                    debug!(
                        "✅ Ontology file {}: {} classes, {} properties, {} axioms",
                        file_name, classes, properties, axioms
                    );
                }
                FileProcessResult::Skipped { file_name, reason } => {
                    stats.skipped_files += 1;
                    progress.files_processed += 1;
                    progress.current_file = file_name.clone();
                    debug!("⏭️ Skipped {}: {}", file_name, reason);
                }
                FileProcessResult::Error { file_name, error } => {
                    stats.failed_files += 1;
                    let error_msg = format!("{}: {}", file_name, error);
                    stats.errors.push(error_msg.clone());
                    progress.errors.push(error_msg.clone());
                    progress.files_processed += 1;
                    progress.files_failed += 1;
                    progress.current_file = file_name.clone();
                    warn!("❌ Error processing {}: {}", file_name, error);
                }
            }

            // Send progress update
            if let Some(tx) = &self.progress_tx {
                let _ = tx.send(progress.clone());
            }
        }

        // Wait for all workers to complete
        while let Some(result) = worker_set.join_next().await {
            match result {
                Ok(worker_result) => {
                    if let Err(e) = worker_result {
                        warn!("Worker encountered error: {}", e);
                        stats.errors.push(format!("Worker error: {}", e));
                    }
                }
                Err(e) => {
                    error!("Worker panicked: {}", e);
                    stats.errors.push(format!("Worker panic: {}", e));
                }
            }
        }

        stats.duration = start_time.elapsed();

        info!("🎉 Streaming GitHub sync complete in {:?}", stats.duration);
        info!("  ✅ Knowledge graph files: {}", stats.kg_files_processed);
        info!("  ✅ Ontology files: {}", stats.ontology_files_processed);
        info!("  ⏭️  Skipped files: {}", stats.skipped_files);
        info!("  ❌ Failed files: {}", stats.failed_files);
        info!("  📊 Total nodes saved: {}", stats.total_nodes);
        info!("  📊 Total edges saved: {}", stats.total_edges);
        info!("  📚 Total classes saved: {}", stats.total_classes);
        if !stats.errors.is_empty() {
            warn!("  ⚠️  Errors encountered: {}", stats.errors.len());
            for error in &stats.errors {
                warn!("    - {}", error);
            }
        }

        Ok(stats)
    }

    /// Worker function to process a batch of files
    async fn worker_process_files(
        worker_id: usize,
        files: Vec<GitHubFileBasicMetadata>,
        content_api: Arc<EnhancedContentAPI>,
        kg_parser: Arc<KnowledgeGraphParser>,
        onto_parser: Arc<OntologyParser>,
        kg_repo: Arc<UnifiedGraphRepository>,
        onto_repo: Arc<UnifiedOntologyRepository>,
        db_semaphore: Arc<Semaphore>,
        result_tx: mpsc::UnboundedSender<FileProcessResult>,
    ) -> Result<(), String> {
        info!(
            "🐝 Worker {} starting with {} files",
            worker_id,
            files.len()
        );

        for (file_idx, file) in files.iter().enumerate() {
            debug!("[StreamingSync][Worker-{}] Starting to process file {}/{}: {}", worker_id, file_idx + 1, files.len(), file.name);

            let result = Self::process_file_worker(
                worker_id,
                &file,
                &content_api,
                &kg_parser,
                &onto_parser,
                &kg_repo,
                &onto_repo,
                &db_semaphore,
            )
            .await;

            // Send result regardless of success/failure
            if let Err(e) = result_tx.send(result) {
                error!("[StreamingSync][Worker-{}] Failed to send result for {}: {}", worker_id, file.name, e);
            } else {
                debug!("[StreamingSync][Worker-{}] Successfully sent result for {}", worker_id, file.name);
            }

            // Small delay to be nice to GitHub API and reduce contention
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        info!("[StreamingSync][Worker-{}] Completed processing {} files", worker_id, files.len());
        Ok(())
    }

    /// Process a single file: fetch → parse → save immediately
    async fn process_file_worker(
        worker_id: usize,
        file: &GitHubFileBasicMetadata,
        content_api: &Arc<EnhancedContentAPI>,
        kg_parser: &Arc<KnowledgeGraphParser>,
        onto_parser: &Arc<OntologyParser>,
        kg_repo: &Arc<UnifiedGraphRepository>,
        onto_repo: &Arc<UnifiedOntologyRepository>,
        db_semaphore: &Arc<Semaphore>,
    ) -> FileProcessResult {
        debug!("[StreamingSync][Worker-{}] Fetching content from: {}", worker_id, file.download_url);
        let fetch_start = Instant::now();

        // Fetch file content with retry
        let content = match Self::fetch_with_retry(&file.download_url, content_api, 3).await {
            Ok(content) => {
                let fetch_duration = fetch_start.elapsed();
                debug!("[StreamingSync][Worker-{}] Fetched {} bytes in {:?} for {}", worker_id, content.len(), fetch_duration, file.name);
                content
            }
            Err(e) => {
                error!("[StreamingSync][Worker-{}] Failed to fetch {}: {}", worker_id, file.name, e);
                return FileProcessResult::Error {
                    file_name: file.name.clone(),
                    error: format!("Failed to fetch content: {}", e),
                }
            }
        };

        // Detect file type
        let file_type = Self::detect_file_type(&content);
        debug!("[StreamingSync][Worker-{}] Detected file type for {}: {:?}", worker_id, file.name, file_type);

        match file_type {
            FileType::KnowledgeGraph => {
                Self::process_kg_file_streaming(
                    worker_id,
                    file,
                    &content,
                    kg_parser,
                    kg_repo,
                    db_semaphore,
                )
                .await
            }
            FileType::Ontology => {
                Self::process_ontology_file_streaming(
                    worker_id,
                    file,
                    &content,
                    onto_parser,
                    onto_repo,
                    db_semaphore,
                )
                .await
            }
            FileType::Skip => {
                debug!("[StreamingSync][Worker-{}] Skipping {} - no markers found", worker_id, file.name);
                FileProcessResult::Skipped {
                    file_name: file.name.clone(),
                    reason: "No markers found".to_string(),
                }
            }
        }
    }

    /// Process knowledge graph file with immediate saving
    async fn process_kg_file_streaming(
        worker_id: usize,
        file: &GitHubFileBasicMetadata,
        content: &str,
        kg_parser: &Arc<KnowledgeGraphParser>,
        kg_repo: &Arc<UnifiedGraphRepository>,
        db_semaphore: &Arc<Semaphore>,
    ) -> FileProcessResult {
        debug!("[StreamingSync][Worker-{}] Parsing KG file: {}", worker_id, file.name);
        let parse_start = Instant::now();

        // Parse the file
        let parsed_graph = match kg_parser.parse(content, &file.path) {
            Ok(graph) => {
                let parse_duration = parse_start.elapsed();
                debug!("[StreamingSync][Worker-{}] Parsed {} in {:?}: {} nodes, {} edges",
                    worker_id, file.name, parse_duration, graph.nodes.len(), graph.edges.len());
                graph
            }
            Err(e) => {
                error!("[StreamingSync][Worker-{}] Parse error for {}: {}", worker_id, file.name, e);
                return FileProcessResult::Error {
                    file_name: file.name.clone(),
                    error: format!("Parse error: {}", e),
                }
            }
        };

        let node_count = parsed_graph.nodes.len();
        let edge_count = parsed_graph.edges.len();

        // Extract public page name if this is a public page
        let public_page_name = parsed_graph
            .nodes
            .iter()
            .find(|n| {
                n.metadata.get("type").map(|s| s.as_str()) == Some("page")
                    && n.metadata.get("public").map(|s| s.as_str()) == Some("true")
            })
            .map(|n| n.metadata_id.clone());

        debug!("[StreamingSync][Worker-{}] Waiting for DB semaphore to save {} nodes and {} edges", worker_id, node_count, edge_count);
        let semaphore_start = Instant::now();

        // Acquire semaphore permit for database write
        let _permit = db_semaphore.acquire().await.ok();
        let wait_duration = semaphore_start.elapsed();
        debug!("[StreamingSync][Worker-{}] Acquired DB semaphore after {:?}, saving to database", worker_id, wait_duration);

        let save_start = Instant::now();
        let mut nodes_saved = 0;
        let mut nodes_failed = 0;

        // Save nodes incrementally
        for node in &parsed_graph.nodes {
            if let Err(e) = kg_repo.add_node(node).await {
                // Log error but continue processing
                warn!(
                    "[StreamingSync][Worker-{}] Failed to save node {} from file {}: {}",
                    worker_id, node.metadata_id, file.name, e
                );
                nodes_failed += 1;
            } else {
                nodes_saved += 1;
            }
        }

        let mut edges_saved = 0;
        let mut edges_failed = 0;

        // Save edges incrementally
        for edge in &parsed_graph.edges {
            if let Err(e) = kg_repo.add_edge(edge).await {
                // Log error but continue processing
                warn!(
                    "[StreamingSync][Worker-{}] Failed to save edge {} from file {}: {}",
                    worker_id, edge.id, file.name, e
                );
                edges_failed += 1;
            } else {
                edges_saved += 1;
            }
        }

        let save_duration = save_start.elapsed();
        debug!("[StreamingSync][Worker-{}] Saved {} nodes ({} failed) and {} edges ({} failed) in {:?}",
            worker_id, nodes_saved, nodes_failed, edges_saved, edges_failed, save_duration);
        debug!("[StreamingSync][Worker-{}] Released DB semaphore", worker_id);

        FileProcessResult::KnowledgeGraph {
            file_name: file.name.clone(),
            nodes: node_count,
            edges: edge_count,
            public_page_name,
        }
    }

    /// Process ontology file with immediate saving
    async fn process_ontology_file_streaming(
        worker_id: usize,
        file: &GitHubFileBasicMetadata,
        content: &str,
        onto_parser: &Arc<OntologyParser>,
        onto_repo: &Arc<UnifiedOntologyRepository>,
        db_semaphore: &Arc<Semaphore>,
    ) -> FileProcessResult {
        debug!("[StreamingSync][Worker-{}] Parsing ontology file: {}", worker_id, file.name);
        let parse_start = Instant::now();

        // Parse the file
        let ontology_data = match onto_parser.parse(content, &file.path)
        {
            Ok(result) => {
                let parse_duration = parse_start.elapsed();
                debug!("[StreamingSync][Worker-{}] Parsed {} in {:?}: {} classes, {} properties, {} axioms",
                    worker_id, file.name, parse_duration, result.classes.len(), result.properties.len(), result.axioms.len());
                result
            }
            Err(e) => {
                error!("[StreamingSync][Worker-{}] Parse error for {}: {}", worker_id, file.name, e);
                return FileProcessResult::Error {
                    file_name: file.name.clone(),
                    error: format!("Parse error: {}", e),
                }
            }
        };

        let class_count = ontology_data.classes.len();
        let property_count = ontology_data.properties.len();
        let axiom_count = ontology_data.axioms.len();

        debug!("[StreamingSync][Worker-{}] Waiting for DB semaphore to save {} classes, {} properties, {} axioms",
            worker_id, class_count, property_count, axiom_count);
        let semaphore_start = Instant::now();

        // Acquire semaphore permit for database write
        let _permit = db_semaphore.acquire().await.ok();
        let wait_duration = semaphore_start.elapsed();
        debug!("[StreamingSync][Worker-{}] Acquired DB semaphore after {:?}, saving to database", worker_id, wait_duration);

        let save_start = Instant::now();
        let mut classes_saved = 0;
        let mut classes_failed = 0;

        // Save classes incrementally
        for class in &ontology_data.classes {
            if let Err(e) = onto_repo.add_owl_class(class).await {
                warn!(
                    "[StreamingSync][Worker-{}] Failed to save class {} from file {}: {}",
                    worker_id, class.iri, file.name, e
                );
                classes_failed += 1;
            } else {
                classes_saved += 1;
            }
        }

        let mut properties_saved = 0;
        let mut properties_failed = 0;

        // Save properties incrementally
        for property in &ontology_data.properties {
            if let Err(e) = onto_repo.add_owl_property(property).await {
                warn!(
                    "[StreamingSync][Worker-{}] Failed to save property {} from file {}: {}",
                    worker_id, property.iri, file.name, e
                );
                properties_failed += 1;
            } else {
                properties_saved += 1;
            }
        }

        let mut axioms_saved = 0;
        let mut axioms_failed = 0;

        // Save axioms incrementally
        for axiom in &ontology_data.axioms {
            if let Err(e) = onto_repo.add_axiom(axiom).await {
                warn!("[StreamingSync][Worker-{}] Failed to save axiom from file {}: {}", worker_id, file.name, e);
                axioms_failed += 1;
            } else {
                axioms_saved += 1;
            }
        }

        let save_duration = save_start.elapsed();
        debug!("[StreamingSync][Worker-{}] Saved {} classes ({} failed), {} properties ({} failed), {} axioms ({} failed) in {:?}",
            worker_id, classes_saved, classes_failed, properties_saved, properties_failed, axioms_saved, axioms_failed, save_duration);
        debug!("[StreamingSync][Worker-{}] Released DB semaphore", worker_id);

        FileProcessResult::Ontology {
            file_name: file.name.clone(),
            classes: class_count,
            properties: property_count,
            axioms: axiom_count,
        }
    }

    /// Fetch all markdown files from the repository
    async fn fetch_all_markdown_files(&self) -> Result<Vec<GitHubFileBasicMetadata>, String> {
        // Empty string uses the configured base_path
        self.content_api
            .list_markdown_files("")
            .await
            .map_err(|e| format!("GitHub API error: {}", e))
    }

    /// Fetch content with retry logic
    async fn fetch_with_retry(
        url: &str,
        content_api: &Arc<EnhancedContentAPI>,
        max_retries: usize,
    ) -> Result<String, String> {
        let mut retries = 0;
        loop {
            debug!("[StreamingSync][Fetch] Attempt {}/{} for URL: {}", retries + 1, max_retries, url);

            match reqwest::get(url).await {
                Ok(response) => {
                    debug!("[StreamingSync][Fetch] Received response, reading text...");
                    match response.text().await {
                        Ok(text) => {
                            debug!("[StreamingSync][Fetch] Successfully fetched {} bytes", text.len());
                            return Ok(text);
                        }
                        Err(e) => {
                            retries += 1;
                            if retries >= max_retries {
                                error!("[StreamingSync][Fetch] Failed to read response text after {} retries: {}", max_retries, e);
                                return Err(format!("Failed to read response text: {}", e));
                            }
                            let delay = Duration::from_millis(500 * retries as u64);
                            warn!("[StreamingSync][Fetch] Retry {}/{} for {} - text read error, waiting {:?}", retries, max_retries, url, delay);
                            tokio::time::sleep(delay).await;
                        }
                    }
                },
                Err(e) => {
                    retries += 1;
                    if retries >= max_retries {
                        error!("[StreamingSync][Fetch] Failed to fetch after {} retries: {}", max_retries, e);
                        return Err(format!("Failed to fetch: {}", e));
                    }
                    let delay = Duration::from_millis(500 * retries as u64);
                    warn!("[StreamingSync][Fetch] Retry {}/{} for {} - request error, waiting {:?}", retries, max_retries, url, delay);
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Detect file type based on content markers
    fn detect_file_type(content: &str) -> FileType {
        let has_public = content.contains("public:: true");
        let has_ontology = content.contains("- ### OntologyBlock");

        if has_ontology {
            FileType::Ontology
        } else if has_public {
            FileType::KnowledgeGraph
        } else {
            FileType::Skip
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_file_type_knowledge_graph() {
        let content = "Some content\npublic:: true\nMore content";
        assert_eq!(
            StreamingSyncService::detect_file_type(content),
            FileType::KnowledgeGraph
        );
    }

    #[test]
    fn test_detect_file_type_ontology() {
        let content = "Some content\n- ### OntologyBlock\nMore content";
        assert_eq!(
            StreamingSyncService::detect_file_type(content),
            FileType::Ontology
        );
    }

    #[test]
    fn test_detect_file_type_skip() {
        let content = "Some regular markdown content without markers";
        assert_eq!(
            StreamingSyncService::detect_file_type(content),
            FileType::Skip
        );
    }

    #[test]
    fn test_detect_file_type_ontology_priority() {
        // Ontology should have priority over knowledge graph
        let content = "public:: true\n- ### OntologyBlock\nContent";
        assert_eq!(
            StreamingSyncService::detect_file_type(content),
            FileType::Ontology
        );
    }

    #[test]
    fn test_sync_progress_initialization() {
        let progress = SyncProgress::new(100);
        assert_eq!(progress.files_total, 100);
        assert_eq!(progress.files_processed, 0);
        assert_eq!(progress.files_succeeded, 0);
        assert_eq!(progress.files_failed, 0);
    }
}
