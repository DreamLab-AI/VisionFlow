// src/services/github_sync_service.rs
//! GitHub Sync Service
//!
//! Orchestrates data ingestion from GitHub repository to populate
//! knowledge_graph.db and ontology.db databases.

use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use crate::ports::ontology_repository::OntologyRepository;
use crate::services::github::content_enhanced::EnhancedContentAPI;
use crate::services::github::types::GitHubFileBasicMetadata;
use crate::services::parsers::{KnowledgeGraphParser, OntologyParser};
use log::{debug, error, info, warn};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// File type detection result
#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    KnowledgeGraph,  // Contains "public:: true"
    Ontology,        // Contains "- ### OntologyBlock"
    Skip,            // Neither marker found
}

/// Sync statistics returned after ingestion
#[derive(Debug, Clone)]
pub struct SyncStatistics {
    pub total_files: usize,
    pub kg_files_processed: usize,
    pub ontology_files_processed: usize,
    pub skipped_files: usize,
    pub errors: Vec<String>,
    pub duration: Duration,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Result of processing a single file
#[derive(Debug)]
enum FileProcessResult {
    KnowledgeGraph { nodes: usize, edges: usize },
    Ontology { classes: usize, properties: usize, axioms: usize },
    Skipped { reason: String },
    Error { error: String },
}

/// GitHub Sync Service - orchestrates data ingestion
pub struct GitHubSyncService {
    content_api: Arc<EnhancedContentAPI>,
    kg_parser: Arc<KnowledgeGraphParser>,
    onto_parser: Arc<OntologyParser>,
    kg_repo: Arc<SqliteKnowledgeGraphRepository>,
    onto_repo: Arc<SqliteOntologyRepository>,
}

impl GitHubSyncService {
    /// Create new GitHubSyncService
    pub fn new(
        content_api: Arc<EnhancedContentAPI>,
        kg_repo: Arc<SqliteKnowledgeGraphRepository>,
        onto_repo: Arc<SqliteOntologyRepository>,
    ) -> Self {
        Self {
            content_api,
            kg_parser: Arc::new(KnowledgeGraphParser::new()),
            onto_parser: Arc::new(OntologyParser::new()),
            kg_repo,
            onto_repo,
        }
    }

    /// Synchronize all graphs from GitHub repository
    pub async fn sync_graphs(&self) -> Result<SyncStatistics, String> {
        info!("Starting GitHub data synchronization...");
        let start_time = Instant::now();

        let mut stats = SyncStatistics {
            total_files: 0,
            kg_files_processed: 0,
            ontology_files_processed: 0,
            skipped_files: 0,
            errors: Vec::new(),
            duration: Duration::from_secs(0),
            total_nodes: 0,
            total_edges: 0,
        };

        // Accumulate all knowledge graph data before saving
        // Use HashMap for automatic node deduplication by ID
        let mut accumulated_nodes: std::collections::HashMap<u32, crate::models::node::Node> = std::collections::HashMap::new();
        // Use HashMap for automatic edge deduplication by ID to prevent UNIQUE constraint violations
        let mut accumulated_edges: std::collections::HashMap<String, crate::models::edge::Edge> = std::collections::HashMap::new();

        // Accumulate all ontology data before saving
        let mut accumulated_classes: Vec<crate::ports::ontology_repository::OwlClass> = Vec::new();
        let mut accumulated_properties: Vec<crate::ports::ontology_repository::OwlProperty> = Vec::new();
        let mut accumulated_axioms: Vec<crate::ports::ontology_repository::OwlAxiom> = Vec::new();

        // Fetch all markdown files from repository
        let files = match self.fetch_all_markdown_files().await {
            Ok(files) => {
                info!("Found {} markdown files in repository", files.len());
                files
            }
            Err(e) => {
                let error_msg = format!("Failed to fetch files from GitHub: {}", e);
                error!("{}", error_msg);
                stats.errors.push(error_msg);
                stats.duration = start_time.elapsed();
                return Ok(stats); // Return empty stats, allow manual import
            }
        };

        stats.total_files = files.len();

        // Process each file
        for (index, file) in files.iter().enumerate() {
            if index > 0 && index % 10 == 0 {
                info!("Progress: {}/{} files processed", index, files.len());
            }

            match self.process_file(file, &mut accumulated_nodes, &mut accumulated_edges, &mut accumulated_classes, &mut accumulated_properties, &mut accumulated_axioms).await {
                FileProcessResult::KnowledgeGraph { nodes, edges } => {
                    stats.kg_files_processed += 1;
                    stats.total_nodes += nodes;
                    stats.total_edges += edges;
                    debug!("KG file {}: {} nodes, {} edges", file.name, nodes, edges);
                }
                FileProcessResult::Ontology {
                    classes,
                    properties,
                    axioms,
                } => {
                    stats.ontology_files_processed += 1;
                    debug!(
                        "Ontology file {}: {} classes, {} properties, {} axioms",
                        file.name, classes, properties, axioms
                    );
                }
                FileProcessResult::Skipped { reason } => {
                    stats.skipped_files += 1;
                    debug!("Skipped {}: {}", file.name, reason);
                }
                FileProcessResult::Error { error } => {
                    stats.errors.push(format!("{}: {}", file.name, error));
                    warn!("Error processing {}: {}", file.name, error);
                }
            }

            // Rate limiting: small delay between files to be nice to GitHub API
            if index < files.len() - 1 {
                sleep(Duration::from_millis(100)).await;
            }
        }

        // Convert HashMap to Vec and create GraphData for saving
        if !accumulated_nodes.is_empty() {
            let node_vec: Vec<crate::models::node::Node> = accumulated_nodes.into_values().collect();
            let edge_vec: Vec<crate::models::edge::Edge> = accumulated_edges.into_values().collect();
            info!("Saving accumulated knowledge graph: {} unique nodes, {} unique edges",
                  node_vec.len(), edge_vec.len());

            let mut final_graph = crate::models::graph::GraphData::new();
            final_graph.nodes = node_vec;
            final_graph.edges = edge_vec;

            match self.kg_repo.save_graph(&final_graph).await {
                Ok(_) => info!("✅ Knowledge graph saved successfully"),
                Err(e) => {
                    let error_msg = format!("Failed to save accumulated knowledge graph: {}", e);
                    error!("{}", error_msg);
                    stats.errors.push(error_msg);
                }
            }
        }

        // Save accumulated ontology data in a single batch operation
        if !accumulated_classes.is_empty() || !accumulated_properties.is_empty() || !accumulated_axioms.is_empty() {
            info!("Saving accumulated ontology: {} classes, {} properties, {} axioms",
                  accumulated_classes.len(), accumulated_properties.len(), accumulated_axioms.len());

            match self.onto_repo.save_ontology(&accumulated_classes, &accumulated_properties, &accumulated_axioms).await {
                Ok(_) => info!("✅ Ontology data saved successfully"),
                Err(e) => {
                    let error_msg = format!("Failed to save accumulated ontology: {}", e);
                    error!("{}", error_msg);
                    stats.errors.push(error_msg);
                }
            }
        }

        stats.duration = start_time.elapsed();

        info!("GitHub sync complete in {:?}", stats.duration);
        info!("  Knowledge graph files: {}", stats.kg_files_processed);
        info!("  Ontology files: {}", stats.ontology_files_processed);
        info!("  Skipped files: {}", stats.skipped_files);
        if !stats.errors.is_empty() {
            warn!("  Errors encountered: {}", stats.errors.len());
        }

        Ok(stats)
    }

    /// Fetch all markdown files from the repository
    async fn fetch_all_markdown_files(&self) -> Result<Vec<GitHubFileBasicMetadata>, String> {
        // Use the base_path from GitHub config (mainKnowledgeGraph/pages)
        let path = ""; // Empty string will use the configured base_path

        self.content_api
            .list_markdown_files(path)
            .await
            .map_err(|e| format!("GitHub API error: {}", e))
    }

    /// Process a single file
    async fn process_file(
        &self,
        file: &GitHubFileBasicMetadata,
        accumulated_nodes: &mut std::collections::HashMap<u32, crate::models::node::Node>,
        accumulated_edges: &mut std::collections::HashMap<String, crate::models::edge::Edge>,
        accumulated_classes: &mut Vec<crate::ports::ontology_repository::OwlClass>,
        accumulated_properties: &mut Vec<crate::ports::ontology_repository::OwlProperty>,
        accumulated_axioms: &mut Vec<crate::ports::ontology_repository::OwlAxiom>,
    ) -> FileProcessResult {
        // Fetch file content
        let content = match self.fetch_file_content_with_retry(&file.download_url, 3).await {
            Ok(content) => content,
            Err(e) => {
                return FileProcessResult::Error {
                    error: format!("Failed to fetch content: {}", e),
                }
            }
        };

        // Detect file type
        let file_type = self.detect_file_type(&content);

        match file_type {
            FileType::KnowledgeGraph => self.process_knowledge_graph_file(file, &content, accumulated_nodes, accumulated_edges).await,
            FileType::Ontology => self.process_ontology_file(file, &content, accumulated_classes, accumulated_properties, accumulated_axioms).await,
            FileType::Skip => FileProcessResult::Skipped {
                reason: "No public:: true or OntologyBlock marker found".to_string(),
            },
        }
    }

    /// Process a knowledge graph file
    async fn process_knowledge_graph_file(
        &self,
        file: &GitHubFileBasicMetadata,
        content: &str,
        accumulated_nodes: &mut std::collections::HashMap<u32, crate::models::node::Node>,
        accumulated_edges: &mut std::collections::HashMap<String, crate::models::edge::Edge>,
    ) -> FileProcessResult {
        // Parse the file
        let graph_data = match self.kg_parser.parse(content, &file.name) {
            Ok(data) => data,
            Err(e) => {
                return FileProcessResult::Error {
                    error: format!("Parse error: {}", e),
                }
            }
        };

        let node_count = graph_data.nodes.len();
        let edge_count = graph_data.edges.len();

        // Accumulate nodes in HashMap for automatic deduplication by ID
        for node in graph_data.nodes {
            accumulated_nodes.insert(node.id, node);
        }

        // Accumulate edges in HashMap for automatic deduplication by edge ID
        for edge in graph_data.edges {
            accumulated_edges.insert(edge.id.clone(), edge);
        }

        FileProcessResult::KnowledgeGraph {
            nodes: node_count,
            edges: edge_count,
        }
    }

    /// Process an ontology file
    async fn process_ontology_file(
        &self,
        file: &GitHubFileBasicMetadata,
        content: &str,
        accumulated_classes: &mut Vec<crate::ports::ontology_repository::OwlClass>,
        accumulated_properties: &mut Vec<crate::ports::ontology_repository::OwlProperty>,
        accumulated_axioms: &mut Vec<crate::ports::ontology_repository::OwlAxiom>,
    ) -> FileProcessResult {
        // Parse the file
        let ontology_data = match self.onto_parser.parse(content, &file.name) {
            Ok(data) => data,
            Err(e) => {
                return FileProcessResult::Error {
                    error: format!("Parse error: {}", e),
                }
            }
        };

        let class_count = ontology_data.classes.len();
        let property_count = ontology_data.properties.len();
        let axiom_count = ontology_data.axioms.len();

        // Accumulate ontology data (will be saved in batch at end)
        accumulated_classes.extend(ontology_data.classes);
        accumulated_properties.extend(ontology_data.properties);
        accumulated_axioms.extend(ontology_data.axioms);

        FileProcessResult::Ontology {
            classes: class_count,
            properties: property_count,
            axioms: axiom_count,
        }
    }

    /// Detect file type based on content markers
    fn detect_file_type(&self, content: &str) -> FileType {
        // Remove UTF-8 BOM if present
        let content = content.trim_start_matches('\u{feff}');
        let lines: Vec<&str> = content.lines().take(20).collect();

        // Debug: Log first 3 lines
        for (i, line) in lines.iter().take(3).enumerate() {
            debug!("detect_file_type line {}: {:?}", i + 1, line);
        }

        // Check for "public:: true" (knowledge graph marker)
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            debug!("Line {} check: '{}' == 'public:: true' ? {}",
                   i + 1, trimmed, trimmed == "public:: true");

            if trimmed == "public:: true" {
                debug!("✅ Knowledge Graph detected!");
                return FileType::KnowledgeGraph;
            }
        }

        // Check for "- ### OntologyBlock" (ontology marker)
        if content.contains("### OntologyBlock") {
            debug!("✅ Ontology detected!");
            return FileType::Ontology;
        }

        debug!("⏭️ File skipped (no markers found)");
        FileType::Skip
    }

    /// Fetch file content with retry logic
    async fn fetch_file_content_with_retry(
        &self,
        download_url: &str,
        max_retries: u32,
    ) -> Result<String, String> {
        let mut last_error = String::new();

        for attempt in 0..max_retries {
            match self.content_api.fetch_file_content(download_url).await {
                Ok(content) => return Ok(content),
                Err(e) => {
                    last_error = e.to_string();
                    if attempt < max_retries - 1 {
                        let delay = Duration::from_secs(2u64.pow(attempt)); // Exponential backoff
                        warn!(
                            "Fetch failed (attempt {}/{}): {}. Retrying in {:?}...",
                            attempt + 1,
                            max_retries,
                            last_error,
                            delay
                        );
                        sleep(delay).await;
                    }
                }
            }
        }

        Err(format!(
            "Failed after {} attempts: {}",
            max_retries, last_error
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_file_type_knowledge_graph() {
        let content = "public:: true\n---\n# Test Page";
        let service = create_test_service();

        assert_eq!(service.detect_file_type(content), FileType::KnowledgeGraph);
    }

    #[test]
    fn test_detect_file_type_ontology() {
        let content = "Some content\n- ### OntologyBlock\n  - owl_class:: Test";
        let service = create_test_service();

        assert_eq!(service.detect_file_type(content), FileType::Ontology);
    }

    #[test]
    fn test_detect_file_type_skip() {
        let content = "Just some regular markdown content";
        let service = create_test_service();

        assert_eq!(service.detect_file_type(content), FileType::Skip);
    }

    fn create_test_service() -> GitHubSyncService {
        // This is a placeholder for testing - in real tests, use mock repositories
        use crate::services::github::api::GitHubClient;
        use std::sync::Arc;

        let client = Arc::new(
            GitHubClient::new(
                crate::services::github::config::GitHubConfig {
                    token: "test".to_string(),
                    owner: "test".to_string(),
                    repo: "test".to_string(),
                    base_path: "test".to_string(),
                },
                Arc::new(tokio::sync::RwLock::new(crate::config::AppFullSettings::default())),
            )
            .expect("Failed to create test client"),
        );

        let content_api = Arc::new(EnhancedContentAPI::new(client));
        let kg_repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").unwrap());
        let onto_repo = Arc::new(SqliteOntologyRepository::new(":memory:").unwrap());

        GitHubSyncService::new(content_api, kg_repo, onto_repo)
    }
}
