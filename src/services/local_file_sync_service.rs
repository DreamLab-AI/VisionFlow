// src/services/local_file_sync_service.rs
//! Local File Sync Service with GitHub SHA1 Delta Updates
//!
//! Strategy:
//! 1. Primary source: Local filesystem at /app/data/pages (mounted from host)
//! 2. GitHub API: Only for SHA1 hash comparison to detect changed files
//! 3. Incremental updates: Download only files with different SHA1 from GitHub
//!
//! This avoids pagination issues with 250k+ files by using local baseline.

use crate::adapters::neo4j_ontology_repository::Neo4jOntologyRepository;
use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use crate::services::github::content_enhanced::EnhancedContentAPI;
use crate::services::parsers::{KnowledgeGraphParser, OntologyParser};
use crate::services::ontology_enrichment_service::OntologyEnrichmentService;
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sha1::{Sha1, Digest};

const BATCH_SIZE: usize = 50;
const LOCAL_PAGES_DIR: &str = "/app/data/pages";

#[derive(Clone)]
pub struct LocalFileSyncService {
    content_api: Arc<EnhancedContentAPI>,
    kg_parser: Arc<KnowledgeGraphParser>,
    onto_parser: Arc<OntologyParser>,
    kg_repo: Arc<dyn KnowledgeGraphRepository>,
    onto_repo: Arc<Neo4jOntologyRepository>,
    enrichment_service: Arc<OntologyEnrichmentService>,
}

#[derive(Debug, Clone)]
pub struct SyncStatistics {
    pub total_files: usize,
    pub files_synced_from_local: usize,
    pub files_updated_from_github: usize,
    pub kg_files_processed: usize,
    pub ontology_files_processed: usize,
    pub skipped_files: usize,
    pub errors: Vec<String>,
    pub duration: Duration,
}

impl LocalFileSyncService {
    pub fn new(
        content_api: Arc<EnhancedContentAPI>,
        kg_repo: Arc<dyn KnowledgeGraphRepository>,
        onto_repo: Arc<Neo4jOntologyRepository>,
        enrichment_service: Arc<OntologyEnrichmentService>,
    ) -> Self {
        Self {
            content_api,
            kg_parser: Arc::new(KnowledgeGraphParser::new()),
            onto_parser: Arc::new(OntologyParser::new()),
            kg_repo,
            onto_repo,
            enrichment_service,
        }
    }

    /// Main sync operation: Use local files as baseline, update from GitHub if SHA1 differs
    pub async fn sync_with_github_delta(&self) -> Result<SyncStatistics, String> {
        info!("🔄 Starting local file sync with GitHub SHA1 delta check");
        let start_time = Instant::now();

        let mut stats = SyncStatistics {
            total_files: 0,
            files_synced_from_local: 0,
            files_updated_from_github: 0,
            kg_files_processed: 0,
            ontology_files_processed: 0,
            skipped_files: 0,
            errors: Vec::new(),
            duration: Duration::from_secs(0),
        };

        // Step 1: Read all local markdown files
        let local_files = self.scan_local_pages()?;
        stats.total_files = local_files.len();
        info!("📂 Found {} local markdown files in {}", local_files.len(), LOCAL_PAGES_DIR);

        // Step 2: Get SHA1 hashes from GitHub (lightweight API call - only metadata, not content)
        info!("🔍 Fetching GitHub SHA1 hashes for comparison...");
        let github_sha_map = match self.fetch_github_sha_map().await {
            Ok(map) => {
                info!("✅ Retrieved SHA1 hashes for {} files from GitHub", map.len());
                map
            }
            Err(e) => {
                warn!("⚠️  Failed to fetch GitHub SHA1 map: {}. Proceeding with local files only.", e);
                HashMap::new()
            }
        };

        // Step 3: Process local files in batches
        let mut nodes = HashMap::new();
        let mut edges = HashMap::new();
        let mut public_pages = std::collections::HashSet::new();
        let mut batch_count = 0;

        for (index, local_file) in local_files.iter().enumerate() {
            let file_name = local_file.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            // Calculate local SHA1
            let local_sha = match self.calculate_file_sha1(&local_file) {
                Ok(sha) => sha,
                Err(e) => {
                    error!("Failed to calculate SHA1 for {:?}: {}", local_file, e);
                    stats.errors.push(format!("SHA1 calculation failed: {}", file_name));
                    continue;
                }
            };

            // Check if file needs update from GitHub
            let needs_github_update = github_sha_map.get(file_name)
                .map(|github_sha| github_sha != &local_sha)
                .unwrap_or(false);

            let content = if needs_github_update {
                // Download updated content from GitHub
                info!("🔄 Updating {} from GitHub (SHA1 mismatch)", file_name);
                match self.fetch_and_update_file(&local_file, file_name).await {
                    Ok(content) => {
                        stats.files_updated_from_github += 1;
                        content
                    }
                    Err(e) => {
                        error!("Failed to update {} from GitHub: {}", file_name, e);
                        stats.errors.push(format!("GitHub update failed: {}", file_name));
                        // Fallback to local file
                        match fs::read_to_string(&local_file) {
                            Ok(c) => c,
                            Err(e) => {
                                error!("Failed to read local file {:?}: {}", local_file, e);
                                continue;
                            }
                        }
                    }
                }
            } else {
                // Use local file (already up-to-date or GitHub unavailable)
                match fs::read_to_string(&local_file) {
                    Ok(content) => {
                        stats.files_synced_from_local += 1;
                        content
                    }
                    Err(e) => {
                        error!("Failed to read local file {:?}: {}", local_file, e);
                        stats.errors.push(format!("Read error: {}", file_name));
                        continue;
                    }
                }
            };

            // Process file content (same as github_sync_service.rs)
            if let Err(e) = self.process_file_content(
                file_name,
                &content,
                &mut nodes,
                &mut edges,
                &mut public_pages,
                &mut stats
            ).await {
                error!("Failed to process {}: {}", file_name, e);
                stats.errors.push(format!("Processing error: {}", file_name));
            }

            // Batch save every BATCH_SIZE files
            if (index + 1) % BATCH_SIZE == 0 || index == local_files.len() - 1 {
                batch_count += 1;
                info!("💾 Saving batch {} ({}/{} files processed)",
                    batch_count, index + 1, local_files.len());

                if let Err(e) = self.save_batch(&nodes, &edges).await {
                    error!("Failed to save batch {}: {}", batch_count, e);
                    stats.errors.push(format!("Batch save error: {}", e));
                } else {
                    nodes.clear();
                    edges.clear();
                }
            }

            if (index + 1) % 100 == 0 {
                info!("Progress: {}/{} files processed", index + 1, local_files.len());
            }
        }

        stats.duration = start_time.elapsed();
        info!("✅ Sync complete! {} files from local, {} updated from GitHub in {:?}",
            stats.files_synced_from_local, stats.files_updated_from_github, stats.duration);

        Ok(stats)
    }

    /// Scan local pages directory for markdown files
    fn scan_local_pages(&self) -> Result<Vec<PathBuf>, String> {
        let pages_dir = Path::new(LOCAL_PAGES_DIR);

        if !pages_dir.exists() {
            return Err(format!("Local pages directory does not exist: {}", LOCAL_PAGES_DIR));
        }

        let mut md_files = Vec::new();

        for entry in fs::read_dir(pages_dir)
            .map_err(|e| format!("Failed to read directory {}: {}", LOCAL_PAGES_DIR, e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("md") {
                md_files.push(path);
            }
        }

        md_files.sort();
        Ok(md_files)
    }

    /// Fetch SHA1 hash map from GitHub API (lightweight - only metadata)
    async fn fetch_github_sha_map(&self) -> Result<HashMap<String, String>, String> {
        // Use GitHub tree API for efficient metadata retrieval
        // This avoids pagination issues by getting all file metadata in one call

        // For now, use the existing list_markdown_files (with pagination fix)
        // Future: Implement git tree API for better efficiency
        let github_files = self.content_api.list_markdown_files("").await
            .map_err(|e| format!("GitHub API error: {}", e))?;

        let mut sha_map = HashMap::new();
        for file in github_files {
            sha_map.insert(file.name, file.sha);
        }

        Ok(sha_map)
    }

    /// Calculate SHA1 hash of local file
    fn calculate_file_sha1(&self, file_path: &Path) -> Result<String, String> {
        let content = fs::read(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let mut hasher = Sha1::new();
        hasher.update(&content);
        let result = hasher.finalize();

        Ok(format!("{:x}", result))
    }

    /// Fetch updated file from GitHub and save to local filesystem
    async fn fetch_and_update_file(&self, local_path: &Path, file_name: &str) -> Result<String, String> {
        // Construct GitHub download URL
        let download_url = format!(
            "https://raw.githubusercontent.com/{}/{}/{}/{}",
            std::env::var("GITHUB_OWNER").unwrap_or_else(|_| "jjohare".to_string()),
            std::env::var("GITHUB_REPO").unwrap_or_else(|_| "logseq".to_string()),
            std::env::var("GITHUB_BRANCH").unwrap_or_else(|_| "main".to_string()),
            format!("{}/{}",
                std::env::var("GITHUB_BASE_PATH").unwrap_or_else(|_| "mainKnowledgeGraph/pages".to_string()),
                file_name
            )
        );

        // Fetch content from GitHub
        let content = self.content_api.fetch_file_content(&download_url).await
            .map_err(|e| format!("Failed to fetch from GitHub: {}", e))?;

        // Write updated content to local file
        fs::write(local_path, &content)
            .map_err(|e| format!("Failed to write local file: {}", e))?;

        info!("✅ Updated local file: {:?}", local_path);
        Ok(content)
    }

    /// Process file content (knowledge graph or ontology)
    async fn process_file_content(
        &self,
        file_name: &str,
        content: &str,
        nodes: &mut HashMap<u32, crate::models::node::Node>,
        edges: &mut HashMap<String, crate::models::edge::Edge>,
        public_pages: &mut std::collections::HashSet<String>,
        stats: &mut SyncStatistics,
    ) -> Result<(), String> {
        // Check for knowledge graph file (public:: true)
        if content.lines().take(20).any(|line| line.trim() == "public:: true") {
            let mut parsed = self.kg_parser.parse(content, file_name)
                .map_err(|e| format!("Parse error: {}", e))?;

            // Enrich with ontology
            match self.enrichment_service.enrich_graph(&mut parsed, file_name, content).await {
                Ok((nodes_enriched, edges_enriched)) => {
                    debug!("Enriched {}: {} nodes, {} edges", file_name, nodes_enriched, edges_enriched);
                }
                Err(e) => {
                    warn!("Failed to enrich {}: {}", file_name, e);
                }
            }

            // Add to collections
            let page_name = file_name.trim_end_matches(".md");
            public_pages.insert(page_name.to_string());

            for node in parsed.nodes {
                nodes.insert(node.id, node);
            }

            for edge in parsed.edges {
                edges.insert(edge.id.clone(), edge);
            }

            stats.kg_files_processed += 1;
        }
        // Check for ontology file
        else if content.contains("### OntologyBlock") {
            match self.onto_parser.parse(content, file_name) {
                Ok(onto_data) => {
                    info!("🦉 Extracted ontology from {}: {} classes, {} properties",
                        file_name, onto_data.classes.len(), onto_data.properties.len());

                    // Save ontology data immediately
                    // TODO: Implement save_ontology_data
                    stats.ontology_files_processed += 1;
                }
                Err(e) => {
                    warn!("Failed to parse ontology from {}: {}", file_name, e);
                }
            }
        }
        else {
            stats.skipped_files += 1;
        }

        Ok(())
    }

    /// Save batch to Neo4j
    async fn save_batch(
        &self,
        nodes: &HashMap<u32, crate::models::node::Node>,
        edges: &HashMap<String, crate::models::edge::Edge>,
    ) -> Result<(), String> {
        if nodes.is_empty() && edges.is_empty() {
            return Ok(());
        }

        let graph = crate::models::graph::GraphData {
            nodes: nodes.values().cloned().collect(),
            edges: edges.values().cloned().collect(),
            metadata: crate::models::metadata::MetadataStore::new(),
            id_to_metadata: std::collections::HashMap::new(),
        };

        self.kg_repo.save_graph(&graph).await
            .map_err(|e| format!("Failed to save graph: {}", e))?;

        Ok(())
    }
}
