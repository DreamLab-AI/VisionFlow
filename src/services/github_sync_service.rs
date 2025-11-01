// src/services/github_sync_service.rs
//! GitHub Sync Service
//!
//! Orchestrates data ingestion from GitHub repository to populate
//! knowledge_graph.db and ontology.db databases.

use crate::repositories::{UnifiedGraphRepository, UnifiedOntologyRepository};
use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use crate::ports::ontology_repository::OntologyRepository;
use crate::services::github::content_enhanced::EnhancedContentAPI;
use crate::services::github::types::GitHubFileBasicMetadata;
use crate::services::parsers::{KnowledgeGraphParser, OntologyParser};
use log::{debug, error, info, warn};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

///
#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    KnowledgeGraph, 
    Ontology,       
    Skip,           
}

///
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

///
#[derive(Debug)]
enum FileProcessResult {
    KnowledgeGraph {
        nodes: usize,
        edges: usize,
    },
    Ontology {
        classes: usize,
        properties: usize,
        axioms: usize,
    },
    Skipped {
        reason: String,
    },
    Error {
        error: String,
    },
}

///
pub struct GitHubSyncService {
    content_api: Arc<EnhancedContentAPI>,
    kg_parser: Arc<KnowledgeGraphParser>,
    onto_parser: Arc<OntologyParser>,
    kg_repo: Arc<UnifiedGraphRepository>,
    onto_repo: Arc<UnifiedOntologyRepository>,
}

impl GitHubSyncService {
    
    pub fn new(
        content_api: Arc<EnhancedContentAPI>,
        kg_repo: Arc<UnifiedGraphRepository>,
        onto_repo: Arc<UnifiedOntologyRepository>,
    ) -> Self {
        Self {
            content_api,
            kg_parser: Arc::new(KnowledgeGraphParser::new()),
            onto_parser: Arc::new(OntologyParser::new()),
            kg_repo,
            onto_repo,
        }
    }

    
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

        
        
        let mut accumulated_nodes: std::collections::HashMap<u32, crate::models::node::Node> =
            std::collections::HashMap::new();
        
        let mut accumulated_edges: std::collections::HashMap<String, crate::models::edge::Edge> =
            std::collections::HashMap::new();

        
        let mut public_page_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        
        let mut accumulated_classes: Vec<crate::ports::ontology_repository::OwlClass> = Vec::new();
        let mut accumulated_properties: Vec<crate::ports::ontology_repository::OwlProperty> =
            Vec::new();
        let mut accumulated_axioms: Vec<crate::ports::ontology_repository::OwlAxiom> = Vec::new();

        
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
                return Ok(stats); 
            }
        };

        stats.total_files = files.len();

        
        let existing_metadata = self.get_existing_file_metadata().await?;
        let mut files_to_process = Vec::new();

        for file in &files {
            if let Some(existing_sha) = existing_metadata.get(&file.name) {
                if existing_sha == &file.sha {
                    debug!("[GitHubSync][SHA1] Skipping unchanged file: {} (SHA: {})", file.name, file.sha);
                    stats.skipped_files += 1;
                    continue;
                }
                info!("[GitHubSync][SHA1] File changed: {} (old: {}, new: {})", file.name, existing_sha, file.sha);
            } else {
                info!("[GitHubSync][SHA1] New file: {}", file.name);
            }
            files_to_process.push(file.clone());
        }

        info!(
            "[GitHubSync][SHA1] Processing {} files ({} unchanged, {} to process)",
            files.len(),
            stats.skipped_files,
            files_to_process.len()
        );

        
        for (index, file) in files_to_process.iter().enumerate() {
            if index > 0 && index % 10 == 0 {
                info!("Progress: {}/{} files processed", index, files_to_process.len());
            }

            
            match self
                .process_file(
                    file,
                    &mut accumulated_nodes,
                    &mut accumulated_edges,
                    &mut public_page_names,
                    &mut accumulated_classes,
                    &mut accumulated_properties,
                    &mut accumulated_axioms,
                )
                .await
            {
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

            
            if index < files.len() - 1 {
                sleep(Duration::from_millis(100)).await;
            }
        }

        
        info!(
            "Filtering linked_page nodes against {} public pages",
            public_page_names.len()
        );
        debug!("[GitHubSync][Filter] Starting node filtering with {} accumulated nodes", accumulated_nodes.len());
        let node_count_before_filter = accumulated_nodes.len();
        let filter_start = Instant::now();
        let mut filtered_count = 0;

        accumulated_nodes.retain(|_id, node| {
            match node.metadata.get("type").map(|s| s.as_str()) {
                Some("page") => {
                    debug!("[GitHubSync][Filter] Keeping page node: {}", node.metadata_id);
                    true
                }
                Some("linked_page") => {
                    
                    let is_public = public_page_names.contains(&node.metadata_id);
                    if !is_public {
                        debug!(
                            "[GitHubSync][Filter] Filtered out linked_page '{}' - not in public pages",
                            node.metadata_id
                        );
                        filtered_count += 1;
                    } else {
                        debug!("[GitHubSync][Filter] Keeping linked_page '{}' - is public", node.metadata_id);
                    }
                    is_public
                }
                _ => {
                    debug!("[GitHubSync][Filter] Keeping unknown type node: {}", node.metadata_id);
                    true
                }
            }
        });
        let nodes_filtered = node_count_before_filter - accumulated_nodes.len();
        info!(
            "[GitHubSync][Filter] Filtered {} linked_page nodes in {:?} (kept {} of {} total nodes)",
            nodes_filtered,
            filter_start.elapsed(),
            accumulated_nodes.len(),
            node_count_before_filter
        );

        
        debug!("[GitHubSync][Filter] Starting edge filtering with {} accumulated edges", accumulated_edges.len());
        let edge_count_before_filter = accumulated_edges.len();
        let edge_filter_start = Instant::now();
        accumulated_edges.retain(|_id, edge| {
            let has_source = accumulated_nodes.contains_key(&edge.source);
            let has_target = accumulated_nodes.contains_key(&edge.target);
            let keep = has_source && has_target;

            if !keep {
                debug!(
                    "[GitHubSync][Filter] Filtered out edge {} -> {}: source_exists={}, target_exists={}",
                    edge.source, edge.target, has_source, has_target
                );
            }
            keep
        });
        let edges_filtered = edge_count_before_filter - accumulated_edges.len();
        info!(
            "[GitHubSync][Filter] Filtered {} orphan edges in {:?} (kept {} of {} total edges)",
            edges_filtered,
            edge_filter_start.elapsed(),
            accumulated_edges.len(),
            edge_count_before_filter
        );

        
        if !accumulated_nodes.is_empty() {
            let node_vec: Vec<crate::models::node::Node> =
                accumulated_nodes.into_values().collect();
            let edge_vec: Vec<crate::models::edge::Edge> =
                accumulated_edges.into_values().collect();
            info!(
                "Saving accumulated knowledge graph: {} unique nodes, {} unique edges",
                node_vec.len(),
                edge_vec.len()
            );

            let mut final_graph = crate::models::graph::GraphData::new();
            final_graph.nodes = node_vec;
            final_graph.edges = edge_vec;

            match self.kg_repo.save_graph(&final_graph).await {
                Ok(_) => {
                    info!("✅ Knowledge graph saved successfully");

                    
                    if let Err(e) = self.update_file_metadata(&files).await {
                        warn!("Failed to update file_metadata: {}", e);
                    }
                }
                Err(e) => {
                    let error_msg = format!("Failed to save accumulated knowledge graph: {}", e);
                    error!("{}", error_msg);
                    stats.errors.push(error_msg);
                }
            }
        }

        
        if !accumulated_classes.is_empty()
            || !accumulated_properties.is_empty()
            || !accumulated_axioms.is_empty()
        {
            info!(
                "Saving accumulated ontology: {} classes, {} properties, {} axioms",
                accumulated_classes.len(),
                accumulated_properties.len(),
                accumulated_axioms.len()
            );

            match self
                .onto_repo
                .save_ontology(
                    &accumulated_classes,
                    &accumulated_properties,
                    &accumulated_axioms,
                )
                .await
            {
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

    
    async fn fetch_all_markdown_files(&self) -> Result<Vec<GitHubFileBasicMetadata>, String> {
        
        let path = ""; 

        debug!("[GitHubSync] Fetching markdown files from GitHub repository");
        let fetch_start = Instant::now();

        let result = self.content_api
            .list_markdown_files(path)
            .await
            .map_err(|e| format!("GitHub API error: {}", e));

        match &result {
            Ok(files) => {
                debug!("[GitHubSync] Fetched {} files in {:?}", files.len(), fetch_start.elapsed());
            }
            Err(e) => {
                error!("[GitHubSync] Failed to fetch file list: {}", e);
            }
        }

        result
    }

    
    async fn process_file(
        &self,
        file: &GitHubFileBasicMetadata,
        accumulated_nodes: &mut std::collections::HashMap<u32, crate::models::node::Node>,
        accumulated_edges: &mut std::collections::HashMap<String, crate::models::edge::Edge>,
        public_page_names: &mut std::collections::HashSet<String>, 
        accumulated_classes: &mut Vec<crate::ports::ontology_repository::OwlClass>,
        accumulated_properties: &mut Vec<crate::ports::ontology_repository::OwlProperty>,
        accumulated_axioms: &mut Vec<crate::ports::ontology_repository::OwlAxiom>,
    ) -> FileProcessResult {
        
        let content = match self
            .fetch_file_content_with_retry(&file.download_url, 3)
            .await
        {
            Ok(content) => content,
            Err(e) => {
                return FileProcessResult::Error {
                    error: format!("Failed to fetch content: {}", e),
                }
            }
        };

        
        let file_type = self.detect_file_type(&content);

        match file_type {
            
            FileType::KnowledgeGraph => {
                self.process_knowledge_graph_file(
                    file,
                    &content,
                    accumulated_nodes,
                    accumulated_edges,
                    public_page_names,
                )
                .await
            }
            FileType::Ontology => {
                self.process_ontology_file(
                    file,
                    &content,
                    accumulated_classes,
                    accumulated_properties,
                    accumulated_axioms,
                )
                .await
            }
            FileType::Skip => FileProcessResult::Skipped {
                reason: "No public:: true or OntologyBlock marker found".to_string(),
            },
        }
    }

    
    async fn process_knowledge_graph_file(
        &self,
        file: &GitHubFileBasicMetadata,
        content: &str,
        accumulated_nodes: &mut std::collections::HashMap<u32, crate::models::node::Node>,
        accumulated_edges: &mut std::collections::HashMap<String, crate::models::edge::Edge>,
        public_page_names: &mut std::collections::HashSet<String>, 
    ) -> FileProcessResult {
        
        let page_name = file.name.strip_suffix(".md").unwrap_or(&file.name);
        public_page_names.insert(page_name.to_string());
        debug!(
            "Added '{}' to public pages set (total: {})",
            page_name,
            public_page_names.len()
        );

        
        let graph_data = match self.kg_parser.parse(content, &file.name) {
            Ok(data) => data,
            Err(e) => {
                return FileProcessResult::Error {
                    error: format!("Parse error: {}", e),
                }
            }
        };

        
        let node_count = graph_data.nodes.len();
        for node in graph_data.nodes {
            accumulated_nodes.insert(node.id, node);
        }

        
        
        let edge_count_before_filter = graph_data.edges.len();
        let mut filtered_edge_count = 0;

        for edge in graph_data.edges {
            
            if accumulated_nodes.contains_key(&edge.source)
                && accumulated_nodes.contains_key(&edge.target)
            {
                accumulated_edges.insert(edge.id.clone(), edge);
                filtered_edge_count += 1;
            } else {
                debug!(
                    "Filtered out edge {} -> {}: references non-public node(s)",
                    edge.source, edge.target
                );
            }
        }

        debug!(
            "Processed {}: {} nodes accumulated, {} edges added ({} filtered out)",
            file.name,
            node_count,
            filtered_edge_count,
            edge_count_before_filter - filtered_edge_count
        );

        FileProcessResult::KnowledgeGraph {
            nodes: node_count,
            edges: filtered_edge_count,
        }
    }

    
    async fn process_ontology_file(
        &self,
        file: &GitHubFileBasicMetadata,
        content: &str,
        accumulated_classes: &mut Vec<crate::ports::ontology_repository::OwlClass>,
        accumulated_properties: &mut Vec<crate::ports::ontology_repository::OwlProperty>,
        accumulated_axioms: &mut Vec<crate::ports::ontology_repository::OwlAxiom>,
    ) -> FileProcessResult {
        use sha1::{Sha1, Digest};

        
        let mut hasher = Sha1::new();
        hasher.update(content.as_bytes());
        let hash_bytes = hasher.finalize();
        let file_sha1 = format!("{:x}", hash_bytes);

        
        let mut ontology_data = match self.onto_parser.parse(content, &file.name) {
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

        
        let now = chrono::Utc::now();
        for class in &mut ontology_data.classes {
            class.markdown_content = Some(content.to_string());
            class.file_sha1 = Some(file_sha1.clone());
            class.last_synced = Some(now);
        }

        
        accumulated_classes.extend(ontology_data.classes);
        accumulated_properties.extend(ontology_data.properties);
        accumulated_axioms.extend(ontology_data.axioms);

        FileProcessResult::Ontology {
            classes: class_count,
            properties: property_count,
            axioms: axiom_count,
        }
    }

    
    fn detect_file_type(&self, content: &str) -> FileType {
        
        let content = content.trim_start_matches('\u{feff}');
        let lines: Vec<&str> = content.lines().take(20).collect();

        debug!("[GitHubSync][FileType] Analyzing file with {} total lines (examining first 20)", content.lines().count());

        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed == "public:: true" {
                debug!("[GitHubSync][FileType] Knowledge Graph detected at line {}", i + 1);
                return FileType::KnowledgeGraph;
            }
        }

        
        if content.contains("### OntologyBlock") {
            debug!("[GitHubSync][FileType] Ontology detected (contains '### OntologyBlock')");
            return FileType::Ontology;
        }

        debug!("[GitHubSync][FileType] File skipped - no markers found");
        FileType::Skip
    }

    
    async fn fetch_file_content_with_retry(
        &self,
        download_url: &str,
        max_retries: u32,
    ) -> Result<String, String> {
        debug!("[GitHubSync][Fetch] Starting fetch for URL: {}", download_url);
        let mut last_error = String::new();

        for attempt in 0..max_retries {
            debug!("[GitHubSync][Fetch] Attempt {}/{} for {}", attempt + 1, max_retries, download_url);

            match self.content_api.fetch_file_content(download_url).await {
                Ok(content) => {
                    debug!("[GitHubSync][Fetch] Successfully fetched {} bytes", content.len());
                    return Ok(content);
                }
                Err(e) => {
                    last_error = e.to_string();
                    if attempt < max_retries - 1 {
                        let delay = Duration::from_secs(2u64.pow(attempt)); 
                        warn!(
                            "[GitHubSync][Fetch] Attempt {}/{} failed: {}. Retrying in {:?}...",
                            attempt + 1,
                            max_retries,
                            last_error,
                            delay
                        );
                        sleep(delay).await;
                    } else {
                        error!("[GitHubSync][Fetch] All {} attempts failed: {}", max_retries, last_error);
                    }
                }
            }
        }

        Err(format!(
            "Failed after {} attempts: {}",
            max_retries, last_error
        ))
    }

    
    async fn get_existing_file_metadata(&self) -> Result<std::collections::HashMap<String, String>, String> {
        let kg_repo = self.kg_repo.clone();

        tokio::task::spawn_blocking(move || {
            let conn = kg_repo.get_connection()
                .map_err(|e| format!("Failed to get database connection: {}", e))?;

            let conn_guard = conn.lock()
                .map_err(|e| format!("Failed to lock connection: {}", e))?;

            let mut stmt = conn_guard.prepare(
                "SELECT file_name, file_blob_sha FROM file_metadata WHERE file_blob_sha IS NOT NULL"
            )
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

            let mut metadata_map = std::collections::HashMap::new();

            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| format!("Failed to query file_metadata: {}", e))?;

            for row in rows {
                let (file_name, sha) = row.map_err(|e| format!("Failed to read row: {}", e))?;
                metadata_map.insert(file_name, sha);
            }

            info!("[GitHubSync][SHA1] Loaded {} existing file metadata entries", metadata_map.len());
            Ok(metadata_map)
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }

    
    async fn update_file_metadata(
        &self,
        files: &[GitHubFileBasicMetadata],
    ) -> Result<(), String> {
        use chrono::Utc;

        let kg_repo = self.kg_repo.clone();
        let files = files.to_vec();

        tokio::task::spawn_blocking(move || {
            use rusqlite::params;

            let conn = kg_repo.get_connection()
                .map_err(|e| format!("Failed to get database connection: {}", e))?;

            let mut conn_guard = conn.lock()
                .map_err(|e| format!("Failed to lock connection: {}", e))?;

            let tx = conn_guard.transaction()
                .map_err(|e| format!("Failed to begin transaction: {}", e))?;

            for file in &files {
                let now = Utc::now().to_rfc3339();

                
                let extension = file.name.rsplit('.').next().unwrap_or("");

                
                tx.execute(
                    r#"
                    INSERT INTO file_metadata
                        (file_name, file_path, file_size, file_extension,
                         file_blob_sha, github_node_id, sha1, content_hash,
                         last_modified, last_content_change, updated_at, processing_status)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'complete')
                    ON CONFLICT(file_name) DO UPDATE SET
                        file_path = ?2,
                        file_size = ?3,
                        file_extension = ?4,
                        file_blob_sha = ?5,
                        sha1 = ?7,
                        content_hash = ?8,
                        last_modified = ?9,
                        last_content_change = CASE
                            WHEN file_blob_sha != ?5 THEN ?10
                            ELSE last_content_change
                        END,
                        updated_at = ?11,
                        processing_status = 'complete',
                        change_count = CASE
                            WHEN file_blob_sha != ?5 THEN COALESCE(change_count, 0) + 1
                            ELSE change_count
                        END
                    "#,
                    params![
                        file.name,
                        file.download_url,
                        file.size as i64,
                        extension,
                        file.sha,
                        "", 
                        file.sha, 
                        file.sha, 
                        now.clone(),
                        now.clone(), 
                        now,
                    ],
                )
                .map_err(|e| format!("Failed to upsert file_metadata for {}: {}", file.name, e))?;
            }

            tx.commit()
                .map_err(|e| format!("Failed to commit file_metadata transaction: {}", e))?;

            info!("✅ Updated file_metadata for {} files", files.len());
            Ok(())
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
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
                Arc::new(tokio::sync::RwLock::new(
                    crate::config::AppFullSettings::default(),
                )),
            )
            .expect("Failed to create test client"),
        );

        let content_api = Arc::new(EnhancedContentAPI::new(client));
        let kg_repo = Arc::new(UnifiedGraphRepository::new(":memory:").unwrap());
        let onto_repo = Arc::new(UnifiedOntologyRepository::new(":memory:").unwrap());

        GitHubSyncService::new(content_api, kg_repo, onto_repo)
    }
}
