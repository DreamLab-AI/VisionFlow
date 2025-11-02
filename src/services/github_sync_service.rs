// src/services/github_sync_service.rs
//! GitHub Sync Service
//!
//! Handles synchronization of markdown files from GitHub repository to database.
//! Uses batch processing to avoid memory issues with large repositories.

use crate::repositories::{UnifiedGraphRepository, UnifiedOntologyRepository};
use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use crate::ports::ontology_repository::OntologyRepository;
use crate::services::github::content_enhanced::EnhancedContentAPI;
use crate::services::github::types::GitHubFileBasicMetadata;
use crate::services::parsers::{KnowledgeGraphParser, OntologyParser};
use log::{debug, error, info, warn};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BATCH_SIZE: usize = 50; // Save to database every 50 files

#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    KnowledgeGraph,
    Ontology,
    Skip,
}

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

    /// Synchronize graphs from GitHub - processes in batches with progress logging
    pub async fn sync_graphs(&self) -> Result<SyncStatistics, String> {
        info!("🔄 Starting GitHub sync (batch size: {})", BATCH_SIZE);
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

        // Fetch files
        let files = match self.fetch_all_markdown_files().await {
            Ok(files) => {
                info!("📂 Found {} markdown files", files.len());
                files
            }
            Err(e) => {
                let error_msg = format!("Failed to fetch files: {}", e);
                error!("{}", error_msg);
                stats.errors.push(error_msg);
                stats.duration = start_time.elapsed();
                return Ok(stats);
            }
        };

        stats.total_files = files.len();

        // SHA1 filtering - only process changed files (unless FORCE_FULL_SYNC is set)
        let force_full_sync = std::env::var("FORCE_FULL_SYNC")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let files_to_process = if force_full_sync {
            info!("🔄 FORCE_FULL_SYNC enabled - processing ALL {} files (bypassing SHA1 filter)", files.len());
            files.clone()
        } else {
            match self.filter_changed_files(&files).await {
                Ok(filtered) => {
                    info!("📋 Processing {} changed files ({} unchanged)",
                        filtered.len(), files.len() - filtered.len());
                    stats.skipped_files = files.len() - filtered.len();
                    filtered
                }
                Err(e) => {
                    error!("SHA1 filter failed: {}", e);
                    files.clone() // Process all if filter fails
                }
            }
        };

        // Clone files_to_process for metadata update later
        let all_files_to_process = files_to_process.clone();

        // Process in batches
        for (batch_idx, batch) in files_to_process.chunks(BATCH_SIZE).enumerate() {
            let batch_start = Instant::now();
            info!("📦 Processing batch {}/{} ({} files)",
                batch_idx + 1,
                (files_to_process.len() + BATCH_SIZE - 1) / BATCH_SIZE,
                batch.len()
            );

            match self.process_batch(batch, &mut stats).await {
                Ok(_) => {
                    info!("✅ Batch {} completed in {:?}", batch_idx + 1, batch_start.elapsed());
                }
                Err(e) => {
                    error!("❌ Batch {} failed: {}", batch_idx + 1, e);
                    stats.errors.push(format!("Batch {}: {}", batch_idx + 1, e));
                }
            }
        }

        // Update metadata
        if let Err(e) = self.update_file_metadata(&all_files_to_process).await {
            warn!("Failed to update file_metadata: {}", e);
        }

        stats.duration = start_time.elapsed();
        info!("🎉 Sync complete: {} nodes, {} edges in {:?}",
            stats.total_nodes, stats.total_edges, stats.duration);

        Ok(stats)
    }

    /// Process a batch of files
    async fn process_batch(
        &self,
        files: &[GitHubFileBasicMetadata],
        stats: &mut SyncStatistics,
    ) -> Result<(), String> {
        let mut batch_nodes = std::collections::HashMap::new();
        let mut batch_edges = std::collections::HashMap::new();
        let mut public_pages = std::collections::HashSet::new();

        info!("🔍 [DEBUG] Starting batch with {} files", files.len());

        // Process each file
        for (idx, file) in files.iter().enumerate() {
            if idx % 10 == 0 && idx > 0 {
                info!("  Progress: {}/{} files in batch (nodes so far: {}, edges: {})",
                    idx, files.len(), batch_nodes.len(), batch_edges.len());
            }

            match self.process_single_file(file, &mut batch_nodes, &mut batch_edges, &mut public_pages).await {
                Ok(()) => {
                    stats.kg_files_processed += 1;
                    debug!("✓ Processed {}: {} nodes total, {} edges total",
                        file.name, batch_nodes.len(), batch_edges.len());
                }
                Err(e) => {
                    warn!("Error processing {}: {}", file.name, e);
                    stats.errors.push(format!("{}: {}", file.name, e));
                }
            }

            // Rate limiting
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        info!("🔍 [DEBUG] After processing: {} nodes, {} edges, {} public_pages",
            batch_nodes.len(), batch_edges.len(), public_pages.len());

        // Don't filter nodes/edges - save everything to maintain graph connectivity
        // Edge cross-references between batches should be preserved
        // let nodes_before_filter = batch_nodes.len();
        // self.filter_linked_pages(&mut batch_nodes, &public_pages);
        // info!("🔍 [DEBUG] After filter_linked_pages: {} nodes (removed {})",
        //     batch_nodes.len(), nodes_before_filter - batch_nodes.len());

        // let edges_before_filter = batch_edges.len();
        // self.filter_orphan_edges(&mut batch_edges, &batch_nodes);
        // info!("🔍 [DEBUG] After filter_orphan_edges: {} edges (removed {})",
        //     batch_edges.len(), edges_before_filter - batch_edges.len());

        // Save batch to database
        if !batch_nodes.is_empty() {
            let node_vec: Vec<_> = batch_nodes.into_values().collect();
            let edge_vec: Vec<_> = batch_edges.into_values().collect();

            stats.total_nodes += node_vec.len();
            stats.total_edges += edge_vec.len();

            info!("💾 Saving batch: {} nodes, {} edges", node_vec.len(), edge_vec.len());

            let mut graph = crate::models::graph::GraphData::new();
            graph.nodes = node_vec;
            graph.edges = edge_vec;

            info!("🔍 [DEBUG] Calling save_graph() with {} nodes, {} edges",
                graph.nodes.len(), graph.edges.len());

            self.kg_repo.save_graph(&graph).await.map_err(|e| {
                error!("❌ save_graph() failed: {}", e);
                format!("Failed to save batch: {}", e)
            })?;

            info!("✅ [DEBUG] save_graph() completed successfully");
        } else {
            warn!("⚠️ [DEBUG] Batch is EMPTY after filtering - nothing to save!");
        }

        Ok(())
    }

    /// Process a single file
    async fn process_single_file(
        &self,
        file: &GitHubFileBasicMetadata,
        nodes: &mut std::collections::HashMap<u32, crate::models::node::Node>,
        edges: &mut std::collections::HashMap<String, crate::models::edge::Edge>,
        public_pages: &mut std::collections::HashSet<String>,
    ) -> Result<(), String> {
        debug!("🔍 Processing file: {}", file.name);

        // Fetch content
        let content = self.content_api
            .fetch_file_content(&file.download_url)
            .await
            .map_err(|e| format!("Failed to fetch content: {}", e))?;

        debug!("🔍 Fetched {} bytes for {}", content.len(), file.name);

        // Detect file type
        let file_type = self.detect_file_type(&content);
        debug!("🔍 Detected file type: {:?} for {}", file_type, file.name);

        let page_name = file.name.trim_end_matches(".md");

        match file_type {
            FileType::KnowledgeGraph => {
                // Process public:: true files as knowledge graph nodes
                debug!("🔍 Parsing knowledge graph from {}", file.name);
                let parsed = self.kg_parser.parse(&content, &file.name)
                    .map_err(|e| format!("Parse error: {}", e))?;

                info!("📊 Parsed {}: {} nodes, {} edges",
                    file.name, parsed.nodes.len(), parsed.edges.len());

                // Add to public pages
                public_pages.insert(page_name.to_string());
                debug!("✓ Added '{}' to public_pages (total: {})", page_name, public_pages.len());

                // Add nodes from KG parser
                let nodes_before = nodes.len();
                for node in parsed.nodes {
                    debug!("  → Node {}: {} (type: {:?})",
                        node.id, node.label,
                        node.metadata.get("type"));
                    nodes.insert(node.id, node);
                }
                let kg_nodes_added = nodes.len() - nodes_before;
                info!("✓ Added {} KG nodes from {} (total now: {})",
                    kg_nodes_added, file.name, nodes.len());

                // Add edges from KG parser
                let edges_before = edges.len();
                for edge in parsed.edges {
                    edges.insert(edge.id.clone(), edge);
                }
                if edges.len() > edges_before {
                    debug!("✓ Added {} edges from {}", edges.len() - edges_before, file.name);
                }

                // Also check for and parse ontology blocks in this file
                if content.contains("### OntologyBlock") {
                    debug!("🦉 Detected OntologyBlock in {}, extracting ontology data", file.name);
                    match self.onto_parser.parse(&content, &file.name) {
                        Ok(onto_data) => {
                            info!("🦉 Extracted from {}: {} classes, {} properties, {} axioms",
                                file.name,
                                onto_data.classes.len(),
                                onto_data.properties.len(),
                                onto_data.axioms.len());

                            // Save ontology data immediately
                            if let Err(e) = self.save_ontology_data(onto_data).await {
                                error!("Failed to save ontology data from {}: {}", file.name, e);
                            } else {
                                debug!("✓ Saved ontology data from {}", file.name);
                            }
                        }
                        Err(e) => {
                            debug!("Failed to parse ontology block in {}: {}", file.name, e);
                        }
                    }
                }

                Ok(())
            }
            FileType::Ontology => {
                // Process files with ontology blocks
                debug!("🦉 Processing ontology file {}", file.name);
                match self.onto_parser.parse(&content, &file.name) {
                    Ok(onto_data) => {
                        info!("🦉 Extracted from {}: {} classes, {} properties, {} axioms",
                            file.name,
                            onto_data.classes.len(),
                            onto_data.properties.len(),
                            onto_data.axioms.len());

                        // Save ontology data immediately
                        if let Err(e) = self.save_ontology_data(onto_data).await {
                            error!("Failed to save ontology data from {}: {}", file.name, e);
                        } else {
                            debug!("✓ Saved ontology data from {}", file.name);
                        }
                    }
                    Err(e) => {
                        debug!("Failed to parse ontology file {}: {}", file.name, e);
                    }
                }
                Ok(())
            }
            FileType::Skip => {
                // Skip regular notes without public:: true or ontology blocks
                debug!("⏭️  Skipped regular note: {} (no public:: true or ontology block)", file.name);
                Ok(())
            }
        }
    }

    /// Filter linked pages
    fn filter_linked_pages(
        &self,
        nodes: &mut std::collections::HashMap<u32, crate::models::node::Node>,
        public_pages: &std::collections::HashSet<String>,
    ) {
        let before = nodes.len();
        nodes.retain(|_, node| {
            match node.metadata.get("type").map(|s| s.as_str()) {
                Some("page") => true,
                Some("linked_page") => public_pages.contains(&node.metadata_id),
                _ => true,
            }
        });
        let filtered = before - nodes.len();
        if filtered > 0 {
            info!("🔍 Filtered {} linked_page nodes", filtered);
        }
    }

    /// Filter orphan edges
    fn filter_orphan_edges(
        &self,
        edges: &mut std::collections::HashMap<String, crate::models::edge::Edge>,
        nodes: &std::collections::HashMap<u32, crate::models::node::Node>,
    ) {
        let before = edges.len();
        edges.retain(|_, edge| {
            nodes.contains_key(&edge.source) && nodes.contains_key(&edge.target)
        });
        let filtered = before - edges.len();
        if filtered > 0 {
            info!("🔍 Filtered {} orphan edges", filtered);
        }
    }

    /// SHA1-based filtering
    async fn filter_changed_files(
        &self,
        files: &[GitHubFileBasicMetadata],
    ) -> Result<Vec<GitHubFileBasicMetadata>, String> {
        let existing = self.get_existing_file_metadata().await?;

        Ok(files
            .iter()
            .filter(|file| {
                match existing.get(&file.name) {
                    Some(existing_sha) if existing_sha == &file.sha => false,
                    _ => true,
                }
            })
            .cloned()
            .collect())
    }

    // ... (rest of helper methods unchanged)
    async fn fetch_all_markdown_files(&self) -> Result<Vec<GitHubFileBasicMetadata>, String> {
        self.content_api
            .list_markdown_files("")
            .await
            .map_err(|e| format!("GitHub API error: {}", e))
    }

    async fn get_existing_file_metadata(
        &self,
    ) -> Result<std::collections::HashMap<String, String>, String> {
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

                // Extract file extension
                let extension = file.name.rsplit('.').next().unwrap_or("");

                // Upsert file metadata
                tx.execute(
                    r#"
                    INSERT INTO file_metadata
                        (file_name, file_path, file_extension,
                         file_blob_sha, github_node_id, sha1, content_hash,
                         last_modified, last_content_change, updated_at, processing_status)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'complete')
                    ON CONFLICT(file_name) DO UPDATE SET
                        file_path = ?2,
                        file_extension = ?3,
                        file_blob_sha = ?4,
                        sha1 = ?6,
                        content_hash = ?7,
                        last_modified = ?8,
                        last_content_change = CASE
                            WHEN file_blob_sha != ?4 THEN ?9
                            ELSE last_content_change
                        END,
                        updated_at = ?10,
                        processing_status = 'complete',
                        change_count = CASE
                            WHEN file_blob_sha != ?4 THEN COALESCE(change_count, 0) + 1
                            ELSE change_count
                        END
                    "#,
                    params![
                        file.name,
                        file.download_url,
                        extension,
                        file.sha,
                        "", // github_node_id
                        file.sha, // sha1
                        file.sha, // content_hash
                        now.clone(),
                        now.clone(), // last_content_change
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

    fn detect_file_type(&self, content: &str) -> FileType {
        let content = content.trim_start_matches('\u{feff}');
        let lines: Vec<&str> = content.lines().take(20).collect();

        // Check for ontology blocks first (highest priority)
        if content.contains("### OntologyBlock") {
            return FileType::Ontology;
        }

        // Check for explicit public:: true (knowledge graph files)
        for line in lines.iter() {
            if line.trim() == "public:: true" {
                return FileType::KnowledgeGraph;
            }
        }

        // Default: skip regular notes without public:: true or ontology blocks
        FileType::Skip
    }

    /// Save ontology data to database
    async fn save_ontology_data(&self, onto_data: crate::services::parsers::ontology_parser::OntologyData) -> Result<(), String> {
        use crate::ports::ontology_repository::OntologyRepository;

        // Save all ontology data in one call
        self.onto_repo.save_ontology(&onto_data.classes, &onto_data.properties, &onto_data.axioms).await
            .map_err(|e| format!("Failed to save ontology data: {}", e))?;

        // Log class hierarchy
        for (subclass_iri, superclass_iri) in onto_data.class_hierarchy {
            debug!("Class hierarchy: {} -> {}", subclass_iri, superclass_iri);
        }

        Ok(())
    }
}
