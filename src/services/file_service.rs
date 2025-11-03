use super::github::{ContentAPI, GitHubClient, GitHubConfig};
use crate::config::AppFullSettings; 
use crate::models::graph::GraphData;
use crate::models::metadata::{Metadata, MetadataOps, MetadataStore};
use actix_web::web;
use chrono::Utc;
use log::{debug, error, info, warn};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::fs;
use std::fs::File;
use std::io::Error;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;

// Constants
const METADATA_PATH: &str = "/workspace/ext/data/metadata/metadata.json";
pub const MARKDOWN_DIR: &str = "/workspace/ext/data/markdown";
const GITHUB_API_DELAY: Duration = Duration::from_millis(500);

#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessedFile {
    pub file_name: String,
    pub content: String,
    pub is_public: bool,
    pub metadata: Metadata,
}

pub struct FileService {
    _settings: Arc<RwLock<AppFullSettings>>, 
    
    node_id_counter: AtomicU32,
}

impl FileService {
    pub fn new(_settings: Arc<RwLock<AppFullSettings>>) -> Self {
        
        
        let service = Self {
            _settings, 
            node_id_counter: AtomicU32::new(1),
        };

        
        if let Ok(metadata) = Self::load_or_create_metadata() {
            let max_id = metadata.get_max_node_id();
            if max_id > 0 {
                
                service.node_id_counter.store(max_id + 1, Ordering::SeqCst);
                info!(
                    "Initialized node ID counter to {} based on existing metadata",
                    max_id + 1
                );
            }
        }

        service
    }

    
    fn get_next_node_id(&self) -> u32 {
        self.node_id_counter.fetch_add(1, Ordering::SeqCst)
    }

    
    fn update_node_ids(&self, processed_files: &mut Vec<ProcessedFile>) {
        for processed_file in processed_files {
            if processed_file.metadata.node_id == "0" {
                processed_file.metadata.node_id = self.get_next_node_id().to_string();
            }
        }
    }

    
    pub async fn process_file_upload(&self, payload: web::Bytes) -> Result<GraphData, Error> {
        let content = String::from_utf8(payload.to_vec())
            .map_err(|e| Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let metadata = Self::load_or_create_metadata()
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e))?;
        let mut graph_data = GraphData::new();

        
        let temp_filename = format!("temp_{}.md", time::timestamp_seconds());
        let temp_path = format!("{}/{}", MARKDOWN_DIR, temp_filename);
        if let Err(e) = fs::write(&temp_path, &content) {
            return Err(Error::new(std::io::ErrorKind::Other, e.to_string()));
        }

        
        let valid_nodes: Vec<String> = metadata
            .keys()
            .map(|name| name.trim_end_matches(".md").to_string())
            .collect();

        let references = Self::extract_references(&content, &valid_nodes);
        let topic_counts = Self::convert_references_to_topic_counts(references);

        
        let file_size = content.len();
        let node_size = Self::calculate_node_size(file_size);
        let file_metadata = Metadata {
            file_name: temp_filename.clone(),
            file_size,
            node_size,
            node_id: "0".to_string(),
            hyperlink_count: Self::count_hyperlinks(&content),
            sha1: Self::calculate_sha1(&content),
            last_modified: time::now(),
            last_content_change: Some(time::now()), 
            last_commit: Some(time::now()),
            change_count: Some(1), 
            file_blob_sha: None,   
            perplexity_link: String::new(),
            last_perplexity_process: None,
            topic_counts,
        };

        
        let mut file_metadata = file_metadata;
        file_metadata.node_id = self.get_next_node_id().to_string();

        
        graph_data
            .metadata
            .insert(temp_filename.clone(), file_metadata);

        
        if let Err(e) = fs::remove_file(&temp_path) {
            error!("Failed to remove temporary file: {}", e);
        }

        Ok(graph_data)
    }

    
    pub async fn list_files(&self) -> Result<Vec<String>, Error> {
        let metadata = Self::load_or_create_metadata()
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e))?;
        Ok(metadata.keys().cloned().collect())
    }

    
    pub async fn load_file(&self, filename: &str) -> Result<GraphData, Error> {
        let file_path = format!("{}/{}", MARKDOWN_DIR, filename);
        if !Path::new(&file_path).exists() {
            return Err(Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", filename),
            ));
        }

        let content = fs::read_to_string(&file_path)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let metadata = Self::load_or_create_metadata()
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e))?;
        let mut graph_data = GraphData::new();

        
        let valid_nodes: Vec<String> = metadata
            .keys()
            .map(|name| name.trim_end_matches(".md").to_string())
            .collect();

        let references = Self::extract_references(&content, &valid_nodes);
        let topic_counts = Self::convert_references_to_topic_counts(references);

        
        let file_size = content.len();
        let node_size = Self::calculate_node_size(file_size);
        let file_metadata = Metadata {
            file_name: filename.to_string(),
            file_size,
            node_size,
            node_id: "0".to_string(),
            hyperlink_count: Self::count_hyperlinks(&content),
            sha1: Self::calculate_sha1(&content),
            last_modified: time::now(),
            last_content_change: Some(time::now()),
            last_commit: Some(time::now()),
            change_count: None,
            file_blob_sha: None,
            perplexity_link: String::new(),
            last_perplexity_process: None,
            topic_counts,
        };

        
        let mut file_metadata = file_metadata;
        file_metadata.node_id = self.get_next_node_id().to_string();

        
        graph_data
            .metadata
            .insert(filename.to_string(), file_metadata);

        Ok(graph_data)
    }

    
    pub fn load_or_create_metadata() -> Result<MetadataStore, String> {
        
        std::fs::create_dir_all("/app/data/metadata")
            .map_err(|e| format!("Failed to create metadata directory: {}", e))?;

        let metadata_path = "/app/data/metadata/metadata.json";

        match File::open(metadata_path) {
            Ok(file) => {
                info!("Loading existing metadata from {}", metadata_path);
                serde_json::from_reader(file)
                    .map_err(|e| format!("Failed to parse metadata: {}", e))
            }
            _ => {
                info!("Creating new metadata file at {}", metadata_path);
                let empty_store = MetadataStore::default();
                let file = File::create(metadata_path)
                    .map_err(|e| format!("Failed to create metadata file: {}", e))?;

                serde_json::to_writer_pretty(file, &empty_store)
                    .map_err(|e| format!("Failed to write metadata: {}", e))?;

                
                let metadata = std::fs::metadata(metadata_path)
                    .map_err(|e| format!("Failed to verify metadata file: {}", e))?;

                if !metadata.is_file() {
                    return Err("Metadata file was not created properly".to_string());
                }

                Ok(empty_store)
            }
        }
    }

    
    pub fn load_graph_data() -> Result<Option<GraphData>, String> {
        let graph_path = "/app/data/metadata/graph.json";

        match File::open(graph_path) {
            Ok(file) => {
                info!("Loading existing graph data from {}", graph_path);
                match serde_json::from_reader(file) {
                    Ok(graph) => {
                        info!("Successfully loaded graph data with positions");
                        Ok(Some(graph))
                    }
                    Err(e) => {
                        error!("Failed to parse graph.json: {}", e);
                        Ok(None)
                    }
                }
            }
            Err(e) => {
                info!(
                    "No existing graph.json found: {}. Will generate positions.",
                    e
                );
                Ok(None)
            }
        }
    }

    
    fn calculate_node_size(file_size: usize) -> f64 {
        const BASE_SIZE: f64 = 1000.0; 
        const MIN_SIZE: f64 = 5.0; 
        const MAX_SIZE: f64 = 50.0; 

        let size = (file_size as f64 / BASE_SIZE).min(5.0);
        MIN_SIZE + (size * (MAX_SIZE - MIN_SIZE) / 5.0)
    }

    
    fn extract_references(content: &str, valid_nodes: &[String]) -> Vec<String> {
        let mut references = Vec::new();
        let content_lower = content.to_lowercase();

        for node_name in valid_nodes {
            let node_name_lower = node_name.to_lowercase();

            
            let pattern = format!(r"\b{}\b", regex::escape(&node_name_lower));
            if let Ok(re) = Regex::new(&pattern) {
                
                let count = re.find_iter(&content_lower).count();

                
                if count > 0 {
                    debug!("Found {} references to {} in content", count, node_name);
                    
                    for _ in 0..count {
                        references.push(node_name.clone());
                    }
                }
            }
        }

        references
    }

    fn convert_references_to_topic_counts(references: Vec<String>) -> HashMap<String, usize> {
        let mut topic_counts = HashMap::new();
        for reference in references {
            *topic_counts.entry(reference).or_insert(0) += 1;
        }
        topic_counts
    }

    
    pub async fn initialize_local_storage(
        settings: Arc<RwLock<AppFullSettings>>, 
    ) -> Result<(), Box<dyn StdError + Send + Sync>> {
        
        let github_config =
            GitHubConfig::from_env().map_err(|e| Box::new(e) as Box<dyn StdError + Send + Sync>)?;

        let github = GitHubClient::new(github_config, Arc::clone(&settings)).await?;
        let content_api = ContentAPI::new(Arc::new(github));

        
        if Self::has_valid_local_setup() {
            info!("Valid local setup found, skipping initialization");
            return Ok(());
        }

        info!("Initializing local storage with files from GitHub");

        
        Self::ensure_directories()?;

        
        let basic_github_files = content_api.list_markdown_files("").await?;
        info!(
            "Found {} markdown files in GitHub",
            basic_github_files.len()
        );

        let mut metadata_store = MetadataStore::new();

        
        const BATCH_SIZE: usize = 5;
        for chunk in basic_github_files.chunks(BATCH_SIZE) {
            let mut futures = Vec::new();

            for file_basic_meta in chunk {
                let file_basic_meta = file_basic_meta.clone();
                let content_api = content_api.clone();

                futures.push(async move {
                    
                    let file_extended_meta = match content_api
                        .get_file_metadata_extended(&file_basic_meta.path)
                        .await
                    {
                        Ok(meta) => meta,
                        Err(e) => {
                            error!(
                                "Failed to get extended metadata for {}: {}",
                                file_basic_meta.name, e
                            );
                            return Err(e);
                        }
                    };

                    
                    match content_api
                        .fetch_file_content(&file_extended_meta.download_url)
                        .await
                    {
                        Ok(content) => {
                            
                            let is_public = if let Some(first_line) = content.lines().next() {
                                first_line.trim() == "public:: true"
                            } else {
                                false
                            };

                            if !is_public {
                                debug!(
                                    "Skipping file without 'public:: true': {}",
                                    file_basic_meta.name
                                );
                                return Ok(None);
                            }

                            let file_path = format!("{}/{}", MARKDOWN_DIR, file_extended_meta.name);
                            if let Err(e) = fs::write(&file_path, &content) {
                                error!("Failed to write file {}: {}", file_path, e);
                                return Err(e.into());
                            }

                            info!(
                                "fetch_and_process_files: Successfully wrote {} to {}",
                                file_extended_meta.name, file_path
                            );

                            Ok(Some((file_extended_meta, content)))
                        }
                        Err(e) => {
                            error!(
                                "Failed to fetch content for {}: {}",
                                file_extended_meta.name, e
                            );
                            Err(e)
                        }
                    }
                });
            }

            
            let results = futures::future::join_all(futures).await;

            for result in results {
                match result {
                    Ok(Some((file_extended_meta, content))) => {
                        let _node_name =
                            file_extended_meta.name.trim_end_matches(".md").to_string();
                        let file_size = content.len();
                        let node_size = Self::calculate_node_size(file_size);

                        
                        let metadata = Metadata {
                            file_name: file_extended_meta.name.clone(),
                            file_size,
                            node_size,
                            node_id: "0".to_string(), 
                            hyperlink_count: Self::count_hyperlinks(&content),
                            sha1: Self::calculate_sha1(&content),
                            last_modified: file_extended_meta.last_content_modified, 
                            last_content_change: Some(file_extended_meta.last_content_modified),
                            last_commit: Some(file_extended_meta.last_content_modified), 
                            change_count: None, 
                            file_blob_sha: Some(file_extended_meta.sha.clone()), 
                            perplexity_link: String::new(),
                            last_perplexity_process: None,
                            topic_counts: HashMap::new(), 
                        };

                        metadata_store.insert(file_extended_meta.name, metadata);
                    }
                    Ok(None) => continue, 
                    Err(e) => {
                        error!("Failed to process file in batch: {}", e);
                    }
                }
            }

            sleep(GITHUB_API_DELAY).await;
        }

        
        Self::update_topic_counts(&mut metadata_store)?;

        
        info!("Saving metadata for {} public files", metadata_store.len());
        Self::save_metadata(&metadata_store)?;

        info!(
            "Initialization complete. Processed {} public files",
            metadata_store.len()
        );
        Ok(())
    }

    
    fn update_topic_counts(metadata_store: &mut MetadataStore) -> Result<(), Error> {
        let valid_nodes: Vec<String> = metadata_store
            .keys()
            .map(|name| name.trim_end_matches(".md").to_string())
            .collect();

        for file_name in metadata_store.keys().cloned().collect::<Vec<_>>() {
            let file_path = format!("{}/{}", MARKDOWN_DIR, file_name);
            if let Ok(content) = fs::read_to_string(&file_path) {
                let references = Self::extract_references(&content, &valid_nodes);
                let topic_counts = Self::convert_references_to_topic_counts(references);

                if let Some(metadata) = metadata_store.get_mut(&file_name) {
                    metadata.topic_counts = topic_counts;
                }
            }
        }

        Ok(())
    }

    
    fn has_valid_local_setup() -> bool {
        if let Ok(metadata_content) = fs::read_to_string(METADATA_PATH) {
            if metadata_content.trim().is_empty() {
                return false;
            }

            if let Ok(metadata) = serde_json::from_str::<MetadataStore>(&metadata_content) {
                return metadata.validate_files(MARKDOWN_DIR);
            }
        }
        false
    }

    
    fn ensure_directories() -> Result<(), Error> {
        
        let markdown_dir = Path::new(MARKDOWN_DIR);
        if !markdown_dir.exists() {
            info!("Creating markdown directory at {:?}", markdown_dir);
            fs::create_dir_all(markdown_dir).map_err(|e| {
                Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create markdown directory: {}", e),
                )
            })?;
            
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(markdown_dir, fs::Permissions::from_mode(0o777)).map_err(
                    |e| {
                        Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to set markdown directory permissions: {}", e),
                        )
                    },
                )?;
            }
        }

        
        let metadata_dir = Path::new(METADATA_PATH).parent().unwrap();
        if !metadata_dir.exists() {
            info!("Creating metadata directory at {:?}", metadata_dir);
            fs::create_dir_all(metadata_dir).map_err(|e| {
                Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create metadata directory: {}", e),
                )
            })?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(metadata_dir, fs::Permissions::from_mode(0o777)).map_err(
                    |e| {
                        Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to set metadata directory permissions: {}", e),
                        )
                    },
                )?;
            }
        }

        
        let test_file = format!("{}/test_permissions", MARKDOWN_DIR);
        match fs::write(&test_file, "test") {
            Ok(_) => {
                info!("Successfully wrote test file to {}", test_file);
                fs::remove_file(&test_file).map_err(|e| {
                    Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to remove test file: {}", e),
                    )
                })?;
                info!("Successfully removed test file");
                info!("Directory permissions verified");
                Ok(())
            }
            Err(e) => {
                error!("Failed to verify directory permissions: {}", e);
                if let Ok(current_dir) = std::env::current_dir() {
                    error!("Current directory: {:?}", current_dir);
                }
                if let Ok(dir_contents) = fs::read_dir(MARKDOWN_DIR) {
                    error!("Directory contents: {:?}", dir_contents);
                }
                Err(Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!("Failed to verify directory permissions: {}", e),
                ))
            }
        }
    }

    
    pub fn save_metadata(metadata: &MetadataStore) -> Result<(), Error> {
        let json = crate::utils::json::to_json_pretty(metadata)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        fs::write(METADATA_PATH, json)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(())
    }

    
    fn calculate_sha1(content: &str) -> String {
        use sha1::{Digest, Sha1};
use crate::utils::json::{from_json, to_json};
        let mut hasher = Sha1::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    
    fn count_hyperlinks(content: &str) -> usize {
        let re = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").expect("Invalid regex pattern");
        re.find_iter(content).count()
    }

    
    async fn should_process_file(
        &self,
        file_name: &str,
        github_blob_sha: &str,
        content_api: &ContentAPI,
        download_url: &str,
        metadata_store: &MetadataStore,
    ) -> Result<bool, Box<dyn StdError + Send + Sync>> {
        
        if let Some(existing_metadata) = metadata_store.get(file_name) {
            
            if let Some(stored_sha) = &existing_metadata.file_blob_sha {
                if stored_sha == github_blob_sha {
                    info!(
                        "should_process_file: File {} has unchanged SHA, skipping",
                        file_name
                    );
                    return Ok(false);
                } else {
                    info!(
                        "should_process_file: File {} SHA changed (old: {}, new: {})",
                        file_name, stored_sha, github_blob_sha
                    );
                }
            } else {
                info!(
                    "should_process_file: File {} has no stored SHA, will check content",
                    file_name
                );
            }
        } else {
            info!(
                "should_process_file: File {} is new, will check content",
                file_name
            );
        }

        
        info!(
            "should_process_file: Downloading content for {} to check public tag",
            file_name
        );
        match content_api.fetch_file_content(download_url).await {
            Ok(content) => {
                
                if let Some(first_line) = content.lines().next() {
                    let is_public = first_line.trim().to_lowercase() == "public:: true";
                    if !is_public {
                        info!("should_process_file: File {} does not have 'public:: true' on first line (found: '{}'), skipping", file_name, first_line.trim());
                    } else {
                        info!(
                            "should_process_file: File {} has 'public:: true' tag, will process",
                            file_name
                        );
                    }
                    Ok(is_public)
                } else {
                    
                    info!("should_process_file: File {} is empty, skipping", file_name);
                    Ok(false)
                }
            }
            Err(e) => {
                error!("Failed to fetch content for {}: {}", file_name, e);
                Err(Box::new(e))
            }
        }
    }

    
    pub async fn fetch_and_process_files(
        &self,
        content_api: Arc<ContentAPI>,
        _settings: Arc<RwLock<AppFullSettings>>, 
        metadata_store: &mut MetadataStore,
    ) -> Result<Vec<ProcessedFile>, Box<dyn StdError + Send + Sync>> {
        info!("fetch_and_process_files: Starting GitHub file fetch process");
        debug!("Attempting to fetch and process files from GitHub repository.");
        let mut processed_files = Vec::new();

        
        info!("fetch_and_process_files: Calling list_markdown_files...");
        let basic_github_files = match content_api.list_markdown_files("").await {
            Ok(files) => {
                info!(
                    "fetch_and_process_files: Successfully retrieved {} file entries from GitHub",
                    files.len()
                );
                debug!(
                    "GitHub API returned {} potential markdown files.",
                    files.len()
                );
                if files.is_empty() {
                    warn!("fetch_and_process_files: No markdown files found in GitHub repository");
                    warn!("fetch_and_process_files: Check GITHUB_OWNER, GITHUB_REPO, and GITHUB_BASE_PATH in .env");
                }
                files
            }
            Err(e) => {
                error!(
                    "fetch_and_process_files: Failed to list markdown files from GitHub: {}",
                    e
                );
                return Err(Box::new(e));
            }
        };

        info!(
            "fetch_and_process_files: Processing {} markdown files from GitHub",
            basic_github_files.len()
        );

        
        const BATCH_SIZE: usize = 5;
        let total_batches = (basic_github_files.len() + BATCH_SIZE - 1) / BATCH_SIZE;
        info!(
            "fetch_and_process_files: Processing files in {} batches of up to {} files each",
            total_batches, BATCH_SIZE
        );

        for (batch_idx, chunk) in basic_github_files.chunks(BATCH_SIZE).enumerate() {
            info!(
                "fetch_and_process_files: Processing batch {}/{} with {} files",
                batch_idx + 1,
                total_batches,
                chunk.len()
            );
            let mut futures = Vec::new();

            for file_basic_meta in chunk {
                let file_basic_meta = file_basic_meta.clone();
                let content_api = content_api.clone();
                let metadata_store_clone = metadata_store.clone();

                info!(
                    "fetch_and_process_files: Checking file: {}",
                    file_basic_meta.name
                );

                futures.push(async move {
                    
                    let file_extended_meta = match content_api.get_file_metadata_extended(&file_basic_meta.path).await {
                        Ok(meta) => meta,
                        Err(e) => {
                            error!("Failed to get extended metadata for {}: {}", file_basic_meta.name, e);
                            return Err(e);
                        }
                    };

                    
                    let needs_download = if let Some(existing_metadata) = metadata_store_clone.get(&file_extended_meta.name) {
                        if let Some(stored_sha) = &existing_metadata.file_blob_sha {
                            if stored_sha == &file_extended_meta.sha {
                                info!("fetch_and_process_files: File {} has unchanged SHA, skipping download", file_extended_meta.name);
                                false
                            } else {
                                info!("fetch_and_process_files: File {} SHA changed (old: {}, new: {})",
                                     file_extended_meta.name, stored_sha, file_extended_meta.sha);
                                true
                            }
                        } else {
                            info!("fetch_and_process_files: File {} has no stored SHA, will download", file_extended_meta.name);
                            true
                        }
                    } else {
                        info!("fetch_and_process_files: File {} is new, will download", file_extended_meta.name);
                        true
                    };

                    if !needs_download {
                        return Ok(None);
                    }

                    
                    match content_api.fetch_file_content(&file_extended_meta.download_url).await {
                        Ok(content) => {
                            
                            let first_line = content.lines().next().unwrap_or("");
                            let is_public = first_line.trim().to_lowercase() == "public:: true";

                            if !is_public {
                                info!("fetch_and_process_files: File {} does not have 'public:: true' on first line (found: '{}')",
                                     file_extended_meta.name, first_line.trim());
                                return Ok(None);
                            }

                            info!("fetch_and_process_files: File {} is marked as public, writing to disk", file_extended_meta.name);

                            let file_path = format!("{}/{}", MARKDOWN_DIR, file_extended_meta.name);
                            if let Err(e) = fs::write(&file_path, &content) {
                                error!("Failed to write file {}: {}", file_path, e);
                                return Err(e.into());
                            }

                            info!("fetch_and_process_files: Successfully wrote {} to {}", file_extended_meta.name, file_path);

                            let file_size = content.len();
                            let node_size = Self::calculate_node_size(file_size);

                            let metadata = Metadata {
                                file_name: file_extended_meta.name.clone(),
                                file_size,
                                node_size,
                                node_id: "0".to_string(), 
                                hyperlink_count: Self::count_hyperlinks(&content),
                                sha1: Self::calculate_sha1(&content),
                                last_modified: file_extended_meta.last_content_modified, 
                                last_content_change: Some(file_extended_meta.last_content_modified),
                                last_commit: Some(file_extended_meta.last_content_modified), 
                                change_count: None, 
                                file_blob_sha: Some(file_extended_meta.sha.clone()), 
                                perplexity_link: String::new(),
                                last_perplexity_process: None,
                                topic_counts: HashMap::new(), 
                            };

                            Ok(Some(ProcessedFile {
                                file_name: file_extended_meta.name.clone(),
                                content,
                                is_public: true,
                                metadata,
                            }))
                        }
                        Err(e) => {
                            error!("Failed to fetch content for {}: {}", file_basic_meta.name, e);
                            Err(e)
                        }
                    }
                });
            }

            
            let results = futures::future::join_all(futures).await;

            for result in results {
                match result {
                    Ok(Some(processed_file)) => {
                        processed_files.push(processed_file);
                    }
                    Ok(None) => continue, 
                    Err(e) => {
                        error!("Failed to process file in batch: {}", e);
                    }
                }
            }

            sleep(GITHUB_API_DELAY).await;
        }

        
        self.update_node_ids(&mut processed_files);

        
        for processed_file in &processed_files {
            metadata_store.insert(
                processed_file.file_name.clone(),
                processed_file.metadata.clone(),
            );
        }

        
        Self::update_topic_counts(metadata_store)?;

        Ok(processed_files)
    }
}
