use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use log::{debug, info, warn, error};
use regex::Regex;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::time::sleep;

#[derive(Error, Debug)]
pub enum DownloaderError {
    #[error("GitHub API error: {0}")]
    GitHubApi(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Rate limit exceeded, retry after: {0:?}")]
    RateLimit(Duration),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyDownloaderConfig {
    pub github_token: String,
    pub repo_owner: String,
    pub repo_name: String,
    pub base_path: String,
    pub max_retries: u32,
    pub initial_retry_delay_ms: u64,
    pub max_retry_delay_ms: u64,
    pub request_timeout_secs: u64,
    pub respect_rate_limits: bool,
}

impl Default for OntologyDownloaderConfig {
    fn default() -> Self {
        Self {
            github_token: String::new(),
            repo_owner: String::from("jjohare"),
            repo_name: String::from("logseq"),
            base_path: String::from("mainKnowledgeGraph/pages"),
            max_retries: 3,
            initial_retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            request_timeout_secs: 30,
            respect_rate_limits: true,
        }
    }
}

impl OntologyDownloaderConfig {
    pub fn from_env() -> Result<Self> {
        let github_token = std::env::var("GITHUB_TOKEN")
            .or_else(|_| std::env::var("GH_TOKEN"))
            .context("GITHUB_TOKEN or GH_TOKEN environment variable not set")?;

        if github_token.is_empty() {
            return Err(anyhow!("GitHub token cannot be empty"));
        }

        Ok(Self {
            github_token,
            ..Default::default()
        })
    }

    pub fn with_token(token: String) -> Self {
        Self {
            github_token: token,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubFile {
    pub name: String,
    pub path: String,
    pub sha: String,
    pub size: u64,
    #[serde(rename = "type")]
    pub file_type: String,
    pub download_url: Option<String>,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyBlock {
    pub id: String,
    pub source_file: String,
    pub title: String,
    pub properties: HashMap<String, Vec<String>>,
    pub owl_content: Vec<String>,
    pub classes: Vec<String>,
    pub properties_list: Vec<String>,
    pub relationships: Vec<OntologyRelationship>,
    pub downloaded_at: DateTime<Utc>,
    pub content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyRelationship {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub relationship_type: RelationshipType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelationshipType {
    SubClassOf,
    ObjectProperty,
    DataProperty,
    DisjointWith,
    EquivalentTo,
    InverseOf,
    Domain,
    Range,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub total_files: usize,
    pub processed_files: usize,
    pub ontology_blocks_found: usize,
    pub errors: Vec<String>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub current_file: Option<String>,
}

impl DownloadProgress {
    pub fn new(total_files: usize) -> Self {
        Self {
            total_files,
            processed_files: 0,
            ontology_blocks_found: 0,
            errors: Vec::new(),
            started_at: Utc::now(),
            completed_at: None,
            current_file: None,
        }
    }

    pub fn percentage(&self) -> f64 {
        if self.total_files == 0 {
            return 0.0;
        }
        (self.processed_files as f64 / self.total_files as f64) * 100.0
    }
}

pub struct OntologyDownloader {
    config: OntologyDownloaderConfig,
    client: Client,
    progress: Arc<RwLock<DownloadProgress>>,
    framework_files: HashSet<String>,
}

impl OntologyDownloader {
    pub fn new(config: OntologyDownloaderConfig) -> Result<Self> {
        if config.github_token.is_empty() {
            return Err(DownloaderError::Config("GitHub token is required".to_string()).into());
        }

        let client = Client::builder()
            .user_agent("ontology-downloader/1.0")
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()?;

        let mut framework_files = HashSet::new();
        framework_files.insert("ETSI.md".to_string());
        framework_files.insert("OntologyDefinition.md".to_string());
        framework_files.insert("PropertySchema.md".to_string());

        Ok(Self {
            config,
            client,
            progress: Arc::new(RwLock::new(DownloadProgress::new(0))),
            framework_files,
        })
    }

    pub async fn get_progress(&self) -> DownloadProgress {
        self.progress.read().await.clone()
    }

    pub async fn download_all(&self) -> Result<Vec<OntologyBlock>> {
        info!("Starting ontology download from GitHub: {}/{}/{}",
              self.config.repo_owner, self.config.repo_name, self.config.base_path);

        let files = self.list_all_files().await?;

        {
            let mut progress = self.progress.write().await;
            progress.total_files = files.len();
            progress.started_at = Utc::now();
        }

        info!("Found {} files to process", files.len());

        let mut all_blocks = Vec::new();

        for file in &files {
            {
                let mut progress = self.progress.write().await;
                progress.current_file = Some(file.name.clone());
            }

            match self.download_and_parse_file(file).await {
                Ok(blocks) => {
                    let block_count = blocks.len();
                    if block_count > 0 {
                        info!("Found {} ontology blocks in {}", block_count, file.name);
                        all_blocks.extend(blocks);

                        let mut progress = self.progress.write().await;
                        progress.ontology_blocks_found += block_count;
                    }
                }
                Err(e) => {
                    warn!("Error processing file {}: {}", file.name, e);
                    let mut progress = self.progress.write().await;
                    progress.errors.push(format!("{}: {}", file.name, e));
                }
            }

            {
                let mut progress = self.progress.write().await;
                progress.processed_files += 1;
            }
        }

        {
            let mut progress = self.progress.write().await;
            progress.completed_at = Some(Utc::now());
        }

        info!("Download complete: {} ontology blocks from {} files",
              all_blocks.len(), files.len());

        Ok(all_blocks)
    }

    async fn list_all_files(&self) -> Result<Vec<GitHubFile>> {
        info!("Listing files in {}", self.config.base_path);

        let mut all_files = Vec::new();
        let mut directories = vec![self.config.base_path.clone()];

        while let Some(dir) = directories.pop() {
            let files = self.list_directory(&dir).await?;

            for file in files {
                match file.file_type.as_str() {
                    "file" => {
                        if file.name.ends_with(".md") {
                            all_files.push(file);
                        }
                    }
                    "dir" => {
                        directories.push(file.path.clone());
                    }
                    _ => {}
                }
            }
        }

        Ok(all_files)
    }

    async fn list_directory(&self, path: &str) -> Result<Vec<GitHubFile>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.config.repo_owner,
            self.config.repo_name,
            path
        );

        let files = self.github_request::<Vec<GitHubFile>>(&url).await?;
        Ok(files)
    }

    async fn download_and_parse_file(&self, file: &GitHubFile) -> Result<Vec<OntologyBlock>> {
        debug!("Processing file: {}", file.name);

        let content = if let Some(ref download_url) = file.download_url {
            self.download_file_content(download_url).await?
        } else {
            self.get_file_content(&file.url).await?
        };

        if !self.should_process_file(&content) {
            debug!("Skipping file {} - no public ontology blocks", file.name);
            return Ok(Vec::new());
        }

        self.parse_ontology_file(&file.name, &file.path, &content)
    }

    fn should_process_file(&self, content: &str) -> bool {
        let has_ontology_marker = content.contains("- ### OntologyBlock")
            || content.contains("OntologyBlock");

        if !has_ontology_marker {
            return false;
        }

        let has_public_gate = content.contains("public:: true");

        has_public_gate
    }

    fn parse_ontology_file(&self, filename: &str, filepath: &str, content: &str) -> Result<Vec<OntologyBlock>> {
        let mut blocks = Vec::new();

        let title = self.extract_title(filename, content);
        let properties = self.extract_properties(content);
        let owl_contents = self.extract_owl_blocks(content)?;

        for (idx, owl_content) in owl_contents.iter().enumerate() {
            let classes = self.extract_classes(owl_content);
            let properties_list = self.extract_owl_properties(owl_content);
            let relationships = self.extract_relationships(owl_content);

            let block_id = format!("{}:block:{}", filepath, idx);
            let content_hash = self.calculate_hash(&format!("{}{}", title, owl_content));

            let block = OntologyBlock {
                id: block_id,
                source_file: filepath.to_string(),
                title: title.clone(),
                properties: properties.clone(),
                owl_content: vec![owl_content.clone()],
                classes,
                properties_list,
                relationships,
                downloaded_at: Utc::now(),
                content_hash,
            };

            blocks.push(block);
        }

        Ok(blocks)
    }

    fn extract_title(&self, filename: &str, content: &str) -> String {
        let heading_re = Regex::new(r"^#\s+(.+)$").unwrap();
        for line in content.lines() {
            if let Some(cap) = heading_re.captures(line) {
                return cap[1].trim().to_string();
            }
        }

        filename.trim_end_matches(".md").to_string()
    }

    fn extract_properties(&self, content: &str) -> HashMap<String, Vec<String>> {
        let mut properties = HashMap::new();
        let property_re = Regex::new(r"^([a-zA-Z][a-zA-Z0-9-_]*)::\s*(.+)$").unwrap();

        for line in content.lines() {
            if let Some(cap) = property_re.captures(line.trim()) {
                let key = cap[1].to_string();
                let value = cap[2].to_string();

                let values: Vec<String> = value
                    .split(',')
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .collect();

                properties.entry(key).or_insert_with(Vec::new).extend(values);
            }
        }

        properties
    }

    fn extract_owl_blocks(&self, content: &str) -> Result<Vec<String>> {
        let mut blocks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            let fence_match = if line.starts_with("```") {
                Some(line)
            } else if line.starts_with("- ```") {
                Some(&line[2..])
            } else {
                None
            };

            if let Some(fence_line) = fence_match {
                let language = fence_line.trim_start_matches("```").trim();

                if language == "clojure" || language.is_empty() {
                    i += 1;
                    if i >= lines.len() {
                        break;
                    }

                    let should_extract = if language == "clojure" {
                        true
                    } else if lines[i].trim().starts_with("owl:functional-syntax::") {
                        i += 1;
                        true
                    } else {
                        false
                    };

                    if should_extract {
                        let mut block_lines = Vec::new();
                        while i < lines.len() {
                            let current_line = lines[i];
                            if current_line.trim().starts_with("```") {
                                break;
                            }
                            let trimmed = current_line.trim_start();
                            if !trimmed.is_empty()
                                && !trimmed.starts_with(";;")
                                && !trimmed.starts_with("#")
                                && trimmed != "|" {
                                block_lines.push(trimmed);
                            }
                            i += 1;
                        }

                        let block_text = block_lines.join("\n");
                        let is_owl = block_text.contains("Declaration(")
                            || block_text.contains("SubClassOf(")
                            || block_text.contains("EquivalentClasses(")
                            || block_text.contains("DisjointClasses(")
                            || block_text.contains("ObjectProperty(")
                            || block_text.contains("DataProperty(");

                        if is_owl && !block_lines.is_empty() {
                            blocks.push(block_text);
                        }
                    }
                }
                i += 1;
                continue;
            }

            if line.starts_with("owl:functional-syntax::") {
                i += 1;
                if i >= lines.len() {
                    break;
                }

                if !lines[i].trim().starts_with('|') {
                    i += 1;
                    continue;
                }

                i += 1;

                let mut block_lines = Vec::new();
                let base_indent = if i < lines.len() {
                    lines[i].len() - lines[i].trim_start().len()
                } else {
                    0
                };

                while i < lines.len() {
                    let current_line = lines[i];
                    let current_indent = current_line.len() - current_line.trim_start().len();

                    if !current_line.trim().is_empty() && current_indent < base_indent {
                        break;
                    }

                    if current_line.trim_start().starts_with('#')
                        || current_line.trim().starts_with("```")
                        || (current_line.contains("::") && !current_line.trim().starts_with("//"))
                    {
                        break;
                    }

                    if current_indent >= base_indent && !current_line.trim().is_empty() {
                        let trimmed = if current_indent >= base_indent {
                            &current_line[base_indent..]
                        } else {
                            current_line.trim_start()
                        };
                        block_lines.push(trimmed);
                    }

                    i += 1;
                }

                if !block_lines.is_empty() {
                    blocks.push(block_lines.join("\n"));
                }
            } else {
                i += 1;
            }
        }

        Ok(blocks)
    }

    fn extract_classes(&self, owl_content: &str) -> Vec<String> {
        let mut classes = Vec::new();
        let class_re = Regex::new(r"Declaration\(Class\(([^)]+)\)\)").unwrap();

        for cap in class_re.captures_iter(owl_content) {
            classes.push(cap[1].to_string());
        }

        let subclass_re = Regex::new(r"SubClassOf\(([^\s]+)\s+[^)]+\)").unwrap();
        for cap in subclass_re.captures_iter(owl_content) {
            let class = cap[1].to_string();
            if !classes.contains(&class) {
                classes.push(class);
            }
        }

        classes
    }

    fn extract_owl_properties(&self, owl_content: &str) -> Vec<String> {
        let mut properties = Vec::new();

        let obj_prop_re = Regex::new(r"Declaration\(ObjectProperty\(([^)]+)\)\)").unwrap();
        for cap in obj_prop_re.captures_iter(owl_content) {
            properties.push(cap[1].to_string());
        }

        let data_prop_re = Regex::new(r"Declaration\(DataProperty\(([^)]+)\)\)").unwrap();
        for cap in data_prop_re.captures_iter(owl_content) {
            properties.push(cap[1].to_string());
        }

        properties
    }

    fn extract_relationships(&self, owl_content: &str) -> Vec<OntologyRelationship> {
        let mut relationships = Vec::new();

        let subclass_re = Regex::new(r"SubClassOf\(([^\s]+)\s+([^)]+)\)").unwrap();
        for cap in subclass_re.captures_iter(owl_content) {
            relationships.push(OntologyRelationship {
                subject: cap[1].to_string(),
                predicate: "rdfs:subClassOf".to_string(),
                object: cap[2].to_string(),
                relationship_type: RelationshipType::SubClassOf,
            });
        }

        let domain_re = Regex::new(r"ObjectPropertyDomain\(([^\s]+)\s+([^)]+)\)").unwrap();
        for cap in domain_re.captures_iter(owl_content) {
            relationships.push(OntologyRelationship {
                subject: cap[1].to_string(),
                predicate: "rdfs:domain".to_string(),
                object: cap[2].to_string(),
                relationship_type: RelationshipType::Domain,
            });
        }

        let range_re = Regex::new(r"ObjectPropertyRange\(([^\s]+)\s+([^)]+)\)").unwrap();
        for cap in range_re.captures_iter(owl_content) {
            relationships.push(OntologyRelationship {
                subject: cap[1].to_string(),
                predicate: "rdfs:range".to_string(),
                object: cap[2].to_string(),
                relationship_type: RelationshipType::Range,
            });
        }

        let disjoint_re = Regex::new(r"DisjointClasses\(([^\s]+)\s+([^)]+)\)").unwrap();
        for cap in disjoint_re.captures_iter(owl_content) {
            relationships.push(OntologyRelationship {
                subject: cap[1].to_string(),
                predicate: "owl:disjointWith".to_string(),
                object: cap[2].to_string(),
                relationship_type: RelationshipType::DisjointWith,
            });
        }

        relationships
    }

    fn calculate_hash(&self, content: &str) -> String {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(content.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    async fn download_file_content(&self, url: &str) -> Result<String> {
        let content = self.retry_with_backoff(|| async {
            self.client
                .get(url)
                .header("Authorization", format!("token {}", self.config.github_token))
                .send()
                .await?
                .error_for_status()?
                .text()
                .await
                .map_err(|e| e.into())
        })
        .await?;

        Ok(content)
    }

    async fn get_file_content(&self, api_url: &str) -> Result<String> {
        #[derive(Deserialize)]
        struct ContentResponse {
            content: String,
            encoding: String,
        }

        let response: ContentResponse = self.github_request(api_url).await?;

        if response.encoding == "base64" {
            let decoded = base64::decode(&response.content.replace('\n', ""))
                .context("Failed to decode base64 content")?;
            String::from_utf8(decoded).context("Invalid UTF-8 in file content")
        } else {
            Ok(response.content)
        }
    }

    async fn github_request<T: for<'de> Deserialize<'de>>(&self, url: &str) -> Result<T> {
        self.retry_with_backoff(|| async {
            let response = self.client
                .get(url)
                .header("Authorization", format!("token {}", self.config.github_token))
                .header("Accept", "application/vnd.github.v3+json")
                .send()
                .await?;

            if response.status() == StatusCode::FORBIDDEN {
                if let Some(retry_after) = response.headers().get("Retry-After") {
                    if let Ok(retry_str) = retry_after.to_str() {
                        if let Ok(seconds) = retry_str.parse::<u64>() {
                            return Err(DownloaderError::RateLimit(Duration::from_secs(seconds)).into());
                        }
                    }
                }

                if let Some(remaining) = response.headers().get("X-RateLimit-Remaining") {
                    if let Ok(remaining_str) = remaining.to_str() {
                        if remaining_str == "0" {
                            if let Some(reset) = response.headers().get("X-RateLimit-Reset") {
                                if let Ok(reset_str) = reset.to_str() {
                                    if let Ok(reset_timestamp) = reset_str.parse::<i64>() {
                                        let now = Utc::now().timestamp();
                                        let wait_seconds = (reset_timestamp - now).max(0) as u64;
                                        return Err(DownloaderError::RateLimit(Duration::from_secs(wait_seconds)).into());
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if response.status() == StatusCode::UNAUTHORIZED {
                return Err(DownloaderError::Auth("Invalid GitHub token".to_string()).into());
            }

            let response = response.error_for_status()?;
            response.json::<T>().await.map_err(|e| e.into())
        })
        .await
    }

    async fn retry_with_backoff<F, Fut, T>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0;
        let mut delay = Duration::from_millis(self.config.initial_retry_delay_ms);

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if let Some(rate_limit_err) = e.downcast_ref::<DownloaderError>() {
                        if let DownloaderError::RateLimit(wait_duration) = rate_limit_err {
                            if self.config.respect_rate_limits {
                                warn!("Rate limit hit, waiting {:?}", wait_duration);
                                sleep(*wait_duration).await;
                                continue;
                            }
                        }
                    }

                    attempt += 1;
                    if attempt >= self.config.max_retries {
                        error!("Max retries ({}) exceeded", self.config.max_retries);
                        return Err(e);
                    }

                    warn!("Request failed (attempt {}/{}): {}", attempt, self.config.max_retries, e);

                    sleep(delay).await;

                    delay = Duration::from_millis(
                        (delay.as_millis() as u64 * 2).min(self.config.max_retry_delay_ms)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_title() {
        let downloader = create_test_downloader();

        let content = "# Test Title\nSome content";
        let title = downloader.extract_title("test.md", content);
        assert_eq!(title, "Test Title");

        let content_no_heading = "Some content without heading";
        let title = downloader.extract_title("myfile.md", content_no_heading);
        assert_eq!(title, "myfile");
    }

    #[test]
    fn test_extract_properties() {
        let downloader = create_test_downloader();

        let content = r#"
term-id:: 20067
maturity:: mature
tags:: ontology, test
"#;

        let props = downloader.extract_properties(content);
        assert_eq!(props.get("term-id").unwrap()[0], "20067");
        assert_eq!(props.get("maturity").unwrap()[0], "mature");
        assert_eq!(props.get("tags").unwrap().len(), 2);
    }

    #[test]
    fn test_extract_classes() {
        let downloader = create_test_downloader();

        let owl_content = r#"
Declaration(Class(mv:Avatar))
SubClassOf(mv:Avatar mv:VirtualEntity)
"#;

        let classes = downloader.extract_classes(owl_content);
        assert!(classes.contains(&"mv:Avatar".to_string()));
        assert!(classes.len() >= 1);
    }

    #[test]
    fn test_extract_relationships() {
        let downloader = create_test_downloader();

        let owl_content = r#"
SubClassOf(mv:Avatar mv:VirtualEntity)
ObjectPropertyDomain(mv:hasProperty mv:Avatar)
"#;

        let relationships = downloader.extract_relationships(owl_content);
        assert_eq!(relationships.len(), 2);
        assert_eq!(relationships[0].relationship_type, RelationshipType::SubClassOf);
        assert_eq!(relationships[1].relationship_type, RelationshipType::Domain);
    }

    #[test]
    fn test_should_process_file() {
        let downloader = create_test_downloader();

        let content_with_public = r#"
- ### OntologyBlock
public:: true
Some OWL content
"#;
        assert!(downloader.should_process_file(content_with_public));

        let content_no_public = r#"
- ### OntologyBlock
Some OWL content
"#;
        assert!(!downloader.should_process_file(content_no_public));

        let content_no_ontology = r#"
public:: true
Some regular content
"#;
        assert!(!downloader.should_process_file(content_no_ontology));
    }

    #[test]
    fn test_config_from_token() {
        let config = OntologyDownloaderConfig::with_token("test_token".to_string());
        assert_eq!(config.github_token, "test_token");
        assert_eq!(config.repo_owner, "jjohare");
        assert_eq!(config.repo_name, "logseq");
    }

    #[test]
    fn test_progress_percentage() {
        let progress = DownloadProgress {
            total_files: 100,
            processed_files: 25,
            ontology_blocks_found: 10,
            errors: Vec::new(),
            started_at: Utc::now(),
            completed_at: None,
            current_file: None,
        };

        assert_eq!(progress.percentage(), 25.0);
    }

    fn create_test_downloader() -> OntologyDownloader {
        let config = OntologyDownloaderConfig::with_token("test_token".to_string());
        OntologyDownloader::new(config).unwrap()
    }
}
