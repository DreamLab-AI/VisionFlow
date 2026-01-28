use super::api::GitHubClient;
use super::types::GitHubFileBasicMetadata;
use crate::errors::VisionFlowResult;
use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use serde_json::Value;
use std::sync::Arc;
use crate::utils::time;

#[derive(Clone)] 
pub struct EnhancedContentAPI {
    client: Arc<GitHubClient>,
}

impl EnhancedContentAPI {
    pub fn new(client: Arc<GitHubClient>) -> Self {
        Self { client }
    }


    pub fn list_markdown_files<'a>(
        &'a self,
        path: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = VisionFlowResult<Vec<GitHubFileBasicMetadata>>> + Send + 'a>> {
        Box::pin(async move {
            self.list_markdown_files_impl(path).await
        })
    }

    async fn list_markdown_files_impl(
        &self,
        path: &str,
    ) -> VisionFlowResult<Vec<GitHubFileBasicMetadata>> {
        let mut all_markdown_files = Vec::new();
        let mut page = 1;
        const PER_PAGE: usize = 100;

        info!("list_markdown_files: Starting paginated fetch from GitHub API");

        loop {
            let contents_url = format!(
                "{}&per_page={}&page={}",
                GitHubClient::get_contents_url(&self.client, path).await,
                PER_PAGE,
                page
            );

            debug!("list_markdown_files: Fetching page {} from: {}", page, contents_url);

            let response = self
                .client
                .client()
                .get(&contents_url)
                .header("Authorization", format!("Bearer {}", self.client.token()))
                .header("Accept", "application/vnd.github+json")
                .send()
                .await?;

            let status = response.status();
            debug!("list_markdown_files: Page {} response status: {}", page, status);

            if !status.is_success() {
                let error_text = response.text().await?;
                error!(
                    "list_markdown_files: GitHub API error on page {} ({}): {}",
                    page, status, error_text
                );
                return Err(format!(
                    "GitHub API error listing files page {} ({}): {}",
                    page, status, error_text
                )
                .into());
            }

            let files: Vec<Value> = response.json().await?;
            let files_count = files.len();
            info!(
                "list_markdown_files: Page {} received {} items from GitHub",
                page, files_count
            );

            // Break if no more files
            if files_count == 0 {
                info!("list_markdown_files: No more files, stopping pagination at page {}", page);
                break;
            }

            // Process files on this page
            for file in files {
                let file_type = file["type"].as_str().unwrap_or("unknown");
                let file_name = file["name"].as_str().unwrap_or("unnamed");

                if file_type == "file" && file_name.ends_with(".md") {
                    debug!("list_markdown_files: Found markdown file: {}", file_name);
                    all_markdown_files.push(GitHubFileBasicMetadata {
                        name: file_name.to_string(),
                        path: file["path"].as_str().unwrap_or("").to_string(),
                        sha: file["sha"].as_str().unwrap_or("").to_string(),
                        size: file["size"].as_u64().unwrap_or(0),
                        download_url: file["download_url"].as_str().unwrap_or("").to_string(),
                    });
                } else if file_type == "dir" {
                    let dir_path = file["path"].as_str().unwrap_or("");
                    debug!("list_markdown_files_impl: Recursively processing directory: {}", dir_path);

                    // Recursively fetch markdown files from subdirectory
                    match self.list_markdown_files(dir_path).await {
                        Ok(mut subdir_files) => {
                            let count = subdir_files.len();
                            debug!("list_markdown_files_impl: Found {} files in subdirectory {}", count, dir_path);
                            all_markdown_files.append(&mut subdir_files);
                        }
                        Err(e) => {
                            warn!("list_markdown_files_impl: Failed to process subdirectory {}: {}", dir_path, e);
                        }
                    }
                }
            }

            // GitHub API returns < PER_PAGE items on last page
            if files_count < PER_PAGE {
                info!("list_markdown_files: Last page detected (received {} < {} items)", files_count, PER_PAGE);
                break;
            }

            page += 1;

            // Safety limit to prevent infinite loops
            if page > 100 {
                warn!("list_markdown_files: Reached safety limit of 100 pages (10,000 files)");
                break;
            }
        }

        info!(
            "list_markdown_files: Pagination complete. Found {} markdown files total across {} pages",
            all_markdown_files.len(),
            page
        );
        Ok(all_markdown_files)
    }

    
    pub async fn fetch_file_content(&self, download_url: &str) -> VisionFlowResult<String> {
        debug!("Fetching file content from: {}", download_url);
        let response = self
            .client
            .client()
            .get(download_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Failed to fetch file content: {}", error_text).into());
        }

        Ok(response.text().await?)
    }

    
    pub async fn get_file_content_last_modified(
        &self,
        file_path: &str,
        check_actual_changes: bool,
    ) -> VisionFlowResult<DateTime<Utc>> {
        let encoded_path = GitHubClient::get_full_path(&self.client, file_path).await;

        
        let commits_url = format!(
            "https://api.github.com/repos/{}/{}/commits",
            self.client.owner(),
            self.client.repo()
        );

        debug!("Fetching commits for path: {}", encoded_path);

        let response = self
            .client
            .client()
            .get(&commits_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .query(&[
                ("path", encoded_path.as_str()),
                ("ref", self.client.branch()),
                ("per_page", if check_actual_changes { "10" } else { "1" }),
            ])
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("GitHub API error: {}", error_text).into());
        }

        let commits: Vec<Value> = response.json().await?;

        if commits.is_empty() {
            return Err(format!("No commit history found for {}", file_path).into());
        }

        
        if !check_actual_changes {
            return self.extract_commit_date(&commits[0]);
        }

        
        for commit in &commits {
            let sha = commit["sha"].as_str().ok_or("Missing commit SHA")?;

            if self.was_file_modified_in_commit(sha, &encoded_path).await? {
                debug!("File was actually modified in commit: {}", sha);
                return self.extract_commit_date(commit);
            } else {
                debug!(
                    "File was not modified in commit: {} (likely a merge commit)",
                    sha
                );
            }
        }

        
        warn!("No actual content changes found in recent commits, using oldest available");
        self.extract_commit_date(&commits[commits.len() - 1])
    }

    
    async fn was_file_modified_in_commit(
        &self,
        commit_sha: &str,
        file_path: &str,
    ) -> VisionFlowResult<bool> {
        let commit_url = format!(
            "https://api.github.com/repos/{}/{}/commits/{}",
            self.client.owner(),
            self.client.repo(),
            commit_sha
        );

        debug!("Checking commit {} for file changes", commit_sha);

        let response = self
            .client
            .client()
            .get(&commit_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            warn!("Failed to get commit details: {}", error_text);
            
            return Ok(true);
        }

        let commit_data: Value = response.json().await?;

        
        if let Some(files) = commit_data["files"].as_array() {
            for file in files {
                if let Some(filename) = file["filename"].as_str() {
                    
                    if filename == file_path
                        || filename.ends_with(&format!("/{}", file_path))
                        || filename == file_path.replace("%2F", "/")
                        || filename.ends_with(&format!("/{}", file_path.replace("%2F", "/")))
                    {
                        
                        let additions = file["additions"].as_u64().unwrap_or(0);
                        let deletions = file["deletions"].as_u64().unwrap_or(0);
                        let changes = file["changes"].as_u64().unwrap_or(0);

                        debug!(
                            "File {} in commit {}: +{} -{} (total: {} changes)",
                            filename, commit_sha, additions, deletions, changes
                        );

                        
                        return Ok(changes > 0);
                    }
                }
            }
        }

        
        Ok(false)
    }

    
    fn extract_commit_date(&self, commit: &Value) -> VisionFlowResult<DateTime<Utc>> {
        
        let date_str = commit["commit"]["committer"]["date"]
            .as_str()
            .or_else(|| commit["commit"]["author"]["date"].as_str())
            .ok_or("No commit date found")?;

        DateTime::parse_from_rfc3339(date_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| format!("Failed to parse date {}: {}", date_str, e).into())
    }

    
    pub async fn get_file_metadata_extended(
        &self,
        file_path: &str,
    ) -> VisionFlowResult<ExtendedFileMetadata> {
        let encoded_path = GitHubClient::get_full_path(&self.client, file_path).await;

        
        let contents_url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}?ref={}",
            self.client.owner(),
            self.client.repo(),
            encoded_path,
            self.client.branch()
        );

        let response = self
            .client
            .client()
            .get(&contents_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Failed to get file metadata: {}", error_text).into());
        }

        let content_data: Value = response.json().await?;

        
        let last_content_modified = match self.get_file_content_last_modified(file_path, true).await
        {
            Ok(date) => date,
            Err(e) => {
                
                debug!(
                    "Could not get commit history for {}: {}. Using current time.",
                    file_path, e
                );
                time::now()
            }
        };

        Ok(ExtendedFileMetadata {
            name: content_data["name"].as_str().unwrap_or("").to_string(),
            path: content_data["path"].as_str().unwrap_or("").to_string(),
            sha: content_data["sha"].as_str().unwrap_or("").to_string(),
            size: content_data["size"].as_u64().unwrap_or(0),
            download_url: content_data["download_url"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            last_content_modified,
            file_type: content_data["type"].as_str().unwrap_or("file").to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExtendedFileMetadata {
    pub name: String,
    pub path: String,
    pub sha: String,
    pub size: u64,
    pub download_url: String,
    pub last_content_modified: DateTime<Utc>,
    pub file_type: String,
}
