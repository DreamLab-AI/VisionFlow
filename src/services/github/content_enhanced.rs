use super::api::GitHubClient;
use super::types::GitHubFileBasicMetadata;
use crate::errors::VisionFlowResult;
use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use serde_json::Value;
use std::sync::Arc;

///
#[derive(Clone)] 
pub struct EnhancedContentAPI {
    client: Arc<GitHubClient>,
}

impl EnhancedContentAPI {
    pub fn new(client: Arc<GitHubClient>) -> Self {
        Self { client }
    }

    
    pub async fn list_markdown_files(
        &self,
        path: &str,
    ) -> VisionFlowResult<Vec<GitHubFileBasicMetadata>> {
        let contents_url = self.client.get_contents_url(path).await;
        info!(
            "list_markdown_files: Fetching from GitHub API: {}",
            contents_url
        );

        let response = self
            .client
            .client()
            .get(&contents_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        let status = response.status();
        info!(
            "list_markdown_files: GitHub API response status: {}",
            status
        );

        if !status.is_success() {
            let error_text = response.text().await?;
            error!(
                "list_markdown_files: GitHub API error ({}): {}",
                status, error_text
            );
            return Err(format!(
                "GitHub API error listing files ({}): {}",
                status, error_text
            )
            .into());
        }

        let files: Vec<Value> = response.json().await?;
        info!(
            "list_markdown_files: Received {} items from GitHub",
            files.len()
        );

        let mut markdown_files = Vec::new();

        for file in files {
            let file_type = file["type"].as_str().unwrap_or("unknown");
            let file_name = file["name"].as_str().unwrap_or("unnamed");
            debug!(
                "list_markdown_files: Processing item: {} (type: {})",
                file_name, file_type
            );

            if file_type == "file" {
                if file_name.ends_with(".md") {
                    info!("list_markdown_files: Found markdown file: {}", file_name);
                    markdown_files.push(GitHubFileBasicMetadata {
                        name: file_name.to_string(),
                        path: file["path"].as_str().unwrap_or("").to_string(),
                        sha: file["sha"].as_str().unwrap_or("").to_string(),
                        size: file["size"].as_u64().unwrap_or(0),
                        download_url: file["download_url"].as_str().unwrap_or("").to_string(),
                    });
                }
            } else if file_type == "dir" {
                debug!("list_markdown_files: Skipping directory: {}", file_name);
                
                
            }
        }

        info!(
            "list_markdown_files: Found {} markdown files total",
            markdown_files.len()
        );
        Ok(markdown_files)
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
        let encoded_path = self.client.get_full_path(file_path).await;

        
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
        let encoded_path = self.client.get_full_path(file_path).await;

        
        let contents_url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.client.owner(),
            self.client.repo(),
            encoded_path
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
                Utc::now()
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
