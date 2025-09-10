use super::api::GitHubClient;
use super::types::GitHubFileBasicMetadata;
use chrono::{DateTime, Utc};
use log::{debug, warn};
use std::error::Error;
use std::sync::Arc;
use serde_json::Value;
use crate::errors::{VisionFlowError, VisionFlowResult, GitHubError, NetworkError, ErrorContext};

/// Enhanced content API that can detect actual file content changes
#[derive(Clone)] // Add Clone trait
pub struct EnhancedContentAPI {
    client: Arc<GitHubClient>,
}

impl EnhancedContentAPI {
    pub fn new(client: Arc<GitHubClient>) -> Self {
        Self { client }
    }

    /// List all markdown files in the repository's base path
    pub async fn list_markdown_files(&self, path: &str) -> VisionFlowResult<Vec<GitHubFileBasicMetadata>> {
        let contents_url = self.client.get_contents_url(path).await;
        debug!("Listing markdown files from: {}", contents_url);

        let response = self.client.client()
            .get(&contents_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("GitHub API error listing files: {}", error_text).into());
        }

        let files: Vec<Value> = response.json().await?;
        let mut markdown_files = Vec::new();

        for file in files {
            if file["type"].as_str() == Some("file") {
                if let Some(name) = file["name"].as_str() {
                    if name.ends_with(".md") {
                        markdown_files.push(GitHubFileBasicMetadata {
                            name: name.to_string(),
                            path: file["path"].as_str().unwrap_or("").to_string(),
                            sha: file["sha"].as_str().unwrap_or("").to_string(),
                            size: file["size"].as_u64().unwrap_or(0),
                            download_url: file["download_url"].as_str().unwrap_or("").to_string(),
                        });
                    }
                }
            } else if file["type"].as_str() == Some("dir") {
                // Recursively list files in subdirectories if needed, but for now, just top-level
                // For this task, we only need top-level markdown files.
            }
        }
        Ok(markdown_files)
    }

    /// Check if a file is public (i.e., its download_url is accessible without authentication)
    pub async fn check_file_public(&self, download_url: &str) -> VisionFlowResult<bool> {
        debug!("Checking if file is public: {}", download_url);
        let client = reqwest::Client::builder()
            .user_agent("github-public-file-checker")
            .timeout(std::time::Duration::from_secs(10))
            .build()?;

        let response = client.get(download_url).send().await?;
        Ok(response.status().is_success())
    }

    /// Fetch the content of a file from its download URL
    pub async fn fetch_file_content(&self, download_url: &str) -> VisionFlowResult<String> {
        debug!("Fetching file content from: {}", download_url);
        let response = self.client.client()
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

    /// Get the last time a file's content was actually modified (not just touched in a commit)
    pub async fn get_file_content_last_modified(
        &self,
        file_path: &str,
        check_actual_changes: bool,
    ) -> VisionFlowResult<DateTime<Utc>> {
        let encoded_path = self.client.get_full_path(file_path).await;
        
        // First, get recent commits for this file
        let commits_url = format!(
            "https://api.github.com/repos/{}/{}/commits",
            self.client.owner(),
            self.client.repo()
        );

        debug!("Fetching commits for path: {}", encoded_path);

        let response = self.client.client()
            .get(&commits_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .query(&[
                ("path", encoded_path.as_str()),
                ("per_page", if check_actual_changes { "10" } else { "1" })
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

        // If not checking actual changes, return the first commit's date
        if !check_actual_changes {
            return self.extract_commit_date(&commits[0]);
        }

        // Check each commit to see if the file was actually modified
        for commit in &commits {
            let sha = commit["sha"]
                .as_str()
                .ok_or("Missing commit SHA")?;
            
            if self.was_file_modified_in_commit(sha, &encoded_path).await? {
                debug!("File was actually modified in commit: {}", sha);
                return self.extract_commit_date(commit);
            } else {
                debug!("File was not modified in commit: {} (likely a merge commit)", sha);
            }
        }

        // If no actual modifications found in recent commits, use the oldest one
        warn!("No actual content changes found in recent commits, using oldest available");
        self.extract_commit_date(&commits[commits.len() - 1])
    }

    /// Check if a specific file was actually modified in a commit
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

        let response = self.client.client()
            .get(&commit_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            warn!("Failed to get commit details: {}", error_text);
            // Assume file was modified if we can't check
            return Ok(true);
        }

        let commit_data: Value = response.json().await?;
        
        // Check if this commit has file changes
        if let Some(files) = commit_data["files"].as_array() {
            for file in files {
                if let Some(filename) = file["filename"].as_str() {
                    // Check if this is our file (need to match the path format)
                    if filename == file_path || filename.ends_with(&format!("/{}", file_path)) || filename == file_path.replace("%2F", "/") || filename.ends_with(&format!("/{}", file_path.replace("%2F", "/"))) {
                        // Check if there were actual changes
                        let additions = file["additions"].as_u64().unwrap_or(0);
                        let deletions = file["deletions"].as_u64().unwrap_or(0);
                        let changes = file["changes"].as_u64().unwrap_or(0);
                        
                        debug!(
                            "File {} in commit {}: +{} -{} (total: {} changes)",
                            filename, commit_sha, additions, deletions, changes
                        );
                        
                        // File was actually modified if there were any changes
                        return Ok(changes > 0);
                    }
                }
            }
        }

        // File was not in the changed files list
        Ok(false)
    }

    /// Extract commit date from commit JSON
    fn extract_commit_date(&self, commit: &Value) -> VisionFlowResult<DateTime<Utc>> {
        // Try committer date first, then author date
        let date_str = commit["commit"]["committer"]["date"]
            .as_str()
            .or_else(|| commit["commit"]["author"]["date"].as_str())
            .ok_or("No commit date found")?;

        DateTime::parse_from_rfc3339(date_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| format!("Failed to parse date {}: {}", date_str, e).into())
    }

    /// Get file metadata including size, SHA, and actual content modification date
    pub async fn get_file_metadata_extended(
        &self,
        file_path: &str,
    ) -> VisionFlowResult<ExtendedFileMetadata> {
        let encoded_path = self.client.get_full_path(file_path).await;
        
        // Get file contents metadata
        let contents_url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.client.owner(),
            self.client.repo(),
            encoded_path
        );

        let response = self.client.client()
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
        
        // Get last content modification date
        let last_content_modified = self
            .get_file_content_last_modified(file_path, true)
            .await?;

        Ok(ExtendedFileMetadata {
            name: content_data["name"].as_str().unwrap_or("").to_string(),
            path: content_data["path"].as_str().unwrap_or("").to_string(),
            sha: content_data["sha"].as_str().unwrap_or("").to_string(),
            size: content_data["size"].as_u64().unwrap_or(0),
            download_url: content_data["download_url"].as_str().unwrap_or("").to_string(),
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