//! GitHubPRService — Creates GitHub branches, commits, and pull requests
//! for agent-proposed ontology changes.
//!
//! Agents inside the container are authorized to write directly to GitHub.
//! This service uses the GitHub REST API (via reqwest) to:
//! 1. Get the base branch SHA
//! 2. Create a blob with the markdown content
//! 3. Create a tree with the file change
//! 4. Create a commit
//! 5. Create a branch reference
//! 6. Open a pull request
//!
//! Notes are per-user — each user's agents write to their own path namespace.

use crate::types::ontology_tools::AgentContext;
use log::info;
use reqwest::header::{ACCEPT, AUTHORIZATION, USER_AGENT};
use serde::{Deserialize, Serialize};
use std::env;

pub struct GitHubPRService {
    token: String,
    owner: String,
    repo: String,
    base_branch: String,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct CreateBlobRequest {
    content: String,
    encoding: String,
}

#[derive(Debug, Deserialize)]
struct BlobResponse {
    sha: String,
}

#[derive(Debug, Serialize)]
struct CreateTreeRequest {
    base_tree: String,
    tree: Vec<TreeEntry>,
}

#[derive(Debug, Serialize)]
struct TreeEntry {
    path: String,
    mode: String,
    #[serde(rename = "type")]
    entry_type: String,
    sha: String,
}

#[derive(Debug, Deserialize)]
struct TreeResponse {
    sha: String,
}

#[derive(Debug, Serialize)]
struct CreateCommitRequest {
    message: String,
    tree: String,
    parents: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CommitResponse {
    sha: String,
}

#[derive(Debug, Serialize)]
struct CreateRefRequest {
    #[serde(rename = "ref")]
    ref_name: String,
    sha: String,
}

#[derive(Debug, Serialize)]
struct CreatePRRequest {
    title: String,
    body: String,
    head: String,
    base: String,
    labels: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct PRResponse {
    html_url: String,
    number: u64,
}

#[derive(Debug, Deserialize)]
struct RefResponse {
    object: RefObject,
}

#[derive(Debug, Deserialize)]
struct RefObject {
    sha: String,
}

impl GitHubPRService {
    pub fn new() -> Self {
        let token = env::var("GITHUB_TOKEN").unwrap_or_default();
        let owner = env::var("GITHUB_REPO_OWNER").unwrap_or_else(|_| "DreamLab-AI".to_string());
        let repo = env::var("GITHUB_REPO_NAME").unwrap_or_else(|_| "VisionFlow".to_string());
        let base_branch = env::var("GITHUB_BASE_BRANCH").unwrap_or_else(|_| "main".to_string());

        Self {
            token,
            owner,
            repo,
            base_branch,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_config(token: String, owner: String, repo: String, base_branch: String) -> Self {
        Self {
            token,
            owner,
            repo,
            base_branch,
            client: reqwest::Client::new(),
        }
    }

    fn api_url(&self, path: &str) -> String {
        format!(
            "https://api.github.com/repos/{}/{}/{}",
            self.owner, self.repo, path
        )
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", self.token).parse().unwrap(),
        );
        headers.insert(ACCEPT, "application/vnd.github+json".parse().unwrap());
        headers.insert(USER_AGENT, "VisionFlow-OntologyAgent/1.0".parse().unwrap());
        headers
    }

    /// Create a full GitHub PR for an ontology change.
    ///
    /// Returns the PR URL on success.
    pub async fn create_ontology_pr(
        &self,
        file_path: &str,
        content: &str,
        title: &str,
        body: &str,
        agent_ctx: &AgentContext,
    ) -> Result<String, String> {
        if self.token.is_empty() {
            return Err("GITHUB_TOKEN not configured — cannot create PR".to_string());
        }

        info!(
            "Creating ontology PR: '{}' for file '{}'",
            title, file_path
        );

        // 1. Get base branch SHA
        let base_sha = self.get_ref_sha(&self.base_branch).await?;

        // 2. Create blob
        let blob_sha = self.create_blob(content).await?;

        // 3. Create tree
        let tree_sha = self.create_tree(&base_sha, file_path, &blob_sha).await?;

        // 4. Create commit
        let commit_message = format!(
            "{}\n\nAgent: {} ({})\nUser: {}\nTask: {}",
            title,
            agent_ctx.agent_type,
            agent_ctx.agent_id,
            agent_ctx.user_id,
            agent_ctx.task_description
        );
        let commit_sha = self.create_commit(&commit_message, &tree_sha, &base_sha).await?;

        // 5. Create branch
        let branch_name = format!(
            "ontology/{}-{}",
            agent_ctx.agent_type,
            &agent_ctx.agent_id[..8.min(agent_ctx.agent_id.len())]
        );
        self.create_ref(&branch_name, &commit_sha).await?;

        // 6. Create PR
        let pr_url = self
            .create_pull_request(title, body, &branch_name)
            .await?;

        info!("Created ontology PR: {}", pr_url);
        Ok(pr_url)
    }

    async fn get_ref_sha(&self, branch: &str) -> Result<String, String> {
        let url = self.api_url(&format!("git/ref/heads/{}", branch));
        let resp = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| format!("Failed to get ref: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Get ref failed ({}): {}", status, body));
        }

        let ref_resp: RefResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse ref response: {}", e))?;

        Ok(ref_resp.object.sha)
    }

    async fn create_blob(&self, content: &str) -> Result<String, String> {
        let url = self.api_url("git/blobs");
        let body = CreateBlobRequest {
            content: content.to_string(),
            encoding: "utf-8".to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Failed to create blob: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Create blob failed ({}): {}", status, body));
        }

        let blob: BlobResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse blob response: {}", e))?;

        Ok(blob.sha)
    }

    async fn create_tree(
        &self,
        base_tree_sha: &str,
        file_path: &str,
        blob_sha: &str,
    ) -> Result<String, String> {
        let url = self.api_url("git/trees");
        let body = CreateTreeRequest {
            base_tree: base_tree_sha.to_string(),
            tree: vec![TreeEntry {
                path: file_path.to_string(),
                mode: "100644".to_string(),
                entry_type: "blob".to_string(),
                sha: blob_sha.to_string(),
            }],
        };

        let resp = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Failed to create tree: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Create tree failed ({}): {}", status, body));
        }

        let tree: TreeResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse tree response: {}", e))?;

        Ok(tree.sha)
    }

    async fn create_commit(
        &self,
        message: &str,
        tree_sha: &str,
        parent_sha: &str,
    ) -> Result<String, String> {
        let url = self.api_url("git/commits");
        let body = CreateCommitRequest {
            message: message.to_string(),
            tree: tree_sha.to_string(),
            parents: vec![parent_sha.to_string()],
        };

        let resp = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Failed to create commit: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Create commit failed ({}): {}", status, body));
        }

        let commit: CommitResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse commit response: {}", e))?;

        Ok(commit.sha)
    }

    async fn create_ref(&self, branch: &str, sha: &str) -> Result<(), String> {
        let url = self.api_url("git/refs");
        let body = CreateRefRequest {
            ref_name: format!("refs/heads/{}", branch),
            sha: sha.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Failed to create ref: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.as_u16() == 422 {
                // Branch already exists — force-update it to the new commit
                info!(
                    "Branch '{}' already exists, force-updating to SHA {}",
                    branch, sha
                );
                self.update_ref(branch, sha).await?;
            } else {
                return Err(format!("Create ref failed ({}): {}", status, body));
            }
        }

        Ok(())
    }

    /// Force-update an existing branch ref to a new SHA.
    async fn update_ref(&self, branch: &str, sha: &str) -> Result<(), String> {
        let url = self.api_url(&format!("git/refs/heads/{}", branch));

        let body = serde_json::json!({
            "sha": sha,
            "force": true,
        });

        let resp = self
            .client
            .patch(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Failed to update ref: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let resp_body = resp.text().await.unwrap_or_default();
            return Err(format!("Update ref failed ({}): {}", status, resp_body));
        }

        info!("Force-updated branch '{}' to SHA {}", branch, sha);
        Ok(())
    }

    async fn create_pull_request(
        &self,
        title: &str,
        body: &str,
        head_branch: &str,
    ) -> Result<String, String> {
        let url = self.api_url("pulls");
        let pr_body = CreatePRRequest {
            title: title.to_string(),
            body: body.to_string(),
            head: head_branch.to_string(),
            base: self.base_branch.clone(),
            labels: Some(vec![
                "ontology".to_string(),
                "agent-proposed".to_string(),
            ]),
        };

        let resp = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&pr_body)
            .send()
            .await
            .map_err(|e| format!("Failed to create PR: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let resp_body = resp.text().await.unwrap_or_default();
            if status.as_u16() == 422 {
                // PR already exists for this head/base pair — fetch the existing one
                info!(
                    "PR already exists for branch '{}', fetching existing PR URL",
                    head_branch
                );
                return self.get_existing_pr_url(head_branch).await;
            }
            return Err(format!("Create PR failed ({}): {}", status, resp_body));
        }

        let pr: PRResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse PR response: {}", e))?;

        Ok(pr.html_url)
    }

    /// Fetch the URL of an existing open PR for the given head branch.
    async fn get_existing_pr_url(&self, head_branch: &str) -> Result<String, String> {
        let url = format!(
            "{}?head={}:{}&state=open",
            self.api_url("pulls"),
            self.owner,
            head_branch
        );

        let resp = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| format!("Failed to fetch existing PRs: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Fetch existing PRs failed ({}): {}", status, body));
        }

        let prs: Vec<PRResponse> = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse PR list response: {}", e))?;

        prs.first()
            .map(|pr| pr.html_url.clone())
            .ok_or_else(|| {
                format!(
                    "PR creation returned 422 but no open PR found for branch '{}'",
                    head_branch
                )
            })
    }
}
