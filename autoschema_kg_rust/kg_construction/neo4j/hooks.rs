//! Coordination hooks integration for Neo4j operations

use crate::neo4j::{
    connection::Neo4jConnectionManager,
    error::{Neo4jError, Result},
    models::{GraphData, Triple},
    operations::{GraphOperations, GraphOperationsImpl, ImportResult, BatchResult},
};
use log::{debug, info, warn, error};
use serde_json::{json, Value};
use std::process::Command;
use std::time::Instant;
use tokio::process::Command as AsyncCommand;

/// Hook events for coordination
#[derive(Debug, Clone)]
pub enum HookEvent {
    PreTask { description: String },
    PostTask { task_id: String },
    PreEdit { file_path: String },
    PostEdit { file_path: String, memory_key: String },
    Notify { message: String },
    SessionRestore { session_id: String },
    SessionEnd { export_metrics: bool },
    MemoryStore { key: String, data: Value },
    MemoryRetrieve { key: String },
}

/// Coordination hooks for Neo4j operations
pub struct Neo4jHooks {
    connection_manager: Neo4jConnectionManager,
    graph_ops: GraphOperationsImpl,
    session_id: Option<String>,
    task_id: Option<String>,
}

impl Neo4jHooks {
    /// Create a new hooks instance
    pub fn new(connection_manager: Neo4jConnectionManager) -> Self {
        let graph_ops = GraphOperationsImpl::new(connection_manager.clone());
        
        Self {
            connection_manager,
            graph_ops,
            session_id: None,
            task_id: None,
        }
    }
    
    /// Execute a hook event
    pub async fn execute_hook(&mut self, event: HookEvent) -> Result<()> {
        debug!("Executing hook event: {:?}", event);
        
        match event {
            HookEvent::PreTask { description } => {
                self.handle_pre_task(&description).await
            }
            HookEvent::PostTask { task_id } => {
                self.handle_post_task(&task_id).await
            }
            HookEvent::PreEdit { file_path } => {
                self.handle_pre_edit(&file_path).await
            }
            HookEvent::PostEdit { file_path, memory_key } => {
                self.handle_post_edit(&file_path, &memory_key).await
            }
            HookEvent::Notify { message } => {
                self.handle_notify(&message).await
            }
            HookEvent::SessionRestore { session_id } => {
                self.handle_session_restore(&session_id).await
            }
            HookEvent::SessionEnd { export_metrics } => {
                self.handle_session_end(export_metrics).await
            }
            HookEvent::MemoryStore { key, data } => {
                self.handle_memory_store(&key, data).await
            }
            HookEvent::MemoryRetrieve { key } => {
                self.handle_memory_retrieve(&key).await
            }
        }
    }
    
    /// Handle pre-task hook
    async fn handle_pre_task(&mut self, description: &str) -> Result<()> {
        info!("Starting task: {}", description);
        
        // Execute claude-flow hook
        match self.execute_claude_flow_hook("pre-task", &[("description", description)]).await {
            Ok(output) => {
                debug!("Pre-task hook output: {}", output);
                
                // Parse task ID from output if available
                if let Some(task_id) = self.extract_task_id(&output) {
                    self.task_id = Some(task_id.clone());
                    info!("Task ID: {}", task_id);
                }
            }
            Err(e) => {
                warn!("Pre-task hook failed: {}", e);
            }
        }
        
        // Store task metadata in graph
        let task_triple = Triple::new(
            "system:task",
            "hasDescription",
            description
        )
        .with_source("neo4j_hooks")
        .with_metadata("timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        
        if let Err(e) = self.graph_ops.store_triple(&task_triple).await {
            warn!("Failed to store task metadata: {}", e);
        }
        
        Ok(())
    }
    
    /// Handle post-task hook
    async fn handle_post_task(&mut self, task_id: &str) -> Result<()> {
        info!("Completing task: {}", task_id);
        
        // Execute claude-flow hook
        if let Err(e) = self.execute_claude_flow_hook("post-task", &[("task-id", task_id)]).await {
            warn!("Post-task hook failed: {}", e);
        }
        
        // Store completion metadata
        let completion_triple = Triple::new(
            &format!("task:{}", task_id),
            "hasStatus",
            "completed"
        )
        .with_source("neo4j_hooks")
        .with_metadata("timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        
        if let Err(e) = self.graph_ops.store_triple(&completion_triple).await {
            warn!("Failed to store completion metadata: {}", e);
        }
        
        self.task_id = None;
        Ok(())
    }
    
    /// Handle pre-edit hook
    async fn handle_pre_edit(&self, file_path: &str) -> Result<()> {
        debug!("Pre-edit: {}", file_path);
        
        // Store file edit metadata
        let edit_triple = Triple::new(
            &format!("file:{}", file_path),
            "willBeEdited",
            "true"
        )
        .with_source("neo4j_hooks")
        .with_metadata("timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        
        if let Err(e) = self.graph_ops.store_triple(&edit_triple).await {
            warn!("Failed to store pre-edit metadata: {}", e);
        }
        
        Ok(())
    }
    
    /// Handle post-edit hook
    async fn handle_post_edit(&self, file_path: &str, memory_key: &str) -> Result<()> {
        debug!("Post-edit: {} -> {}", file_path, memory_key);
        
        // Execute claude-flow hook
        if let Err(e) = self.execute_claude_flow_hook(
            "post-edit",
            &[("file", file_path), ("memory-key", memory_key)]
        ).await {
            warn!("Post-edit hook failed: {}", e);
        }
        
        // Store edit completion metadata
        let edit_triple = Triple::new(
            &format!("file:{}", file_path),
            "wasEdited",
            "true"
        )
        .with_source("neo4j_hooks")
        .with_metadata("memory_key".to_string(), json!(memory_key))
        .with_metadata("timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        
        if let Err(e) = self.graph_ops.store_triple(&edit_triple).await {
            warn!("Failed to store post-edit metadata: {}", e);
        }
        
        Ok(())
    }
    
    /// Handle notify hook
    async fn handle_notify(&self, message: &str) -> Result<()> {
        info!("Notification: {}", message);
        
        // Execute claude-flow hook
        if let Err(e) = self.execute_claude_flow_hook("notify", &[("message", message)]).await {
            warn!("Notify hook failed: {}", e);
        }
        
        // Store notification in graph
        let notify_triple = Triple::new(
            "system:notifications",
            "hasMessage",
            message
        )
        .with_source("neo4j_hooks")
        .with_metadata("timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        
        if let Err(e) = self.graph_ops.store_triple(&notify_triple).await {
            warn!("Failed to store notification: {}", e);
        }
        
        Ok(())
    }
    
    /// Handle session restore hook
    async fn handle_session_restore(&mut self, session_id: &str) -> Result<()> {
        info!("Restoring session: {}", session_id);
        
        self.session_id = Some(session_id.to_string());
        
        // Execute claude-flow hook
        if let Err(e) = self.execute_claude_flow_hook(
            "session-restore",
            &[("session-id", session_id)]
        ).await {
            warn!("Session restore hook failed: {}", e);
        }
        
        // Try to restore session data from graph
        match self.graph_ops.search_triples(
            Some(&format!("session:{}", session_id)),
            None,
            None
        ).await {
            Ok(triples) => {
                info!("Restored {} session triples", triples.len());
                for triple in &triples {
                    debug!("Session data: {} -> {} -> {}", triple.subject, triple.predicate, triple.object);
                }
            }
            Err(e) => {
                warn!("Failed to restore session data: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Handle session end hook
    async fn handle_session_end(&self, export_metrics: bool) -> Result<()> {
        info!("Ending session, export_metrics: {}", export_metrics);
        
        // Execute claude-flow hook
        let export_flag = if export_metrics { "true" } else { "false" };
        if let Err(e) = self.execute_claude_flow_hook(
            "session-end",
            &[("export-metrics", export_flag)]
        ).await {
            warn!("Session end hook failed: {}", e);
        }
        
        if export_metrics {
            // Export session metrics
            if let Err(e) = self.export_session_metrics().await {
                warn!("Failed to export session metrics: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Handle memory store hook
    async fn handle_memory_store(&self, key: &str, data: Value) -> Result<()> {
        debug!("Storing memory: {} -> {:?}", key, data);
        
        // Store in graph as triple
        let memory_triple = Triple::new(
            &format!("memory:{}", key),
            "hasData",
            &data.to_string()
        )
        .with_source("neo4j_hooks")
        .with_metadata("data_type".to_string(), json!("memory"))
        .with_metadata("timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        
        self.graph_ops.store_triple(&memory_triple).await?;
        info!("Stored memory data for key: {}", key);
        Ok(())
    }
    
    /// Handle memory retrieve hook
    async fn handle_memory_retrieve(&self, key: &str) -> Result<()> {
        debug!("Retrieving memory: {}", key);
        
        // Retrieve from graph
        match self.graph_ops.search_triples(
            Some(&format!("memory:{}", key)),
            Some("hasData"),
            None
        ).await {
            Ok(triples) => {
                if let Some(triple) = triples.first() {
                    info!("Retrieved memory data for key {}: {}", key, triple.object);
                } else {
                    warn!("No memory data found for key: {}", key);
                }
            }
            Err(e) => {
                warn!("Failed to retrieve memory data for key {}: {}", key, e);
            }
        }
        
        Ok(())
    }
    
    /// Execute claude-flow hook command
    async fn execute_claude_flow_hook(&self, hook_name: &str, args: &[(&str, &str)]) -> Result<String> {
        let mut cmd = AsyncCommand::new("npx");
        cmd.arg("claude-flow@alpha")
           .arg("hooks")
           .arg(hook_name);
        
        // Add arguments
        for (key, value) in args {
            cmd.arg(format!("--{}", key))
               .arg(value);
        }
        
        debug!("Executing hook command: {:?}", cmd);
        
        let output = cmd.output().await.map_err(|e| {
            Neo4jError::query_error(format!("Failed to execute hook command: {}", e))
        })?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Hook command failed: {}", stderr);
            return Err(Neo4jError::query_error(format!("Hook command failed: {}", stderr)));
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout)
    }
    
    /// Extract task ID from hook output
    fn extract_task_id(&self, output: &str) -> Option<String> {
        // Look for task ID pattern in output
        if let Some(start) = output.find("Task ID: ") {
            let start_pos = start + "Task ID: ".len();
            if let Some(end) = output[start_pos..].find('\n') {
                return Some(output[start_pos..start_pos + end].trim().to_string());
            }
        }
        None
    }
    
    /// Export session metrics to graph
    async fn export_session_metrics(&self) -> Result<()> {
        info!("Exporting session metrics");
        
        // Get graph statistics
        let stats = self.graph_ops.get_graph_statistics().await?;
        
        // Store session metrics as triples
        let session_key = self.session_id.as_deref().unwrap_or("unknown");
        
        let metrics_triples = vec![
            Triple::new(
                &format!("session:{}", session_key),
                "hasNodeCount",
                &stats.node_count.to_string()
            ),
            Triple::new(
                &format!("session:{}", session_key),
                "hasRelationshipCount",
                &stats.relationship_count.to_string()
            ),
            Triple::new(
                &format!("session:{}", session_key),
                "exportedAt",
                &chrono::Utc::now().to_rfc3339()
            ),
        ];
        
        for triple in metrics_triples {
            if let Err(e) = self.graph_ops.store_triple(&triple).await {
                warn!("Failed to store metric triple: {}", e);
            }
        }
        
        info!("Session metrics exported successfully");
        Ok(())
    }
    
    /// Store operation metadata in graph
    pub async fn store_operation_metadata(&self, operation: &str, metadata: Value) -> Result<()> {
        let operation_triple = Triple::new(
            &format!("operation:{}", operation),
            "hasMetadata",
            &metadata.to_string()
        )
        .with_source("neo4j_hooks")
        .with_metadata("timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        
        self.graph_ops.store_triple(&operation_triple).await?;
        debug!("Stored operation metadata for: {}", operation);
        Ok(())
    }
    
    /// Retrieve operation history from graph
    pub async fn get_operation_history(&self, operation: &str) -> Result<Vec<Triple>> {
        self.graph_ops.search_triples(
            Some(&format!("operation:{}", operation)),
            None,
            None
        ).await
    }
    
    /// Store performance metrics
    pub async fn store_performance_metrics(&self, operation: &str, duration_ms: u64, success: bool) -> Result<()> {
        let perf_triple = Triple::new(
            &format!("performance:{}", operation),
            "hasDuration",
            &duration_ms.to_string()
        )
        .with_source("neo4j_hooks")
        .with_metadata("success".to_string(), json!(success))
        .with_metadata("timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        
        self.graph_ops.store_triple(&perf_triple).await?;
        debug!("Stored performance metrics for: {} ({}ms, success: {})", operation, duration_ms, success);
        Ok(())
    }
    
    /// Get current session ID
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }
    
    /// Get current task ID
    pub fn task_id(&self) -> Option<&str> {
        self.task_id.as_deref()
    }
    
    /// Check hook availability
    pub async fn check_hooks_available(&self) -> bool {
        match self.execute_claude_flow_hook("status", &[]).await {
            Ok(_) => {
                info!("Claude-flow hooks are available");
                true
            }
            Err(_) => {
                warn!("Claude-flow hooks are not available");
                false
            }
        }
    }
}

/// Convenience functions for common hook operations
pub struct HookHelpers;

impl HookHelpers {
    /// Execute pre-task hook
    pub async fn pre_task(hooks: &mut Neo4jHooks, description: &str) -> Result<()> {
        hooks.execute_hook(HookEvent::PreTask {
            description: description.to_string(),
        }).await
    }
    
    /// Execute post-task hook
    pub async fn post_task(hooks: &mut Neo4jHooks, task_id: &str) -> Result<()> {
        hooks.execute_hook(HookEvent::PostTask {
            task_id: task_id.to_string(),
        }).await
    }
    
    /// Execute post-edit hook
    pub async fn post_edit(hooks: &mut Neo4jHooks, file_path: &str, memory_key: &str) -> Result<()> {
        hooks.execute_hook(HookEvent::PostEdit {
            file_path: file_path.to_string(),
            memory_key: memory_key.to_string(),
        }).await
    }
    
    /// Execute notify hook
    pub async fn notify(hooks: &mut Neo4jHooks, message: &str) -> Result<()> {
        hooks.execute_hook(HookEvent::Notify {
            message: message.to_string(),
        }).await
    }
    
    /// Store data in memory via hooks
    pub async fn store_memory(hooks: &mut Neo4jHooks, key: &str, data: Value) -> Result<()> {
        hooks.execute_hook(HookEvent::MemoryStore {
            key: key.to_string(),
            data,
        }).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_hook_event_creation() {
        let event = HookEvent::PreTask {
            description: "Test task".to_string(),
        };
        
        match event {
            HookEvent::PreTask { description } => {
                assert_eq!(description, "Test task");
            }
            _ => panic!("Wrong event type"),
        }
    }
    
    #[test]
    fn test_memory_event() {
        let data = json!({"key": "value"});
        let event = HookEvent::MemoryStore {
            key: "test_key".to_string(),
            data: data.clone(),
        };
        
        match event {
            HookEvent::MemoryStore { key, data: stored_data } => {
                assert_eq!(key, "test_key");
                assert_eq!(stored_data, data);
            }
            _ => panic!("Wrong event type"),
        }
    }
}
