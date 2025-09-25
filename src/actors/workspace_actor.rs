//! Workspace Actor for managing workspace state and persistence
//!
//! This actor handles all workspace operations including:
//! - CRUD operations for workspaces
//! - Persistent storage to file system
//! - WebSocket notifications for real-time updates
//! - Filtering and pagination support

use actix::prelude::*;
use anyhow::{anyhow, Result};
use log::{debug, error, info};
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use std::fs;
use std::path::Path;
use validator::Validate;

use crate::actors::messages::{
    WorkspaceStateChanged, WorkspaceChangeType, GetWorkspaces, GetWorkspace, CreateWorkspace,
    UpdateWorkspace, DeleteWorkspace, ToggleFavoriteWorkspace, ArchiveWorkspace,
    GetWorkspaceCount, LoadWorkspaces, SaveWorkspaces,
};
use crate::models::workspace::{
    Workspace, WorkspaceListResponse, WorkspaceSortBy, SortDirection, WorkspaceFilter,
};

/// WebSocket client for broadcasting workspace updates
pub trait WorkspaceWebSocketClient: Send + Sync {
    fn broadcast_workspace_change(&self, workspace: &Workspace, change_type: WorkspaceChangeType);
}

/// Default WebSocket client implementation (no-op for testing)
#[derive(Clone, Default)]
pub struct DefaultWebSocketClient;

impl WorkspaceWebSocketClient for DefaultWebSocketClient {
    fn broadcast_workspace_change(&self, _workspace: &Workspace, _change_type: WorkspaceChangeType) {
        debug!("WebSocket broadcast: workspace change (no-op implementation)");
    }
}

/// Workspace Actor manages all workspace-related operations
pub struct WorkspaceActor {
    /// In-memory storage of workspaces indexed by ID
    workspaces: HashMap<String, Workspace>,

    /// Path to the persistent storage file
    storage_path: String,

    /// WebSocket client for broadcasting updates
    websocket_client: Arc<dyn WorkspaceWebSocketClient>,

    /// Actor started flag for lazy loading
    initialized: bool,
}

impl WorkspaceActor {
    /// Create a new WorkspaceActor with default storage path
    pub fn new() -> Self {
        Self::with_storage_path("data/workspaces.json".to_string())
    }

    /// Create a new WorkspaceActor with custom storage path
    pub fn with_storage_path(storage_path: String) -> Self {
        Self {
            workspaces: HashMap::new(),
            storage_path,
            websocket_client: Arc::new(DefaultWebSocketClient),
            initialized: false,
        }
    }

    /// Create WorkspaceActor with custom WebSocket client
    pub fn with_websocket_client(storage_path: String, websocket_client: Arc<dyn WorkspaceWebSocketClient>) -> Self {
        Self {
            workspaces: HashMap::new(),
            storage_path,
            websocket_client,
            initialized: false,
        }
    }

    /// Load workspaces from persistent storage
    fn load_from_storage(&mut self) -> Result<()> {
        if !Path::new(&self.storage_path).exists() {
            info!("Storage file {} doesn't exist, starting with empty workspace collection", self.storage_path);
            return Ok(());
        }

        let contents = fs::read_to_string(&self.storage_path)
            .map_err(|e| anyhow!("Failed to read workspaces file: {}", e))?;

        if contents.trim().is_empty() {
            info!("Storage file {} is empty, starting with empty workspace collection", self.storage_path);
            return Ok(());
        }

        let workspaces: Vec<Workspace> = serde_json::from_str(&contents)
            .map_err(|e| anyhow!("Failed to parse workspaces JSON: {}", e))?;

        self.workspaces.clear();
        for workspace in workspaces {
            self.workspaces.insert(workspace.id.clone(), workspace);
        }

        info!("Loaded {} workspaces from storage", self.workspaces.len());
        Ok(())
    }

    /// Save workspaces to persistent storage
    fn save_to_storage(&self) -> Result<()> {
        // Ensure data directory exists
        if let Some(parent) = Path::new(&self.storage_path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Failed to create data directory: {}", e))?;
        }

        let workspaces: Vec<&Workspace> = self.workspaces.values().collect();
        let json = serde_json::to_string_pretty(&workspaces)
            .map_err(|e| anyhow!("Failed to serialize workspaces: {}", e))?;

        fs::write(&self.storage_path, json)
            .map_err(|e| anyhow!("Failed to write workspaces file: {}", e))?;

        debug!("Saved {} workspaces to storage", workspaces.len());
        Ok(())
    }

    /// Apply filters to workspace list
    fn apply_filters<'a>(&self, workspaces: Vec<&'a Workspace>, filter: &Option<WorkspaceFilter>) -> Vec<&'a Workspace> {
        let Some(filter) = filter else {
            return workspaces;
        };

        workspaces
            .into_iter()
            .filter(|workspace| {
                // Status filter
                if let Some(status) = &filter.status {
                    if workspace.status != *status {
                        return false;
                    }
                }

                // Type filter
                if let Some(workspace_type) = &filter.workspace_type {
                    if workspace.workspace_type != *workspace_type {
                        return false;
                    }
                }

                // Favorite filter
                if let Some(is_favorite) = filter.is_favorite {
                    if workspace.is_favorite != is_favorite {
                        return false;
                    }
                }

                // Owner filter
                if let Some(owner_id) = &filter.owner_id {
                    if workspace.owner_id.as_ref() != Some(owner_id) {
                        return false;
                    }
                }

                // Search filter (name and description)
                if let Some(search) = &filter.search {
                    let search_lower = search.to_lowercase();
                    let name_matches = workspace.name.to_lowercase().contains(&search_lower);
                    let description_matches = workspace.description
                        .as_ref()
                        .map(|desc| desc.to_lowercase().contains(&search_lower))
                        .unwrap_or(false);

                    if !name_matches && !description_matches {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    /// Apply sorting to workspace list
    fn apply_sorting<'a>(&self, mut workspaces: Vec<&'a Workspace>, sort_by: &WorkspaceSortBy, sort_direction: &SortDirection) -> Vec<&'a Workspace> {
        workspaces.sort_by(|a, b| {
            let ordering = match sort_by {
                WorkspaceSortBy::Name => a.name.cmp(&b.name),
                WorkspaceSortBy::CreatedAt => a.created_at.cmp(&b.created_at),
                WorkspaceSortBy::UpdatedAt => a.updated_at.cmp(&b.updated_at),
                WorkspaceSortBy::MemberCount => a.member_count.cmp(&b.member_count),
            };

            match sort_direction {
                SortDirection::Ascending => ordering,
                SortDirection::Descending => ordering.reverse(),
            }
        });

        workspaces
    }

    /// Apply pagination to workspace list
    fn apply_pagination(&self, workspaces: Vec<&Workspace>, page: usize, page_size: usize) -> Vec<Workspace> {
        let start = page * page_size;
        let end = std::cmp::min(start + page_size, workspaces.len());

        if start >= workspaces.len() {
            return Vec::new();
        }

        workspaces[start..end]
            .iter()
            .map(|&workspace| workspace.clone())
            .collect()
    }

    /// Notify WebSocket clients of workspace changes
    fn notify_change(&self, workspace: &Workspace, change_type: WorkspaceChangeType) {
        self.websocket_client.broadcast_workspace_change(workspace, change_type);
    }

    /// Initialize the actor by loading existing data
    fn ensure_initialized(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        self.load_from_storage()?;
        self.initialized = true;
        Ok(())
    }
}

impl Default for WorkspaceActor {
    fn default() -> Self {
        Self::new()
    }
}

impl Actor for WorkspaceActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("WorkspaceActor started");

        // Load existing workspaces
        if let Err(e) = self.ensure_initialized() {
            error!("Failed to initialize WorkspaceActor: {}", e);
        }
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("WorkspaceActor stopped");

        // Save workspaces before shutdown
        if let Err(e) = self.save_to_storage() {
            error!("Failed to save workspaces during shutdown: {}", e);
        }
    }
}

// Message Handlers

impl Handler<GetWorkspaces> for WorkspaceActor {
    type Result = Result<WorkspaceListResponse, String>;

    fn handle(&mut self, msg: GetWorkspaces, _ctx: &mut Self::Context) -> Self::Result {
        self.ensure_initialized()
            .map_err(|e| format!("Failed to initialize: {}", e))?;

        let query = msg.query;

        // Apply defaults
        let page = query.page.unwrap_or(0);
        let page_size = query.page_size.unwrap_or(20).min(1000); // Cap at 1000
        let sort_by = query.sort_by.unwrap_or(WorkspaceSortBy::UpdatedAt);
        let sort_direction = query.sort_direction.unwrap_or(SortDirection::Descending);

        // Get all workspaces as references
        let all_workspaces: Vec<&Workspace> = self.workspaces.values().collect();
        let total_count = all_workspaces.len();

        // Apply filters
        let filtered_workspaces = self.apply_filters(all_workspaces, &query.filter);
        let filtered_count = filtered_workspaces.len();

        // Apply sorting
        let sorted_workspaces = self.apply_sorting(filtered_workspaces, &sort_by, &sort_direction);

        // Apply pagination
        let paginated_workspaces = self.apply_pagination(sorted_workspaces, page, page_size);

        debug!("Retrieved {} workspaces (filtered from {} total)", paginated_workspaces.len(), total_count);

        Ok(WorkspaceListResponse::success(
            paginated_workspaces,
            filtered_count,
            page,
            page_size
        ))
    }
}

impl Handler<GetWorkspace> for WorkspaceActor {
    type Result = Result<Workspace, String>;

    fn handle(&mut self, msg: GetWorkspace, _ctx: &mut Self::Context) -> Self::Result {
        self.ensure_initialized()
            .map_err(|e| format!("Failed to initialize: {}", e))?;

        self.workspaces
            .get(&msg.workspace_id)
            .cloned()
            .ok_or_else(|| format!("Workspace with ID '{}' not found", msg.workspace_id))
    }
}

impl Handler<CreateWorkspace> for WorkspaceActor {
    type Result = Result<Workspace, String>;

    fn handle(&mut self, msg: CreateWorkspace, _ctx: &mut Self::Context) -> Self::Result {
        self.ensure_initialized()
            .map_err(|e| format!("Failed to initialize: {}", e))?;

        // Validate the request
        msg.request.validate()
            .map_err(|e| format!("Validation error: {}", e))?;

        let mut workspace = Workspace::new(
            msg.request.name.clone(),
            msg.request.description.clone(),
            msg.request.workspace_type.unwrap_or_default(),
        );

        // Set optional fields
        if let Some(owner_id) = msg.request.owner_id {
            workspace.set_owner(owner_id);
        }

        if let Some(metadata) = msg.request.metadata {
            for (key, value) in metadata {
                workspace.set_metadata(key, value);
            }
        }

        // Insert into storage
        let workspace_id = workspace.id.clone();
        self.workspaces.insert(workspace_id.clone(), workspace.clone());

        // Save to persistent storage
        if let Err(e) = self.save_to_storage() {
            error!("Failed to save workspaces after creation: {}", e);
            self.workspaces.remove(&workspace_id); // Rollback
            return Err(format!("Failed to persist workspace: {}", e));
        }

        // Notify WebSocket clients
        self.notify_change(&workspace, WorkspaceChangeType::Created);

        info!("Created workspace '{}' with ID: {}", workspace.name, workspace.id);
        Ok(workspace)
    }
}

impl Handler<UpdateWorkspace> for WorkspaceActor {
    type Result = Result<Workspace, String>;

    fn handle(&mut self, msg: UpdateWorkspace, _ctx: &mut Self::Context) -> Self::Result {
        self.ensure_initialized()
            .map_err(|e| format!("Failed to initialize: {}", e))?;

        // Validate the request
        msg.request.validate()
            .map_err(|e| format!("Validation error: {}", e))?;

        let workspace = self.workspaces
            .get_mut(&msg.workspace_id)
            .ok_or_else(|| format!("Workspace with ID '{}' not found", msg.workspace_id))?;

        // Update the workspace
        workspace.update(
            msg.request.name,
            msg.request.description,
            msg.request.workspace_type,
        );

        // Update metadata if provided
        if let Some(metadata) = msg.request.metadata {
            for (key, value) in metadata {
                workspace.set_metadata(key, value);
            }
        }

        let updated_workspace = workspace.clone();

        // Save to persistent storage
        if let Err(e) = self.save_to_storage() {
            error!("Failed to save workspaces after update: {}", e);
            return Err(format!("Failed to persist workspace update: {}", e));
        }

        // Notify WebSocket clients
        self.notify_change(&updated_workspace, WorkspaceChangeType::Updated);

        info!("Updated workspace '{}' with ID: {}", updated_workspace.name, updated_workspace.id);
        Ok(updated_workspace)
    }
}

impl Handler<DeleteWorkspace> for WorkspaceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: DeleteWorkspace, _ctx: &mut Self::Context) -> Self::Result {
        self.ensure_initialized()
            .map_err(|e| format!("Failed to initialize: {}", e))?;

        let mut workspace = self.workspaces
            .get(&msg.workspace_id)
            .cloned()
            .ok_or_else(|| format!("Workspace with ID '{}' not found", msg.workspace_id))?;

        // Soft delete - just archive the workspace
        workspace.archive();
        self.workspaces.insert(msg.workspace_id.clone(), workspace.clone());

        // Save to persistent storage
        if let Err(e) = self.save_to_storage() {
            error!("Failed to save workspaces after deletion: {}", e);
            return Err(format!("Failed to persist workspace deletion: {}", e));
        }

        // Notify WebSocket clients
        self.notify_change(&workspace, WorkspaceChangeType::Deleted);

        info!("Soft deleted (archived) workspace '{}' with ID: {}", workspace.name, workspace.id);
        Ok(())
    }
}

impl Handler<ToggleFavoriteWorkspace> for WorkspaceActor {
    type Result = Result<bool, String>;

    fn handle(&mut self, msg: ToggleFavoriteWorkspace, _ctx: &mut Self::Context) -> Self::Result {
        self.ensure_initialized()
            .map_err(|e| format!("Failed to initialize: {}", e))?;

        let workspace = self.workspaces
            .get_mut(&msg.workspace_id)
            .ok_or_else(|| format!("Workspace with ID '{}' not found", msg.workspace_id))?;

        let is_favorite = workspace.toggle_favorite();
        let updated_workspace = workspace.clone();

        // Save to persistent storage
        if let Err(e) = self.save_to_storage() {
            error!("Failed to save workspaces after favorite toggle: {}", e);
            return Err(format!("Failed to persist favorite toggle: {}", e));
        }

        // Notify WebSocket clients
        let change_type = if is_favorite {
            WorkspaceChangeType::Favorited
        } else {
            WorkspaceChangeType::Unfavorited
        };
        self.notify_change(&updated_workspace, change_type);

        info!("Toggled favorite for workspace '{}': {}", updated_workspace.name, is_favorite);
        Ok(is_favorite)
    }
}

impl Handler<ArchiveWorkspace> for WorkspaceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: ArchiveWorkspace, _ctx: &mut Self::Context) -> Self::Result {
        self.ensure_initialized()
            .map_err(|e| format!("Failed to initialize: {}", e))?;

        let workspace = self.workspaces
            .get_mut(&msg.workspace_id)
            .ok_or_else(|| format!("Workspace with ID '{}' not found", msg.workspace_id))?;

        if msg.archive {
            workspace.archive();
        } else {
            workspace.unarchive();
        }

        let updated_workspace = workspace.clone();

        // Save to persistent storage
        if let Err(e) = self.save_to_storage() {
            error!("Failed to save workspaces after archive operation: {}", e);
            return Err(format!("Failed to persist archive operation: {}", e));
        }

        // Notify WebSocket clients
        let change_type = if msg.archive {
            WorkspaceChangeType::Archived
        } else {
            WorkspaceChangeType::Unarchived
        };
        self.notify_change(&updated_workspace, change_type);

        let action = if msg.archive { "Archived" } else { "Unarchived" };
        info!("{} workspace '{}' with ID: {}", action, updated_workspace.name, updated_workspace.id);
        Ok(())
    }
}

impl Handler<GetWorkspaceCount> for WorkspaceActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, msg: GetWorkspaceCount, _ctx: &mut Self::Context) -> Self::Result {
        self.ensure_initialized()
            .map_err(|e| format!("Failed to initialize: {}", e))?;

        if msg.filter.is_none() {
            return Ok(self.workspaces.len());
        }

        let all_workspaces: Vec<&Workspace> = self.workspaces.values().collect();
        let filtered_workspaces = self.apply_filters(all_workspaces, &msg.filter);

        Ok(filtered_workspaces.len())
    }
}

impl Handler<LoadWorkspaces> for WorkspaceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: LoadWorkspaces, _ctx: &mut Self::Context) -> Self::Result {
        self.load_from_storage()
            .map_err(|e| format!("Failed to load workspaces: {}", e))?;

        self.initialized = true;
        info!("Reloaded {} workspaces from storage", self.workspaces.len());
        Ok(())
    }
}

impl Handler<SaveWorkspaces> for WorkspaceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: SaveWorkspaces, _ctx: &mut Self::Context) -> Self::Result {
        self.save_to_storage()
            .map_err(|e| format!("Failed to save workspaces: {}", e))?;

        info!("Saved {} workspaces to storage", self.workspaces.len());
        Ok(())
    }
}

impl Handler<WorkspaceStateChanged> for WorkspaceActor {
    type Result = ();

    fn handle(&mut self, msg: WorkspaceStateChanged, _ctx: &mut Self::Context) {
        // This is mainly for internal coordination - just broadcast the change
        debug!("Broadcasting workspace state change: {:?} for workspace {}", msg.change_type, msg.workspace.id);
        self.notify_change(&msg.workspace, msg.change_type);
    }
}