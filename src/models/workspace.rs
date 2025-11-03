//! Workspace model definitions and related structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::collections::HashMap;
use uuid::Uuid;
use validator::Validate;
use crate::utils::time;

///
#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq)]
pub enum WorkspaceType {
    #[serde(rename = "personal")]
    Personal,
    #[serde(rename = "team")]
    Team,
    #[serde(rename = "public")]
    Public,
}

impl Default for WorkspaceType {
    fn default() -> Self {
        WorkspaceType::Personal
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq)]
pub enum WorkspaceStatus {
    #[serde(rename = "active")]
    Active,
    #[serde(rename = "archived")]
    Archived,
}

impl Default for WorkspaceStatus {
    fn default() -> Self {
        WorkspaceStatus::Active
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
pub struct Workspace {
    
    pub id: String,

    
    #[validate(length(
        min = 1,
        max = 100,
        message = "Name must be between 1 and 100 characters"
    ))]
    pub name: String,

    
    #[validate(length(max = 500, message = "Description cannot exceed 500 characters"))]
    pub description: Option<String>,

    
    pub workspace_type: WorkspaceType,

    
    pub status: WorkspaceStatus,

    
    pub member_count: u32,

    
    pub is_favorite: bool,

    
    pub owner_id: Option<String>,

    
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[specta(skip)]
    pub metadata: HashMap<String, serde_json::Value>,

    
    #[specta(type = String)]
    pub created_at: DateTime<Utc>,

    
    #[specta(type = String)]
    pub updated_at: DateTime<Utc>,
}

impl Default for Workspace {
    fn default() -> Self {
        let now = time::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "New Workspace".to_string(),
            description: None,
            workspace_type: WorkspaceType::default(),
            status: WorkspaceStatus::default(),
            member_count: 1,
            is_favorite: false,
            owner_id: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

impl Workspace {
    
    pub fn new(name: String, description: Option<String>, workspace_type: WorkspaceType) -> Self {
        let now = time::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            description,
            workspace_type,
            status: WorkspaceStatus::Active,
            member_count: 1,
            is_favorite: false,
            owner_id: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    
    pub fn update(
        &mut self,
        name: Option<String>,
        description: Option<String>,
        workspace_type: Option<WorkspaceType>,
    ) {
        if let Some(new_name) = name {
            self.name = new_name;
        }
        if let Some(new_description) = description {
            self.description = Some(new_description);
        }
        if let Some(new_type) = workspace_type {
            self.workspace_type = new_type;
        }
        self.updated_at = time::now();
    }

    
    pub fn toggle_favorite(&mut self) -> bool {
        self.is_favorite = !self.is_favorite;
        self.updated_at = time::now();
        self.is_favorite
    }

    
    pub fn archive(&mut self) {
        self.status = WorkspaceStatus::Archived;
        self.updated_at = time::now();
    }

    
    pub fn unarchive(&mut self) {
        self.status = WorkspaceStatus::Active;
        self.updated_at = time::now();
    }

    
    pub fn is_archived(&self) -> bool {
        self.status == WorkspaceStatus::Archived
    }

    
    pub fn set_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
        self.updated_at = time::now();
    }

    
    pub fn remove_metadata(&mut self, key: &str) -> Option<serde_json::Value> {
        let result = self.metadata.remove(key);
        if result.is_some() {
            self.updated_at = time::now();
        }
        result
    }

    
    pub fn set_member_count(&mut self, count: u32) {
        self.member_count = count;
        self.updated_at = time::now();
    }

    
    pub fn set_owner(&mut self, owner_id: String) {
        self.owner_id = Some(owner_id);
        self.updated_at = time::now();
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
pub struct CreateWorkspaceRequest {
    #[validate(length(
        min = 1,
        max = 100,
        message = "Name must be between 1 and 100 characters"
    ))]
    pub name: String,

    #[validate(length(max = 500, message = "Description cannot exceed 500 characters"))]
    pub description: Option<String>,

    pub workspace_type: Option<WorkspaceType>,
    pub owner_id: Option<String>,

    #[specta(skip)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
pub struct UpdateWorkspaceRequest {
    #[validate(length(
        min = 1,
        max = 100,
        message = "Name must be between 1 and 100 characters"
    ))]
    pub name: Option<String>,

    #[validate(length(max = 500, message = "Description cannot exceed 500 characters"))]
    pub description: Option<String>,

    pub workspace_type: Option<WorkspaceType>,

    #[specta(skip)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct WorkspaceResponse {
    pub success: bool,
    pub message: String,
    pub workspace: Option<Workspace>,
}

impl WorkspaceResponse {
    pub fn success(workspace: Workspace, message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            workspace: Some(workspace),
        }
    }

    pub fn success_no_data(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            workspace: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            workspace: None,
        }
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct WorkspaceListResponse {
    pub success: bool,
    pub message: String,
    pub workspaces: Vec<Workspace>,
    pub total_count: usize,
    pub page: usize,
    pub page_size: usize,
}

impl WorkspaceListResponse {
    pub fn success(
        workspaces: Vec<Workspace>,
        total_count: usize,
        page: usize,
        page_size: usize,
    ) -> Self {
        Self {
            success: true,
            message: "Workspaces retrieved successfully".to_string(),
            workspaces,
            total_count,
            page,
            page_size,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            workspaces: Vec::new(),
            total_count: 0,
            page: 0,
            page_size: 0,
        }
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct WorkspaceFilter {
    
    pub status: Option<WorkspaceStatus>,
    
    pub workspace_type: Option<WorkspaceType>,
    
    pub is_favorite: Option<bool>,
    
    pub owner_id: Option<String>,
    
    pub search: Option<String>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub enum WorkspaceSortBy {
    #[serde(rename = "name")]
    Name,
    #[serde(rename = "created_at")]
    CreatedAt,
    #[serde(rename = "updated_at")]
    UpdatedAt,
    #[serde(rename = "member_count")]
    MemberCount,
}

impl Default for WorkspaceSortBy {
    fn default() -> Self {
        WorkspaceSortBy::UpdatedAt
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub enum SortDirection {
    #[serde(rename = "asc")]
    Ascending,
    #[serde(rename = "desc")]
    Descending,
}

impl Default for SortDirection {
    fn default() -> Self {
        SortDirection::Descending
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
pub struct WorkspaceQuery {
    #[validate(range(min = 1, max = 1000, message = "Page size must be between 1 and 1000"))]
    pub page_size: Option<usize>,

    #[validate(range(min = 0, message = "Page must be non-negative"))]
    pub page: Option<usize>,

    pub sort_by: Option<WorkspaceSortBy>,
    pub sort_direction: Option<SortDirection>,
    pub filter: Option<WorkspaceFilter>,
}

impl Default for WorkspaceQuery {
    fn default() -> Self {
        Self {
            page_size: Some(20),
            page: Some(0),
            sort_by: Some(WorkspaceSortBy::default()),
            sort_direction: Some(SortDirection::default()),
            filter: None,
        }
    }
}
