use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use crate::utils::time;

///
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Json,
    Gexf,
    Graphml,
    Csv,
    Dot,
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::Json => write!(f, "json"),
            ExportFormat::Gexf => write!(f, "gexf"),
            ExportFormat::Graphml => write!(f, "graphml"),
            ExportFormat::Csv => write!(f, "csv"),
            ExportFormat::Dot => write!(f, "dot"),
        }
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    pub format: ExportFormat,
    pub graph_id: Option<String>,
    pub include_metadata: bool,
    pub compress: bool,
    pub custom_attributes: Option<Vec<String>>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResponse {
    pub export_id: String,
    pub format: ExportFormat,
    pub file_size: u64,
    pub compressed: bool,
    pub download_url: String,
    pub expires_at: DateTime<Utc>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedGraph {
    pub id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub creator_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub access_count: u32,
    pub max_access_count: Option<u32>,
    pub is_public: bool,
    pub password_hash: Option<String>,
    pub file_path: String,
    pub file_size: u64,
    pub compressed: bool,
    pub original_format: ExportFormat,
    pub node_count: u32,
    pub edge_count: u32,
    pub metadata: HashMap<String, serde_json::Value>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareRequest {
    pub title: String,
    pub description: Option<String>,
    pub expires_in_hours: Option<u32>,
    pub max_access_count: Option<u32>,
    pub is_public: bool,
    pub password: Option<String>,
    pub graph_id: Option<String>,
    pub export_format: ExportFormat,
    pub include_metadata: bool,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareResponse {
    pub share_id: Uuid,
    pub share_url: String,
    pub qr_code_url: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishRequest {
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub license: Option<String>,
    pub category: String,
    pub is_featured: bool,
    pub graph_id: Option<String>,
    pub export_format: ExportFormat,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishResponse {
    pub publication_id: Uuid,
    pub repository_url: String,
    pub doi: Option<String>,
    pub published_at: DateTime<Utc>,
    pub status: PublicationStatus,
}

///
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PublicationStatus {
    Pending,
    Approved,
    Rejected,
    Published,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStats {
    pub total_exports: u64,
    pub exports_by_format: HashMap<String, u64>,
    pub shared_graphs: u64,
    pub published_graphs: u64,
    pub avg_file_size: f64,
    pub last_export: Option<DateTime<Utc>>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub remaining_exports: u32,
    pub reset_time: DateTime<Utc>,
    pub daily_limit: u32,
    pub hourly_limit: u32,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub algorithm: String,
}

impl SharedGraph {
    
    pub fn new(
        title: String,
        description: Option<String>,
        creator_id: Option<String>,
        file_path: String,
        file_size: u64,
        compressed: bool,
        original_format: ExportFormat,
        node_count: u32,
        edge_count: u32,
    ) -> Self {
        let now = time::now();
        Self {
            id: Uuid::new_v4(),
            title,
            description,
            creator_id,
            created_at: now,
            updated_at: now,
            expires_at: None,
            access_count: 0,
            max_access_count: None,
            is_public: true,
            password_hash: None,
            file_path,
            file_size,
            compressed,
            original_format,
            node_count,
            edge_count,
            metadata: HashMap::new(),
        }
    }

    
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            time::now() > expires_at
        } else {
            false
        }
    }

    
    pub fn access_limit_reached(&self) -> bool {
        if let Some(max_count) = self.max_access_count {
            self.access_count >= max_count
        } else {
            false
        }
    }

    
    pub fn increment_access(&mut self) {
        self.access_count += 1;
        self.updated_at = time::now();
    }

    
    pub fn set_expiration(&mut self, hours: u32) {
        self.expires_at = Some(time::now() + chrono::Duration::hours(hours as i64));
    }

    
    pub fn validate_password(&self, password: &str) -> bool {
        if let Some(hash) = &self.password_hash {
            bcrypt::verify(password, hash).unwrap_or(false)
        } else {
            true 
        }
    }
}

impl Default for ExportRequest {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            graph_id: None,
            include_metadata: true,
            compress: false,
            custom_attributes: None,
        }
    }
}

impl Default for ShareRequest {
    fn default() -> Self {
        Self {
            title: "Shared Graph".to_string(),
            description: None,
            expires_in_hours: Some(24 * 7), 
            max_access_count: None,
            is_public: true,
            password: None,
            graph_id: None,
            export_format: ExportFormat::Json,
            include_metadata: true,
        }
    }
}
