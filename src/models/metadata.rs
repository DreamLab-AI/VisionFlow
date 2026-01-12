use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Metadata {
    #[serde(default)]
    pub file_name: String,
    #[serde(default)]
    pub file_size: usize,
    #[serde(default)]
    pub node_size: f64,
    #[serde(default)]
    pub hyperlink_count: usize,
    #[serde(default)]
    pub sha1: String,
    #[serde(default = "default_node_id")]
    pub node_id: String,
    #[serde(default = "Utc::now")]
    pub last_modified: DateTime<Utc>,
    #[serde(default)]
    pub last_content_change: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_commit: Option<DateTime<Utc>>,
    #[serde(default)]
    pub change_count: Option<u32>,
    #[serde(default)]
    pub file_blob_sha: Option<String>,
    #[serde(default)]
    pub perplexity_link: String,
    #[serde(default)]
    pub last_perplexity_process: Option<DateTime<Utc>>,
    #[serde(default)]
    pub topic_counts: HashMap<String, usize>,
    // Ontology fields from new header format
    #[serde(default)]
    pub term_id: Option<String>,
    #[serde(default)]
    pub preferred_term: Option<String>,
    #[serde(default)]
    pub source_domain: Option<String>,
    #[serde(default)]
    pub ontology_status: Option<String>,
    #[serde(default)]
    pub owl_class: Option<String>,
    #[serde(default)]
    pub owl_physicality: Option<String>,
    #[serde(default)]
    pub owl_role: Option<String>,
    #[serde(default)]
    pub quality_score: Option<f64>,
    #[serde(default)]
    pub authority_score: Option<f64>,
    #[serde(default)]
    pub belongs_to_domain: Vec<String>,
    #[serde(default)]
    pub maturity: Option<String>,
    #[serde(default)]
    pub is_subclass_of: Vec<String>,
    #[serde(default)]
    pub definition: Option<String>,
}

// Default function for node_id to ensure backward compatibility
fn default_node_id() -> String {
    
    "0".to_string()
}

pub type MetadataStore = HashMap<String, Metadata>;

pub type FileMetadata = Metadata;

// Implement helper methods directly on HashMap<String, Metadata>
pub trait MetadataOps {
    fn validate_files(&self, markdown_dir: &str) -> bool;
    fn get_max_node_id(&self) -> u32;
}

impl MetadataOps for MetadataStore {
    fn get_max_node_id(&self) -> u32 {
        
        self.values()
            .map(|m| m.node_id.parse::<u32>().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }

    fn validate_files(&self, markdown_dir: &str) -> bool {
        if self.is_empty() {
            return false;
        }

        
        for filename in self.keys() {
            let file_path = format!("{}/{}", markdown_dir, filename);
            if !std::path::Path::new(&file_path).exists() {
                return false;
            }
        }

        true
    }
}
