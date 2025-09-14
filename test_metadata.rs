use std::fs::File;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

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
    #[serde(default)]
    pub last_modified: DateTime<Utc>,
    #[serde(default)]
    pub perplexity_link: String,
    #[serde(default)]
    pub last_perplexity_process: Option<DateTime<Utc>>,
    #[serde(default)]
    pub topic_counts: HashMap<String, u32>,
}

type MetadataStore = HashMap<String, Metadata>;

fn main() {
    let metadata_path = "/app/data/metadata/metadata.json";

    match File::open(metadata_path) {
        Ok(file) => {
            println!("Loading metadata from {}", metadata_path);
            match serde_json::from_reader::<_, MetadataStore>(file) {
                Ok(metadata) => {
                    println!("Successfully loaded {} entries", metadata.len());
                    for (key, value) in metadata.iter().take(3) {
                        println!("  Key: {}", key);
                        println!("    file_name: {}", value.file_name);
                        println!("    file_size: {}", value.file_size);
                        println!("    node_size: {}", value.node_size);
                    }
                }
                Err(e) => {
                    println!("Failed to parse metadata: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Failed to open metadata file: {}", e);
        }
    }
}