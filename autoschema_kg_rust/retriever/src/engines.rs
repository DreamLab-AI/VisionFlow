//! Retrieval engines for different search strategies

use crate::query::Query;
use crate::RetrievalConfig;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utils::{Result, UtilsError};

/// Result of a retrieval operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub items: Vec<RetrievedItem>,
    pub total_count: usize,
    pub query_time_ms: u64,
    pub metadata: HashMap<String, String>,
}

/// An individual retrieved item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedItem {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub item_type: ItemType,
    pub metadata: HashMap<String, String>,
}

/// Types of items that can be retrieved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemType {
    Entity,
    Relation,
    Triple,
    Concept,
    Document,
}

/// Trait for retrieval engines
#[async_trait]
pub trait RetrievalEngine: Send + Sync {
    /// Retrieve items based on a query
    async fn retrieve(&self, query: &Query, config: &RetrievalConfig) -> Result<RetrievalResult>;

    /// Index new content for retrieval
    async fn index(&mut self, items: Vec<IndexableItem>) -> Result<()>;

    /// Get engine statistics
    fn get_stats(&self) -> EngineStats;

    /// Get engine name
    fn name(&self) -> &str;
}

/// Item that can be indexed for retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexableItem {
    pub id: String,
    pub content: String,
    pub item_type: ItemType,
    pub metadata: HashMap<String, String>,
    pub embedding: Option<Vec<f32>>,
}

/// Statistics about a retrieval engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub indexed_items: usize,
    pub query_count: u64,
    pub average_query_time_ms: f64,
    pub cache_hit_rate: f32,
}

/// Vector-based retrieval engine
pub struct VectorRetrievalEngine {
    name: String,
    index: vectorstore::VectorIndex,
    items: HashMap<String, IndexableItem>,
    stats: EngineStats,
}

impl VectorRetrievalEngine {
    /// Create a new vector retrieval engine
    pub fn new(name: String, vector_config: vectorstore::VectorConfig) -> Result<Self> {
        let index = vectorstore::VectorIndex::new(vector_config)?;

        Ok(Self {
            name,
            index,
            items: HashMap::new(),
            stats: EngineStats {
                indexed_items: 0,
                query_count: 0,
                average_query_time_ms: 0.0,
                cache_hit_rate: 0.0,
            },
        })
    }

    /// Calculate similarity scores
    fn calculate_similarity(&self, query_embedding: &[f32], item_embedding: &[f32]) -> f32 {
        // Cosine similarity
        let dot_product: f32 = query_embedding
            .iter()
            .zip(item_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = item_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl RetrievalEngine for VectorRetrievalEngine {
    async fn retrieve(&self, query: &Query, config: &RetrievalConfig) -> Result<RetrievalResult> {
        let start_time = std::time::Instant::now();

        // Get query embedding
        let query_embedding = match &query.embedding {
            Some(emb) => emb.clone(),
            None => {
                return Err(UtilsError::Custom(
                    "Query must have embedding for vector retrieval".to_string()
                ));
            }
        };

        // Search in vector index
        let search_results = self.index.search(&query_embedding, config.top_k)?;

        // Convert to retrieved items
        let mut items = Vec::new();
        for result in search_results {
            if let Some(item) = self.items.get(&result.id) {
                if result.score >= config.similarity_threshold {
                    items.push(RetrievedItem {
                        id: item.id.clone(),
                        content: item.content.clone(),
                        score: result.score,
                        item_type: item.item_type.clone(),
                        metadata: item.metadata.clone(),
                    });
                }
            }
        }

        let query_time_ms = start_time.elapsed().as_millis() as u64;

        let mut metadata = HashMap::new();
        metadata.insert("engine".to_string(), self.name.clone());
        metadata.insert("search_type".to_string(), "vector".to_string());

        Ok(RetrievalResult {
            total_count: items.len(),
            items,
            query_time_ms,
            metadata,
        })
    }

    async fn index(&mut self, items: Vec<IndexableItem>) -> Result<()> {
        for item in items {
            // Add to vector index if embedding is available
            if let Some(embedding) = &item.embedding {
                self.index.add_vector(&item.id, embedding)?;
            }

            // Store item
            self.items.insert(item.id.clone(), item);
        }

        self.stats.indexed_items = self.items.len();
        Ok(())
    }

    fn get_stats(&self) -> EngineStats {
        self.stats.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Keyword-based retrieval engine
pub struct KeywordRetrievalEngine {
    name: String,
    items: HashMap<String, IndexableItem>,
    keyword_index: HashMap<String, Vec<String>>, // keyword -> item_ids
    stats: EngineStats,
}

impl KeywordRetrievalEngine {
    /// Create a new keyword retrieval engine
    pub fn new(name: String) -> Self {
        Self {
            name,
            items: HashMap::new(),
            keyword_index: HashMap::new(),
            stats: EngineStats {
                indexed_items: 0,
                query_count: 0,
                average_query_time_ms: 0.0,
                cache_hit_rate: 0.0,
            },
        }
    }

    /// Extract keywords from text
    fn extract_keywords(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2)
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Calculate keyword-based score
    fn calculate_keyword_score(&self, query_keywords: &[String], item_keywords: &[String]) -> f32 {
        let query_set: std::collections::HashSet<_> = query_keywords.iter().collect();
        let item_set: std::collections::HashSet<_> = item_keywords.iter().collect();

        let intersection = query_set.intersection(&item_set).count();
        let union = query_set.union(&item_set).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

#[async_trait]
impl RetrievalEngine for KeywordRetrievalEngine {
    async fn retrieve(&self, query: &Query, config: &RetrievalConfig) -> Result<RetrievalResult> {
        let start_time = std::time::Instant::now();

        // Extract keywords from query
        let query_keywords = self.extract_keywords(&query.text);

        // Find matching items
        let mut candidates = HashMap::new();

        for keyword in &query_keywords {
            if let Some(item_ids) = self.keyword_index.get(keyword) {
                for item_id in item_ids {
                    *candidates.entry(item_id.clone()).or_insert(0usize) += 1;
                }
            }
        }

        // Score and rank candidates
        let mut scored_items = Vec::new();

        for (item_id, keyword_matches) in candidates {
            if let Some(item) = self.items.get(&item_id) {
                let item_keywords = self.extract_keywords(&item.content);
                let score = self.calculate_keyword_score(&query_keywords, &item_keywords);

                if score >= config.similarity_threshold {
                    scored_items.push(RetrievedItem {
                        id: item.id.clone(),
                        content: item.content.clone(),
                        score,
                        item_type: item.item_type.clone(),
                        metadata: item.metadata.clone(),
                    });
                }
            }
        }

        // Sort by score (highest first) and limit results
        scored_items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored_items.truncate(config.top_k);

        let query_time_ms = start_time.elapsed().as_millis() as u64;

        let mut metadata = HashMap::new();
        metadata.insert("engine".to_string(), self.name.clone());
        metadata.insert("search_type".to_string(), "keyword".to_string());

        Ok(RetrievalResult {
            total_count: scored_items.len(),
            items: scored_items,
            query_time_ms,
            metadata,
        })
    }

    async fn index(&mut self, items: Vec<IndexableItem>) -> Result<()> {
        for item in items {
            // Extract keywords and update index
            let keywords = self.extract_keywords(&item.content);
            for keyword in keywords {
                self.keyword_index
                    .entry(keyword)
                    .or_default()
                    .push(item.id.clone());
            }

            // Store item
            self.items.insert(item.id.clone(), item);
        }

        self.stats.indexed_items = self.items.len();
        Ok(())
    }

    fn get_stats(&self) -> EngineStats {
        self.stats.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::QueryBuilder;

    #[tokio::test]
    async fn test_keyword_retrieval_engine() {
        let mut engine = KeywordRetrievalEngine::new("test".to_string());

        // Index some items
        let items = vec![
            IndexableItem {
                id: "1".to_string(),
                content: "artificial intelligence machine learning".to_string(),
                item_type: ItemType::Concept,
                metadata: HashMap::new(),
                embedding: None,
            },
            IndexableItem {
                id: "2".to_string(),
                content: "natural language processing".to_string(),
                item_type: ItemType::Concept,
                metadata: HashMap::new(),
                embedding: None,
            },
        ];

        engine.index(items).await.unwrap();

        // Test retrieval
        let query = QueryBuilder::new("machine learning".to_string()).build();
        let config = RetrievalConfig::default();
        let result = engine.retrieve(&query, &config).await.unwrap();

        assert!(!result.items.is_empty());
        assert!(result.items[0].score > 0.0);
    }

    #[test]
    fn test_keyword_extraction() {
        let engine = KeywordRetrievalEngine::new("test".to_string());
        let keywords = engine.extract_keywords("Hello world! This is a test.");

        assert!(keywords.contains(&"hello".to_string()));
        assert!(keywords.contains(&"world".to_string()));
        assert!(keywords.contains(&"test".to_string()));
    }
}