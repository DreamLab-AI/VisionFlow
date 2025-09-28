//! Advanced ranking and filtering system for search results
//!
//! Sophisticated result ranking with multiple features and learned models

use crate::error::{Result, RetrieverError};
use crate::config::{RankingConfig, RankingModelConfig, RankingFeature, RerankingConfig};
use crate::search::SearchResult;
use crate::query::ProcessedQuery;
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Scored document with ranking features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDocument {
    /// Original search result
    pub result: SearchResult,

    /// Final ranking score
    pub ranking_score: f32,

    /// Feature scores
    pub feature_scores: HashMap<String, f32>,

    /// Ranking explanation
    pub ranking_explanation: String,

    /// Original position before ranking
    pub original_position: usize,

    /// New position after ranking
    pub new_position: usize,
}

/// Ranking strategy enumeration
#[derive(Debug, Clone)]
pub enum RankingStrategy {
    BM25 { k1: f32, b: f32 },
    TfIdf { normalized: bool },
    LearnedRanking { model_path: String },
    Ensemble { strategies: Vec<RankingStrategy> },
}

impl From<RankingModelConfig> for RankingStrategy {
    fn from(config: RankingModelConfig) -> Self {
        match config {
            RankingModelConfig::BM25 { k1, b } => RankingStrategy::BM25 { k1, b },
            RankingModelConfig::TfIdf { normalization } => RankingStrategy::TfIdf { normalized: normalization },
            RankingModelConfig::LearnedRanking { model_path } => RankingStrategy::LearnedRanking { model_path },
            RankingModelConfig::Ensemble { models } => {
                RankingStrategy::Ensemble {
                    strategies: models.into_iter().map(Into::into).collect()
                }
            }
        }
    }
}

/// Feature extractor trait for ranking features
#[async_trait]
pub trait FeatureExtractor: Send + Sync {
    /// Extract feature value for a document given a query
    async fn extract(&self, result: &SearchResult, query: &ProcessedQuery) -> Result<f32>;

    /// Get feature name
    fn name(&self) -> &str;
}

/// Text similarity feature extractor
pub struct TextSimilarityExtractor;

#[async_trait]
impl FeatureExtractor for TextSimilarityExtractor {
    async fn extract(&self, result: &SearchResult, query: &ProcessedQuery) -> Result<f32> {
        // Calculate character-level similarity
        let query_chars: std::collections::HashSet<char> = query.preprocessed.chars().collect();
        let doc_chars: std::collections::HashSet<char> = result.content.chars().collect();

        let intersection = query_chars.intersection(&doc_chars).count();
        let union = query_chars.union(&doc_chars).count();

        if union == 0 {
            Ok(0.0)
        } else {
            Ok(intersection as f32 / union as f32)
        }
    }

    fn name(&self) -> &str {
        "text_similarity"
    }
}

/// Graph distance feature extractor
pub struct GraphDistanceExtractor {
    max_distance: f32,
}

impl GraphDistanceExtractor {
    pub fn new(max_distance: f32) -> Self {
        Self { max_distance }
    }
}

#[async_trait]
impl FeatureExtractor for GraphDistanceExtractor {
    async fn extract(&self, result: &SearchResult, _query: &ProcessedQuery) -> Result<f32> {
        // Extract graph distance from metadata or compute
        let distance = result.metadata
            .get("graph_distance")
            .and_then(|d| d.parse::<f32>().ok())
            .unwrap_or(self.max_distance);

        // Convert distance to similarity (inverse relationship)
        Ok(1.0 - (distance / self.max_distance).min(1.0))
    }

    fn name(&self) -> &str {
        "graph_distance"
    }
}

/// Document popularity feature extractor
pub struct PopularityExtractor;

#[async_trait]
impl FeatureExtractor for PopularityExtractor {
    async fn extract(&self, result: &SearchResult, _query: &ProcessedQuery) -> Result<f32> {
        // Extract popularity score from metadata
        let popularity = result.metadata
            .get("popularity")
            .and_then(|p| p.parse::<f32>().ok())
            .unwrap_or(0.0);

        Ok(popularity.min(1.0))
    }

    fn name(&self) -> &str {
        "popularity"
    }
}

/// Document recency feature extractor
pub struct RecencyExtractor {
    max_age_days: f32,
}

impl RecencyExtractor {
    pub fn new(max_age_days: f32) -> Self {
        Self { max_age_days }
    }
}

#[async_trait]
impl FeatureExtractor for RecencyExtractor {
    async fn extract(&self, result: &SearchResult, _query: &ProcessedQuery) -> Result<f32> {
        // Extract timestamp from metadata
        let timestamp = result.metadata
            .get("timestamp")
            .and_then(|t| t.parse::<i64>().ok())
            .unwrap_or(0);

        let current_time = chrono::Utc::now().timestamp();
        let age_seconds = (current_time - timestamp).max(0) as f32;
        let age_days = age_seconds / (24.0 * 3600.0);

        // Convert age to recency score
        Ok(1.0 - (age_days / self.max_age_days).min(1.0))
    }

    fn name(&self) -> &str {
        "recency"
    }
}

/// Document length feature extractor
pub struct LengthExtractor {
    optimal_length: f32,
}

impl LengthExtractor {
    pub fn new(optimal_length: f32) -> Self {
        Self { optimal_length }
    }
}

#[async_trait]
impl FeatureExtractor for LengthExtractor {
    async fn extract(&self, result: &SearchResult, _query: &ProcessedQuery) -> Result<f32> {
        let doc_length = result.content.len() as f32;

        // Score based on how close to optimal length
        let length_ratio = doc_length / self.optimal_length;

        // Gaussian-like scoring: peak at optimal length
        Ok((-0.5 * (length_ratio - 1.0).powi(2)).exp())
    }

    fn name(&self) -> &str {
        "length"
    }
}

/// Query term coverage feature extractor
pub struct QueryTermCoverageExtractor;

#[async_trait]
impl FeatureExtractor for QueryTermCoverageExtractor {
    async fn extract(&self, result: &SearchResult, query: &ProcessedQuery) -> Result<f32> {
        let query_terms: std::collections::HashSet<&str> = query.preprocessed.split_whitespace().collect();
        let doc_words: std::collections::HashSet<&str> = result.content.split_whitespace().collect();

        if query_terms.is_empty() {
            return Ok(0.0);
        }

        let covered_terms = query_terms.intersection(&doc_words).count();
        Ok(covered_terms as f32 / query_terms.len() as f32)
    }

    fn name(&self) -> &str {
        "query_term_coverage"
    }
}

/// Entity density feature extractor
pub struct EntityDensityExtractor;

#[async_trait]
impl FeatureExtractor for EntityDensityExtractor {
    async fn extract(&self, result: &SearchResult, _query: &ProcessedQuery) -> Result<f32> {
        // Count capitalized words as potential entities
        let words: Vec<&str> = result.content.split_whitespace().collect();
        let entity_count = words.iter()
            .filter(|word| word.chars().next().map_or(false, |c| c.is_uppercase()))
            .count();

        if words.is_empty() {
            Ok(0.0)
        } else {
            Ok(entity_count as f32 / words.len() as f32)
        }
    }

    fn name(&self) -> &str {
        "entity_density"
    }
}

/// BM25 ranking model
pub struct BM25Ranker {
    k1: f32,
    b: f32,
    avg_doc_length: f32,
    term_frequencies: HashMap<String, HashMap<String, f32>>,
    document_frequencies: HashMap<String, usize>,
    total_documents: usize,
}

impl BM25Ranker {
    /// Create new BM25 ranker
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            avg_doc_length: 0.0,
            term_frequencies: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
        }
    }

    /// Add document to BM25 index
    pub fn add_document(&mut self, doc_id: String, content: &str) {
        let tokens: Vec<&str> = content.split_whitespace().collect();
        let doc_length = tokens.len() as f32;

        // Update average document length
        let total_length = self.avg_doc_length * self.total_documents as f32 + doc_length;
        self.total_documents += 1;
        self.avg_doc_length = total_length / self.total_documents as f32;

        // Calculate term frequencies
        let mut tf_map = HashMap::new();
        for token in tokens {
            let term = token.to_lowercase();
            *tf_map.entry(term.clone()).or_insert(0.0) += 1.0;

            // Update document frequency
            if tf_map.get(&term).unwrap_or(&0.0) == &1.0 {
                *self.document_frequencies.entry(term).or_insert(0) += 1;
            }
        }

        self.term_frequencies.insert(doc_id, tf_map);
    }

    /// Calculate BM25 score for a document given a query
    pub fn calculate_score(&self, doc_id: &str, query_terms: &[&str]) -> f32 {
        let doc_tf = match self.term_frequencies.get(doc_id) {
            Some(tf) => tf,
            None => return 0.0,
        };

        let doc_length = doc_tf.values().sum::<f32>();
        let mut score = 0.0;

        for term in query_terms {
            let term_lower = term.to_lowercase();
            let tf = doc_tf.get(&term_lower).unwrap_or(&0.0);
            let df = self.document_frequencies.get(&term_lower).unwrap_or(&0);

            if *df > 0 {
                let idf = ((self.total_documents as f32 - *df as f32 + 0.5) / (*df as f32 + 0.5)).ln();
                let tf_component = (tf * (self.k1 + 1.0)) /
                    (tf + self.k1 * (1.0 - self.b + self.b * (doc_length / self.avg_doc_length)));

                score += idf * tf_component;
            }
        }

        score
    }
}

/// TF-IDF ranking model
pub struct TfIdfRanker {
    normalized: bool,
    term_frequencies: HashMap<String, HashMap<String, f32>>,
    document_frequencies: HashMap<String, usize>,
    total_documents: usize,
}

impl TfIdfRanker {
    /// Create new TF-IDF ranker
    pub fn new(normalized: bool) -> Self {
        Self {
            normalized,
            term_frequencies: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
        }
    }

    /// Add document to TF-IDF index
    pub fn add_document(&mut self, doc_id: String, content: &str) {
        let tokens: Vec<&str> = content.split_whitespace().collect();
        let mut tf_map = HashMap::new();

        for token in tokens {
            let term = token.to_lowercase();
            *tf_map.entry(term.clone()).or_insert(0.0) += 1.0;
        }

        // Normalize term frequencies if requested
        if self.normalized {
            let max_tf = tf_map.values().fold(0.0f32, |max, &tf| max.max(tf));
            if max_tf > 0.0 {
                for tf in tf_map.values_mut() {
                    *tf /= max_tf;
                }
            }
        }

        // Update document frequencies
        for term in tf_map.keys() {
            if !self.term_frequencies.contains_key(&doc_id) ||
               !self.term_frequencies[&doc_id].contains_key(term) {
                *self.document_frequencies.entry(term.clone()).or_insert(0) += 1;
            }
        }

        self.term_frequencies.insert(doc_id, tf_map);
        self.total_documents += 1;
    }

    /// Calculate TF-IDF score for a document given a query
    pub fn calculate_score(&self, doc_id: &str, query_terms: &[&str]) -> f32 {
        let doc_tf = match self.term_frequencies.get(doc_id) {
            Some(tf) => tf,
            None => return 0.0,
        };

        let mut score = 0.0;

        for term in query_terms {
            let term_lower = term.to_lowercase();
            let tf = doc_tf.get(&term_lower).unwrap_or(&0.0);
            let df = self.document_frequencies.get(&term_lower).unwrap_or(&0);

            if *df > 0 {
                let idf = (self.total_documents as f32 / *df as f32).ln();
                score += tf * idf;
            }
        }

        score
    }
}

/// Main ranking model orchestrating features and strategies
pub struct RankingModel {
    strategy: RankingStrategy,
    feature_extractors: Vec<Box<dyn FeatureExtractor>>,
    feature_weights: HashMap<String, f32>,
    config: RankingConfig,
}

impl RankingModel {
    /// Create new ranking model
    pub fn new(config: RankingConfig) -> Result<Self> {
        let strategy = RankingStrategy::from(config.model.clone());
        let mut feature_extractors: Vec<Box<dyn FeatureExtractor>> = Vec::new();
        let mut feature_weights = HashMap::new();

        // Initialize feature extractors based on configuration
        for feature in &config.features {
            match feature {
                RankingFeature::TextSimilarity => {
                    feature_extractors.push(Box::new(TextSimilarityExtractor));
                    feature_weights.insert("text_similarity".to_string(), 1.0);
                }
                RankingFeature::GraphDistance => {
                    feature_extractors.push(Box::new(GraphDistanceExtractor::new(10.0)));
                    feature_weights.insert("graph_distance".to_string(), 0.8);
                }
                RankingFeature::Popularity => {
                    feature_extractors.push(Box::new(PopularityExtractor));
                    feature_weights.insert("popularity".to_string(), 0.6);
                }
                RankingFeature::Recency => {
                    feature_extractors.push(Box::new(RecencyExtractor::new(365.0)));
                    feature_weights.insert("recency".to_string(), 0.7);
                }
                RankingFeature::Length => {
                    feature_extractors.push(Box::new(LengthExtractor::new(1000.0)));
                    feature_weights.insert("length".to_string(), 0.5);
                }
                RankingFeature::QueryTermCoverage => {
                    feature_extractors.push(Box::new(QueryTermCoverageExtractor));
                    feature_weights.insert("query_term_coverage".to_string(), 1.2);
                }
                RankingFeature::EntityDensity => {
                    feature_extractors.push(Box::new(EntityDensityExtractor));
                    feature_weights.insert("entity_density".to_string(), 0.4);
                }
            }
        }

        Ok(Self {
            strategy,
            feature_extractors,
            feature_weights,
            config,
        })
    }

    /// Rank search results
    pub async fn rank(&self, results: Vec<SearchResult>, query: &ProcessedQuery) -> Result<Vec<ScoredDocument>> {
        // Extract features for all results in parallel
        let scored_results: Result<Vec<_>> = stream::iter(results.into_iter().enumerate())
            .map(|(original_position, result)| async move {
                let feature_scores = self.extract_features(&result, query).await?;
                let ranking_score = self.calculate_ranking_score(&feature_scores, &result, query).await?;

                Ok(ScoredDocument {
                    result,
                    ranking_score,
                    feature_scores,
                    ranking_explanation: self.generate_explanation(&feature_scores),
                    original_position,
                    new_position: 0, // Will be set after sorting
                })
            })
            .buffer_unordered(10) // Process up to 10 documents concurrently
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();

        let mut scored_results = scored_results?;

        // Sort by ranking score (descending)
        scored_results.sort_by(|a, b| b.ranking_score.partial_cmp(&a.ranking_score).unwrap_or(std::cmp::Ordering::Equal));

        // Update new positions
        for (new_position, scored_result) in scored_results.iter_mut().enumerate() {
            scored_result.new_position = new_position;
        }

        Ok(scored_results)
    }

    async fn extract_features(&self, result: &SearchResult, query: &ProcessedQuery) -> Result<HashMap<String, f32>> {
        let mut features = HashMap::new();

        // Extract all features in parallel
        let feature_results: Result<Vec<_>> = stream::iter(&self.feature_extractors)
            .map(|extractor| async move {
                let score = extractor.extract(result, query).await?;
                Ok((extractor.name().to_string(), score))
            })
            .buffer_unordered(self.feature_extractors.len())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();

        for (feature_name, score) in feature_results? {
            features.insert(feature_name, score);
        }

        Ok(features)
    }

    async fn calculate_ranking_score(&self, features: &HashMap<String, f32>, result: &SearchResult, query: &ProcessedQuery) -> Result<f32> {
        match &self.strategy {
            RankingStrategy::BM25 { k1, b } => {
                // Use BM25 as base score, enhance with features
                let query_terms: Vec<&str> = query.preprocessed.split_whitespace().collect();
                let mut ranker = BM25Ranker::new(*k1, *b);
                ranker.add_document(result.id.clone(), &result.content);
                let base_score = ranker.calculate_score(&result.id, &query_terms);

                Ok(self.combine_with_features(base_score, features))
            }
            RankingStrategy::TfIdf { normalized } => {
                // Use TF-IDF as base score, enhance with features
                let query_terms: Vec<&str> = query.preprocessed.split_whitespace().collect();
                let mut ranker = TfIdfRanker::new(*normalized);
                ranker.add_document(result.id.clone(), &result.content);
                let base_score = ranker.calculate_score(&result.id, &query_terms);

                Ok(self.combine_with_features(base_score, features))
            }
            RankingStrategy::LearnedRanking { model_path: _ } => {
                // Placeholder for learned ranking model
                // In practice, you'd load and use a trained model
                Ok(self.feature_based_score(features))
            }
            RankingStrategy::Ensemble { strategies } => {
                // Combine multiple ranking strategies
                let mut ensemble_score = 0.0;
                let strategy_weight = 1.0 / strategies.len() as f32;

                for strategy in strategies {
                    let strategy_model = RankingModel {
                        strategy: strategy.clone(),
                        feature_extractors: vec![], // Reuse current extractors
                        feature_weights: self.feature_weights.clone(),
                        config: self.config.clone(),
                    };

                    let strategy_score = strategy_model.calculate_ranking_score(features, result, query).await?;
                    ensemble_score += strategy_score * strategy_weight;
                }

                Ok(ensemble_score)
            }
        }
    }

    fn combine_with_features(&self, base_score: f32, features: &HashMap<String, f32>) -> f32 {
        let feature_score = self.feature_based_score(features);

        // Weighted combination of base score and feature score
        0.6 * base_score + 0.4 * feature_score
    }

    fn feature_based_score(&self, features: &HashMap<String, f32>) -> f32 {
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;

        for (feature_name, &feature_value) in features {
            if let Some(&weight) = self.feature_weights.get(feature_name) {
                weighted_score += feature_value * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        }
    }

    fn generate_explanation(&self, features: &HashMap<String, f32>) -> String {
        let mut explanations = Vec::new();

        for (feature_name, &score) in features {
            if score > 0.1 { // Only include significant features
                explanations.push(format!("{}: {:.3}", feature_name, score));
            }
        }

        if explanations.is_empty() {
            "No significant features".to_string()
        } else {
            explanations.join(", ")
        }
    }
}

/// Re-ranking engine for final result refinement
pub struct ReRanker {
    config: RerankingConfig,
    cross_encoder: Option<CrossEncoder>,
}

struct CrossEncoder {
    model_path: String,
}

impl CrossEncoder {
    fn new(model_path: String) -> Self {
        Self { model_path }
    }

    async fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        // Placeholder for cross-encoder model scoring
        // In practice, you'd use a transformer-based cross-encoder
        let query_len = query.len() as f32;
        let doc_len = document.len() as f32;
        let common_words = query.split_whitespace()
            .filter(|word| document.contains(word))
            .count() as f32;

        Ok((common_words / (query_len + doc_len).sqrt()).min(1.0))
    }
}

impl ReRanker {
    /// Create new re-ranker
    pub fn new(config: RerankingConfig) -> Self {
        let cross_encoder = if config.enabled {
            Some(CrossEncoder::new(config.model.clone()))
        } else {
            None
        };

        Self {
            config,
            cross_encoder,
        }
    }

    /// Re-rank top results using cross-encoder
    pub async fn rerank(&self, scored_results: Vec<ScoredDocument>, query: &ProcessedQuery) -> Result<Vec<ScoredDocument>> {
        if !self.config.enabled || self.cross_encoder.is_none() {
            return Ok(scored_results);
        }

        let cross_encoder = self.cross_encoder.as_ref().unwrap();
        let top_k = self.config.top_k.min(scored_results.len());

        // Split into top-k for re-ranking and rest
        let (mut top_results, rest_results) = {
            let mut all = scored_results;
            let rest = all.split_off(top_k);
            (all, rest)
        };

        // Re-rank top-k results
        let reranked_scores: Result<Vec<_>> = stream::iter(&top_results)
            .map(|scored_doc| async move {
                let cross_score = cross_encoder.score_pair(&query.original, &scored_doc.result.content).await?;
                Ok(cross_score)
            })
            .buffer_unordered(top_k)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();

        let reranked_scores = reranked_scores?;

        // Update scores with cross-encoder results
        for (scored_doc, cross_score) in top_results.iter_mut().zip(reranked_scores.iter()) {
            if self.config.context_aware {
                // Combine original score with cross-encoder score
                scored_doc.ranking_score = 0.7 * scored_doc.ranking_score + 0.3 * cross_score;
            } else {
                scored_doc.ranking_score = *cross_score;
            }

            scored_doc.ranking_explanation = format!("{}, cross-encoder: {:.3}",
                scored_doc.ranking_explanation, cross_score);
        }

        // Re-sort top-k results
        top_results.sort_by(|a, b| b.ranking_score.partial_cmp(&a.ranking_score).unwrap_or(std::cmp::Ordering::Equal));

        // Update positions
        for (new_position, scored_result) in top_results.iter_mut().enumerate() {
            scored_result.new_position = new_position;
        }

        // Combine with rest
        let mut final_results = top_results;
        final_results.extend(rest_results);

        Ok(final_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::ResultType;

    fn create_test_result(id: &str, content: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score,
            content: content.to_string(),
            result_type: ResultType::Document,
            metadata: HashMap::new(),
            source_strategy: "test".to_string(),
            strategy_scores: HashMap::new(),
            explanation: "test".to_string(),
            snippet: None,
        }
    }

    fn create_test_query() -> ProcessedQuery {
        ProcessedQuery {
            original: "machine learning".to_string(),
            preprocessed: "machine learning".to_string(),
            expanded: vec![],
            rewritten: vec![],
            features: crate::query::QueryFeatures {
                token_count: 2,
                char_count: 16,
                entities: vec![],
                key_phrases: vec![],
                query_type: crate::query::QueryType::Factual,
                intent: crate::query::QueryIntent::Information,
                complexity: 0.5,
                ambiguity: 0.3,
            },
            metadata: crate::query::QueryMetadata {
                timestamp: 0,
                processing_time_ms: 10,
                language: "en".to_string(),
                steps: vec![],
                warnings: vec![],
            },
        }
    }

    #[tokio::test]
    async fn test_text_similarity_extractor() {
        let extractor = TextSimilarityExtractor;
        let result = create_test_result("doc1", "machine learning algorithms", 0.8);
        let query = create_test_query();

        let similarity = extractor.extract(&result, &query).await.unwrap();
        assert!(similarity > 0.0);
    }

    #[tokio::test]
    async fn test_query_term_coverage_extractor() {
        let extractor = QueryTermCoverageExtractor;
        let result = create_test_result("doc1", "this document discusses machine learning in detail", 0.8);
        let query = create_test_query();

        let coverage = extractor.extract(&result, &query).await.unwrap();
        assert_eq!(coverage, 1.0); // Both "machine" and "learning" are covered
    }

    #[test]
    fn test_bm25_ranker() {
        let mut ranker = BM25Ranker::new(1.2, 0.75);
        ranker.add_document("doc1".to_string(), "machine learning is a subset of artificial intelligence");
        ranker.add_document("doc2".to_string(), "deep learning uses neural networks for machine learning");

        let query_terms = vec!["machine", "learning"];
        let score1 = ranker.calculate_score("doc1", &query_terms);
        let score2 = ranker.calculate_score("doc2", &query_terms);

        assert!(score1 > 0.0);
        assert!(score2 > 0.0);
        assert!(score2 > score1); // doc2 has more term occurrences
    }

    #[test]
    fn test_tfidf_ranker() {
        let mut ranker = TfIdfRanker::new(true);
        ranker.add_document("doc1".to_string(), "machine learning algorithms");
        ranker.add_document("doc2".to_string(), "machine learning neural networks");

        let query_terms = vec!["machine", "learning"];
        let score1 = ranker.calculate_score("doc1", &query_terms);
        let score2 = ranker.calculate_score("doc2", &query_terms);

        assert!(score1 > 0.0);
        assert!(score2 > 0.0);
    }

    #[tokio::test]
    async fn test_ranking_model() {
        let config = RankingConfig::default();
        let ranking_model = RankingModel::new(config).unwrap();

        let results = vec![
            create_test_result("doc1", "machine learning is important", 0.8),
            create_test_result("doc2", "artificial intelligence and machine learning", 0.7),
        ];

        let query = create_test_query();
        let scored_results = ranking_model.rank(results, &query).await.unwrap();

        assert_eq!(scored_results.len(), 2);
        assert!(scored_results[0].ranking_score >= scored_results[1].ranking_score);
    }
}
"