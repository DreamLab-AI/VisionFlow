//! Search functionality and similarity metrics

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, VectorError};

/// Similarity metrics for vector comparison
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
    Hamming,
    Jaccard,
}

/// Search query with filters and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
    pub k: usize,
    pub filters: Option<QueryFilter>,
    pub threshold: Option<f32>,
    pub include_vectors: bool,
    pub include_metadata: bool,
}

/// Query filters for metadata-based filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFilter {
    pub must: Option<Vec<FilterCondition>>,
    pub must_not: Option<Vec<FilterCondition>>,
    pub should: Option<Vec<FilterCondition>>,
    pub range: Option<HashMap<String, RangeFilter>>,
}

/// Individual filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

/// Filter operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    In,
    NotIn,
    Contains,
    StartsWith,
    EndsWith,
    Regex,
}

/// Range filter for numeric values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeFilter {
    pub gte: Option<f64>,  // Greater than or equal
    pub gt: Option<f64>,   // Greater than
    pub lte: Option<f64>,  // Less than or equal
    pub lt: Option<f64>,   // Less than
}

/// Search result from vector queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    pub vector: Option<Vec<f32>>,
}

/// Batch search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSearchResult {
    pub results: Vec<Vec<SearchResult>>,
    pub total_time_ms: u64,
    pub per_query_time_ms: Vec<u64>,
}

/// Search engine that handles similarity computations and filtering
pub struct SearchEngine {
    metric: SimilarityMetric,
}

impl SearchEngine {
    pub fn new(metric: SimilarityMetric) -> Self {
        Self { metric }
    }

    /// Compute similarity between two vectors
    pub fn compute_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VectorError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let similarity = match self.metric {
            SimilarityMetric::Cosine => self.cosine_similarity(a, b)?,
            SimilarityMetric::Euclidean => self.euclidean_distance(a, b)?,
            SimilarityMetric::DotProduct => self.dot_product(a, b)?,
            SimilarityMetric::Manhattan => self.manhattan_distance(a, b)?,
            SimilarityMetric::Hamming => self.hamming_distance(a, b)?,
            SimilarityMetric::Jaccard => self.jaccard_similarity(a, b)?,
        };

        Ok(similarity)
    }

    /// Compute cosine similarity between vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Compute Euclidean distance (converted to similarity)
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let distance: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();

        // Convert distance to similarity (0-1 range)
        Ok(1.0 / (1.0 + distance))
    }

    /// Compute dot product
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    /// Compute Manhattan distance (converted to similarity)
    fn manhattan_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let distance: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        Ok(1.0 / (1.0 + distance))
    }

    /// Compute Hamming distance for binary vectors
    fn hamming_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let different_bits: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| if (x - y).abs() > 0.5 { 1.0 } else { 0.0 })
            .sum();

        let similarity = 1.0 - (different_bits / a.len() as f32);
        Ok(similarity)
    }

    /// Compute Jaccard similarity for binary vectors
    fn jaccard_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let mut intersection = 0.0;
        let mut union = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            let x_bool = *x > 0.5;
            let y_bool = *y > 0.5;

            if x_bool && y_bool {
                intersection += 1.0;
            }
            if x_bool || y_bool {
                union += 1.0;
            }
        }

        if union == 0.0 {
            Ok(1.0) // Both vectors are all zeros
        } else {
            Ok(intersection / union)
        }
    }

    /// Batch compute similarities between query and multiple vectors
    pub fn batch_similarity(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        let mut similarities = Vec::with_capacity(vectors.len());

        for vector in vectors {
            let similarity = self.compute_similarity(query, vector)?;
            similarities.push(similarity);
        }

        Ok(similarities)
    }

    /// Apply metadata filters to search results
    pub fn apply_filters(
        &self,
        results: Vec<SearchResult>,
        filter: &QueryFilter,
    ) -> Result<Vec<SearchResult>> {
        let mut filtered_results = results;

        // Apply must conditions (AND logic)
        if let Some(must_conditions) = &filter.must {
            filtered_results.retain(|result| {
                must_conditions.iter().all(|condition| {
                    self.evaluate_condition(&result.metadata, condition).unwrap_or(false)
                })
            });
        }

        // Apply must_not conditions (NOT logic)
        if let Some(must_not_conditions) = &filter.must_not {
            filtered_results.retain(|result| {
                !must_not_conditions.iter().any(|condition| {
                    self.evaluate_condition(&result.metadata, condition).unwrap_or(false)
                })
            });
        }

        // Apply should conditions (OR logic)
        if let Some(should_conditions) = &filter.should {
            if !should_conditions.is_empty() {
                filtered_results.retain(|result| {
                    should_conditions.iter().any(|condition| {
                        self.evaluate_condition(&result.metadata, condition).unwrap_or(false)
                    })
                });
            }
        }

        // Apply range filters
        if let Some(range_filters) = &filter.range {
            for (field, range_filter) in range_filters {
                filtered_results.retain(|result| {
                    self.evaluate_range_filter(&result.metadata, field, range_filter)
                        .unwrap_or(false)
                });
            }
        }

        Ok(filtered_results)
    }

    /// Evaluate a single filter condition
    fn evaluate_condition(
        &self,
        metadata: &serde_json::Value,
        condition: &FilterCondition,
    ) -> Result<bool> {
        let field_value = metadata.get(&condition.field);

        match &condition.operator {
            FilterOperator::Equals => {
                Ok(field_value == Some(&condition.value))
            }
            FilterOperator::NotEquals => {
                Ok(field_value != Some(&condition.value))
            }
            FilterOperator::In => {
                if let Some(array) = condition.value.as_array() {
                    Ok(field_value.map_or(false, |v| array.contains(v)))
                } else {
                    Ok(false)
                }
            }
            FilterOperator::NotIn => {
                if let Some(array) = condition.value.as_array() {
                    Ok(field_value.map_or(true, |v| !array.contains(v)))
                } else {
                    Ok(true)
                }
            }
            FilterOperator::Contains => {
                if let (Some(field_str), Some(search_str)) = (
                    field_value.and_then(|v| v.as_str()),
                    condition.value.as_str(),
                ) {
                    Ok(field_str.contains(search_str))
                } else {
                    Ok(false)
                }
            }
            FilterOperator::StartsWith => {
                if let (Some(field_str), Some(search_str)) = (
                    field_value.and_then(|v| v.as_str()),
                    condition.value.as_str(),
                ) {
                    Ok(field_str.starts_with(search_str))
                } else {
                    Ok(false)
                }
            }
            FilterOperator::EndsWith => {
                if let (Some(field_str), Some(search_str)) = (
                    field_value.and_then(|v| v.as_str()),
                    condition.value.as_str(),
                ) {
                    Ok(field_str.ends_with(search_str))
                } else {
                    Ok(false)
                }
            }
            FilterOperator::Regex => {
                if let (Some(field_str), Some(pattern)) = (
                    field_value.and_then(|v| v.as_str()),
                    condition.value.as_str(),
                ) {
                    match regex::Regex::new(pattern) {
                        Ok(re) => Ok(re.is_match(field_str)),
                        Err(_) => Ok(false),
                    }
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Evaluate a range filter
    fn evaluate_range_filter(
        &self,
        metadata: &serde_json::Value,
        field: &str,
        range_filter: &RangeFilter,
    ) -> Result<bool> {
        if let Some(field_value) = metadata.get(field).and_then(|v| v.as_f64()) {
            let mut passes = true;

            if let Some(gte) = range_filter.gte {
                passes &= field_value >= gte;
            }
            if let Some(gt) = range_filter.gt {
                passes &= field_value > gt;
            }
            if let Some(lte) = range_filter.lte {
                passes &= field_value <= lte;
            }
            if let Some(lt) = range_filter.lt {
                passes &= field_value < lt;
            }

            Ok(passes)
        } else {
            Ok(false)
        }
    }

    /// Rank and sort search results by score
    pub fn rank_results(&self, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Apply score threshold filtering
    pub fn apply_threshold(&self, results: Vec<SearchResult>, threshold: f32) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter(|result| result.score >= threshold)
            .collect()
    }

    /// Compute diverse results using Maximum Marginal Relevance (MMR)
    pub fn diversify_results(
        &self,
        results: Vec<SearchResult>,
        query: &[f32],
        lambda: f32, // Balance between relevance and diversity (0.0 = pure diversity, 1.0 = pure relevance)
        max_results: usize,
    ) -> Result<Vec<SearchResult>> {
        if results.is_empty() || max_results == 0 {
            return Ok(Vec::new());
        }

        let mut selected = Vec::new();
        let mut remaining = results;

        // Always select the most relevant result first
        if let Some(first) = remaining.first() {
            selected.push(first.clone());
            remaining.remove(0);
        }

        while selected.len() < max_results && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (i, candidate) in remaining.iter().enumerate() {
                if let Some(candidate_vector) = &candidate.vector {
                    // Relevance score (similarity to query)
                    let relevance = self.compute_similarity(query, candidate_vector)?;

                    // Diversity score (minimum similarity to already selected items)
                    let mut max_similarity = 0.0;
                    for selected_result in &selected {
                        if let Some(selected_vector) = &selected_result.vector {
                            let similarity = self.compute_similarity(candidate_vector, selected_vector)?;
                            max_similarity = max_similarity.max(similarity);
                        }
                    }

                    // MMR score
                    let mmr_score = lambda * relevance - (1.0 - lambda) * max_similarity;

                    if mmr_score > best_score {
                        best_score = mmr_score;
                        best_idx = i;
                    }
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        Ok(selected)
    }

    /// Perform re-ranking using cross-encoder or other advanced methods
    pub async fn rerank_results(
        &self,
        query: &str,
        results: Vec<SearchResult>,
        rerank_method: RerankMethod,
    ) -> Result<Vec<SearchResult>> {
        match rerank_method {
            RerankMethod::None => Ok(results),
            RerankMethod::ScoreBoost { boost_factors } => {
                self.apply_score_boost(results, boost_factors)
            }
            RerankMethod::FieldBoost { field_boosts } => {
                self.apply_field_boost(results, field_boosts)
            }
            RerankMethod::CustomFunction { function } => {
                self.apply_custom_rerank(results, function).await
            }
        }
    }

    fn apply_score_boost(
        &self,
        mut results: Vec<SearchResult>,
        boost_factors: HashMap<String, f32>,
    ) -> Result<Vec<SearchResult>> {
        for result in &mut results {
            if let Some(id_boost) = boost_factors.get(&result.id) {
                result.score *= id_boost;
            }
        }
        Ok(self.rank_results(results))
    }

    fn apply_field_boost(
        &self,
        mut results: Vec<SearchResult>,
        field_boosts: HashMap<String, f32>,
    ) -> Result<Vec<SearchResult>> {
        for result in &mut results {
            let mut total_boost = 1.0;

            for (field, boost) in &field_boosts {
                if let Some(field_value) = result.metadata.get(field) {
                    if let Some(num_value) = field_value.as_f64() {
                        total_boost *= boost * num_value as f32;
                    } else if field_value.as_bool().unwrap_or(false) {
                        total_boost *= boost;
                    }
                }
            }

            result.score *= total_boost;
        }

        Ok(self.rank_results(results))
    }

    async fn apply_custom_rerank(
        &self,
        results: Vec<SearchResult>,
        _function: Box<dyn Fn(&SearchResult) -> f32 + Send + Sync>,
    ) -> Result<Vec<SearchResult>> {
        // Custom reranking function application
        // This would apply the custom function to each result
        Ok(results)
    }
}

/// Re-ranking methods for improving search results
#[derive(Debug)]
pub enum RerankMethod {
    None,
    ScoreBoost {
        boost_factors: HashMap<String, f32>,
    },
    FieldBoost {
        field_boosts: HashMap<String, f32>,
    },
    CustomFunction {
        function: Box<dyn Fn(&SearchResult) -> f32 + Send + Sync>,
    },
}

/// Search statistics and metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchStats {
    pub total_queries: u64,
    pub avg_query_time_ms: f64,
    pub cache_hit_rate: f64,
    pub result_distribution: HashMap<String, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let engine = SearchEngine::new(SimilarityMetric::Cosine);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = engine.cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        let similarity = engine.cosine_similarity(&a, &c).unwrap();
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let engine = SearchEngine::new(SimilarityMetric::Euclidean);

        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let similarity = engine.euclidean_distance(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_filter_evaluation() {
        let engine = SearchEngine::new(SimilarityMetric::Cosine);
        let metadata = serde_json::json!({"category": "test", "score": 0.95});

        let condition = FilterCondition {
            field: "category".to_string(),
            operator: FilterOperator::Equals,
            value: serde_json::Value::String("test".to_string()),
        };

        let result = engine.evaluate_condition(&metadata, &condition).unwrap();
        assert!(result);
    }

    #[test]
    fn test_range_filter() {
        let engine = SearchEngine::new(SimilarityMetric::Cosine);
        let metadata = serde_json::json!({"score": 0.75});

        let range_filter = RangeFilter {
            gte: Some(0.5),
            lte: Some(1.0),
            gt: None,
            lt: None,
        };

        let result = engine
            .evaluate_range_filter(&metadata, "score", &range_filter)
            .unwrap();
        assert!(result);
    }
}