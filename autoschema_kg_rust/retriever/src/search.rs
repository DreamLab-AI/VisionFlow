//! Semantic search engine with hybrid retrieval strategies
//!
//! Advanced search capabilities combining vector, keyword, and graph-based retrieval

use crate::error::{Result, RetrieverError};
use crate::config::{SearchConfig, SearchStrategy, HybridSearchConfig, FusionStrategy, NormalizationMethod};
use crate::embeddings::{VectorIndex, SimilarityResult, Embedding};
use crate::graph::{KnowledgeGraph, GraphNode, GraphTraverser, TraversalStrategy};
use crate::query::{ProcessedQuery, QueryProcessor};
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Search result with relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID
    pub id: String,

    /// Relevance score (0.0 to 1.0)
    pub score: f32,

    /// Result content/text
    pub content: String,

    /// Result type
    pub result_type: ResultType,

    /// Metadata
    pub metadata: HashMap<String, String>,

    /// Source strategy that found this result
    pub source_strategy: String,

    /// Individual strategy scores
    pub strategy_scores: HashMap<String, f32>,

    /// Explanation of why this result was selected
    pub explanation: String,

    /// Snippet with highlighted terms
    pub snippet: Option<String>,
}

/// Type of search result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResultType {
    Document,
    Passage,
    Entity,
    Concept,
    Summary,
    Answer,
}

/// Search strategy implementation trait
#[async_trait]
pub trait SearchStrategyTrait: Send + Sync {
    /// Execute search with given query
    async fn search(&self, query: &ProcessedQuery, limit: usize) -> Result<Vec<SearchResult>>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Get strategy weight
    fn weight(&self) -> f32;
}

/// Vector-based search strategy
pub struct VectorSearchStrategy {
    vector_index: Arc<VectorIndex>,
    embedding_model: Arc<dyn crate::embeddings::EmbeddingModel>,
    weight: f32,
}

impl VectorSearchStrategy {
    /// Create new vector search strategy
    pub fn new(
        vector_index: Arc<VectorIndex>,
        embedding_model: Arc<dyn crate::embeddings::EmbeddingModel>,
        weight: f32,
    ) -> Self {
        Self {
            vector_index,
            embedding_model,
            weight,
        }
    }
}

#[async_trait]
impl SearchStrategyTrait for VectorSearchStrategy {
    async fn search(&self, query: &ProcessedQuery, limit: usize) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let query_vector = self.embedding_model.embed_text(&query.preprocessed).await?;

        // Search vector index
        let similarity_results = self.vector_index.search(&query_vector, limit).await?;

        // Convert to search results
        let mut results = Vec::new();
        for sim_result in similarity_results {
            results.push(SearchResult {
                id: sim_result.id.clone(),
                score: sim_result.score * self.weight,
                content: sim_result.embedding.text.clone(),
                result_type: ResultType::Document,
                metadata: sim_result.embedding.metadata.clone(),
                source_strategy: "vector".to_string(),
                strategy_scores: HashMap::from([("vector".to_string(), sim_result.score)]),
                explanation: format!("Vector similarity: {:.3}", sim_result.score),
                snippet: self.generate_snippet(&sim_result.embedding.text, &query.preprocessed),
            });
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "vector"
    }

    fn weight(&self) -> f32 {
        self.weight
    }
}

impl VectorSearchStrategy {
    fn generate_snippet(&self, text: &str, query: &str) -> Option<String> {
        let query_terms: HashSet<_> = query.split_whitespace().map(|s| s.to_lowercase()).collect();
        let words: Vec<&str> = text.split_whitespace().collect();

        // Find first occurrence of any query term
        let mut start_idx = 0;
        for (i, word) in words.iter().enumerate() {
            if query_terms.contains(&word.to_lowercase()) {
                start_idx = i.saturating_sub(10); // 10 words before
                break;
            }
        }

        let end_idx = (start_idx + 50).min(words.len()); // 50 words total
        let snippet_words = &words[start_idx..end_idx];

        if snippet_words.is_empty() {
            return None;
        }

        let mut snippet = snippet_words.join(" ");

        // Highlight query terms
        for term in &query_terms {
            snippet = snippet.replace(term, &format!("**{}**", term));
        }

        Some(snippet)
    }
}

/// Keyword-based search strategy using TF-IDF
pub struct KeywordSearchStrategy {
    documents: Arc<RwLock<HashMap<String, Document>>>,
    term_frequencies: Arc<RwLock<HashMap<String, HashMap<String, f32>>>>,
    document_frequencies: Arc<RwLock<HashMap<String, usize>>>,
    total_documents: Arc<RwLock<usize>>,
    weight: f32,
}

#[derive(Debug, Clone)]
struct Document {
    id: String,
    content: String,
    metadata: HashMap<String, String>,
    term_frequencies: HashMap<String, f32>,
    length: usize,
}

impl KeywordSearchStrategy {
    /// Create new keyword search strategy
    pub fn new(weight: f32) -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            term_frequencies: Arc::new(RwLock::new(HashMap::new())),
            document_frequencies: Arc::new(RwLock::new(HashMap::new())),
            total_documents: Arc::new(RwLock::new(0)),
            weight,
        }
    }

    /// Add document to keyword index
    pub async fn add_document(&self, id: String, content: String, metadata: HashMap<String, String>) -> Result<()> {
        let tokens: Vec<&str> = content.split_whitespace().collect();
        let mut term_freqs = HashMap::new();

        // Calculate term frequencies
        for token in &tokens {
            let term = token.to_lowercase();
            *term_freqs.entry(term).or_insert(0.0) += 1.0;
        }

        // Normalize by document length
        let doc_length = tokens.len() as f32;
        for freq in term_freqs.values_mut() {
            *freq /= doc_length;
        }

        let document = Document {
            id: id.clone(),
            content,
            metadata,
            term_frequencies: term_freqs.clone(),
            length: tokens.len(),
        };

        // Update indices
        let mut documents = self.documents.write().await;
        let mut tf = self.term_frequencies.write().await;
        let mut df = self.document_frequencies.write().await;
        let mut total_docs = self.total_documents.write().await;

        documents.insert(id.clone(), document);
        tf.insert(id, term_freqs.clone());

        // Update document frequencies
        for term in term_freqs.keys() {
            *df.entry(term.clone()).or_insert(0) += 1;
        }

        *total_docs += 1;

        Ok(())
    }

    fn calculate_tfidf(&self, term: &str, doc_tf: f32, doc_freq: usize, total_docs: usize) -> f32 {
        if doc_freq == 0 {
            return 0.0;
        }

        let idf = (total_docs as f32 / doc_freq as f32).ln();
        doc_tf * idf
    }
}

#[async_trait]
impl SearchStrategyTrait for KeywordSearchStrategy {
    async fn search(&self, query: &ProcessedQuery, limit: usize) -> Result<Vec<SearchResult>> {
        let query_terms: Vec<&str> = query.preprocessed.split_whitespace().collect();
        let documents = self.documents.read().await;
        let tf = self.term_frequencies.read().await;
        let df = self.document_frequencies.read().await;
        let total_docs = *self.total_documents.read().await;

        let mut doc_scores: Vec<(String, f32)> = Vec::new();

        // Calculate TF-IDF scores for each document
        for (doc_id, doc) in documents.iter() {
            let mut score = 0.0;

            for query_term in &query_terms {
                let term = query_term.to_lowercase();
                if let Some(doc_tf) = doc.term_frequencies.get(&term) {
                    let doc_freq = df.get(&term).unwrap_or(&0);
                    let tfidf = self.calculate_tfidf(&term, *doc_tf, *doc_freq, total_docs);
                    score += tfidf;
                }
            }

            if score > 0.0 {
                doc_scores.push((doc_id.clone(), score));
            }
        }

        // Sort by score descending
        doc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to search results
        let mut results = Vec::new();
        for (doc_id, score) in doc_scores.into_iter().take(limit) {
            if let Some(doc) = documents.get(&doc_id) {
                results.push(SearchResult {
                    id: doc_id.clone(),
                    score: score * self.weight,
                    content: doc.content.clone(),
                    result_type: ResultType::Document,
                    metadata: doc.metadata.clone(),
                    source_strategy: "keyword".to_string(),
                    strategy_scores: HashMap::from([("keyword".to_string(), score)]),
                    explanation: format!("TF-IDF score: {:.3}", score),
                    snippet: self.generate_snippet(&doc.content, &query.preprocessed),
                });
            }
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "keyword"
    }

    fn weight(&self) -> f32 {
        self.weight
    }
}

impl KeywordSearchStrategy {
    fn generate_snippet(&self, text: &str, query: &str) -> Option<String> {
        let query_terms: HashSet<_> = query.split_whitespace().map(|s| s.to_lowercase()).collect();
        let sentences: Vec<&str> = text.split('.').collect();

        // Find sentence with most query terms
        let mut best_sentence = "";
        let mut max_matches = 0;

        for sentence in &sentences {
            let words: HashSet<_> = sentence.split_whitespace().map(|s| s.to_lowercase()).collect();
            let matches = query_terms.intersection(&words).count();

            if matches > max_matches {
                max_matches = matches;
                best_sentence = sentence;
            }
        }

        if max_matches > 0 {
            Some(best_sentence.trim().to_string())
        } else {
            None
        }
    }
}

/// Graph-based search strategy
pub struct GraphSearchStrategy {
    graph: Arc<KnowledgeGraph>,
    traverser: GraphTraverser,
    weight: f32,
}

impl GraphSearchStrategy {
    /// Create new graph search strategy
    pub fn new(graph: Arc<KnowledgeGraph>, traverser: GraphTraverser, weight: f32) -> Self {
        Self {
            graph,
            traverser,
            weight,
        }
    }
}

#[async_trait]
impl SearchStrategyTrait for GraphSearchStrategy {
    async fn search(&self, query: &ProcessedQuery, limit: usize) -> Result<Vec<SearchResult>> {
        // Extract entities from query as starting points
        let start_nodes: Vec<String> = query.features.entities.clone();

        if start_nodes.is_empty() {
            return Ok(vec![]);
        }

        // Perform graph traversal
        let hop_results = self.traverser.traverse(start_nodes, TraversalStrategy::BestFirst { beam_width: 20 }).await?;

        let mut results = Vec::new();
        let mut seen_ids = HashSet::new();

        // Convert hop results to search results
        for hop_result in hop_results {
            for node in hop_result.nodes {
                if seen_ids.contains(&node.id) {
                    continue;
                }

                seen_ids.insert(node.id.clone());

                let score = node.relevance * self.weight * (1.0 / (hop_result.hop as f32 + 1.0));

                results.push(SearchResult {
                    id: node.id.clone(),
                    score,
                    content: node.content.clone(),
                    result_type: match node.node_type {
                        crate::graph::NodeType::Document => ResultType::Document,
                        crate::graph::NodeType::Entity => ResultType::Entity,
                        crate::graph::NodeType::Concept => ResultType::Concept,
                        _ => ResultType::Document,
                    },
                    metadata: node.metadata.clone(),
                    source_strategy: "graph".to_string(),
                    strategy_scores: HashMap::from([("graph".to_string(), score)]),
                    explanation: format!("Graph traversal at hop {}, relevance: {:.3}", hop_result.hop, node.relevance),
                    snippet: None,
                });

                if results.len() >= limit {
                    break;
                }
            }

            if results.len() >= limit {
                break;
            }
        }

        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    fn name(&self) -> &str {
        "graph"
    }

    fn weight(&self) -> f32 {
        self.weight
    }
}

/// Semantic search strategy combining embeddings with contextual understanding
pub struct SemanticSearchStrategy {
    vector_strategy: VectorSearchStrategy,
    concept_embeddings: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    weight: f32,
}

impl SemanticSearchStrategy {
    /// Create new semantic search strategy
    pub fn new(vector_strategy: VectorSearchStrategy, weight: f32) -> Self {
        Self {
            vector_strategy,
            concept_embeddings: Arc::new(RwLock::new(HashMap::new())),
            weight,
        }
    }

    /// Add concept embeddings for semantic enhancement
    pub async fn add_concept(&self, concept: String, embedding: Vec<f32>) -> Result<()> {
        let mut concepts = self.concept_embeddings.write().await;
        concepts.insert(concept, embedding);
        Ok(())
    }
}

#[async_trait]
impl SearchStrategyTrait for SemanticSearchStrategy {
    async fn search(&self, query: &ProcessedQuery, limit: usize) -> Result<Vec<SearchResult>> {
        // Start with vector search results
        let mut results = self.vector_strategy.search(query, limit * 2).await?;

        // Enhance with semantic understanding
        let concepts = self.concept_embeddings.read().await;

        for result in &mut results {
            let mut semantic_boost = 0.0;

            // Check for concept matches
            for (concept, concept_embedding) in concepts.iter() {
                if result.content.to_lowercase().contains(&concept.to_lowercase()) {
                    // Calculate semantic similarity boost
                    if let Ok(query_embedding) = self.vector_strategy.embedding_model.embed_text(&query.preprocessed).await {
                        let similarity = cosine_similarity(&query_embedding, concept_embedding);
                        semantic_boost += similarity * 0.2; // 20% boost for concept matches
                    }
                }
            }

            // Apply semantic boost
            result.score = (result.score + semantic_boost).min(1.0);
            result.source_strategy = "semantic".to_string();
            result.strategy_scores.insert("semantic".to_string(), semantic_boost);
            result.explanation = format!("{}, semantic boost: {:.3}", result.explanation, semantic_boost);
        }

        // Re-sort and limit
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    fn name(&self) -> &str {
        "semantic"
    }

    fn weight(&self) -> f32 {
        self.weight
    }
}

/// Main search engine orchestrating multiple strategies
pub struct SearchEngine {
    strategies: Vec<Box<dyn SearchStrategyTrait>>,
    config: SearchConfig,
    query_processor: Arc<QueryProcessor>,
}

impl SearchEngine {
    /// Create new search engine
    pub fn new(config: SearchConfig, query_processor: Arc<QueryProcessor>) -> Self {
        Self {
            strategies: Vec::new(),
            config,
            query_processor,
        }
    }

    /// Add search strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn SearchStrategyTrait>) {
        self.strategies.push(strategy);
    }

    /// Execute hybrid search across all strategies
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // Process query
        let processed_query = self.query_processor.process(query).await?;

        // Execute all strategies in parallel
        let strategy_results: Result<Vec<_>> = stream::iter(&self.strategies)
            .map(|strategy| async move {
                let results = strategy.search(&processed_query, self.config.max_results_per_strategy).await?;
                Ok((strategy.name().to_string(), results))
            })
            .buffer_unordered(self.strategies.len())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();

        let strategy_results = strategy_results?;

        // Fuse results
        let fused_results = self.fuse_results(strategy_results).await?;

        // Apply final ranking and filtering
        let final_results = self.rank_and_filter(fused_results, limit).await?;

        Ok(final_results)
    }

    /// Search with multiple query variants
    pub async fn search_multi_query(&self, queries: &[String], limit: usize) -> Result<Vec<SearchResult>> {
        // Process all queries in parallel
        let multi_results: Result<Vec<_>> = stream::iter(queries)
            .map(|query| async move {
                self.search(query, limit / queries.len().max(1)).await
            })
            .buffer_unordered(queries.len())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();

        let multi_results = multi_results?;

        // Combine and deduplicate results
        let mut combined_results = HashMap::new();

        for results in multi_results {
            for result in results {
                combined_results.entry(result.id.clone())
                    .and_modify(|existing: &mut SearchResult| {
                        // Combine scores
                        existing.score = (existing.score + result.score) / 2.0;
                        // Merge strategy scores
                        for (strategy, score) in result.strategy_scores {
                            *existing.strategy_scores.entry(strategy).or_insert(0.0) += score;
                        }
                    })
                    .or_insert(result);
            }
        }

        let mut final_results: Vec<_> = combined_results.into_values().collect();
        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        final_results.truncate(limit);

        Ok(final_results)
    }

    async fn fuse_results(&self, strategy_results: Vec<(String, Vec<SearchResult>)>) -> Result<Vec<SearchResult>> {
        match self.config.hybrid.fusion_strategy {
            FusionStrategy::RankFusion => self.rank_fusion(strategy_results).await,
            FusionStrategy::ScoreFusion => self.score_fusion(strategy_results).await,
            FusionStrategy::WeightedFusion { ref weights } => self.weighted_fusion(strategy_results, weights).await,
            FusionStrategy::AdaptiveFusion => self.adaptive_fusion(strategy_results).await,
        }
    }

    async fn rank_fusion(&self, strategy_results: Vec<(String, Vec<SearchResult>)>) -> Result<Vec<SearchResult>> {
        let mut document_ranks: HashMap<String, Vec<usize>> = HashMap::new();

        // Collect ranks for each document from each strategy
        for (_strategy, results) in strategy_results {
            for (rank, result) in results.iter().enumerate() {
                document_ranks.entry(result.id.clone()).or_default().push(rank + 1);
            }
        }

        // Calculate Reciprocal Rank Fusion (RRF) scores
        let mut fused_results = Vec::new();
        for (doc_id, ranks) in document_ranks {
            let rrf_score: f32 = ranks.iter().map(|&rank| 1.0 / (60.0 + rank as f32)).sum();

            // Find the document from strategy results
            if let Some(result) = self.find_result_by_id(&strategy_results, &doc_id) {
                let mut fused_result = result.clone();
                fused_result.score = rrf_score;
                fused_result.source_strategy = "fused".to_string();
                fused_results.push(fused_result);
            }
        }

        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(fused_results)
    }

    async fn score_fusion(&self, strategy_results: Vec<(String, Vec<SearchResult>)>) -> Result<Vec<SearchResult>> {
        let mut document_scores: HashMap<String, (SearchResult, Vec<f32>)> = HashMap::new();

        // Collect scores for each document from each strategy
        for (_strategy, results) in strategy_results {
            for result in results {
                document_scores.entry(result.id.clone())
                    .and_modify(|(_, scores)| scores.push(result.score))
                    .or_insert((result, vec![result.score]));
            }
        }

        // Normalize and combine scores
        let mut fused_results = Vec::new();
        for (doc_id, (mut result, scores)) in document_scores {
            let normalized_scores = self.normalize_scores(&scores).await?;
            let combined_score: f32 = normalized_scores.iter().sum::<f32>() / normalized_scores.len() as f32;

            result.score = combined_score;
            result.source_strategy = "fused".to_string();
            fused_results.push(result);
        }

        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(fused_results)
    }

    async fn weighted_fusion(&self, strategy_results: Vec<(String, Vec<SearchResult>)>, weights: &[f32]) -> Result<Vec<SearchResult>> {
        let mut document_scores: HashMap<String, (SearchResult, f32)> = HashMap::new();

        // Apply weights to strategy results
        for ((strategy, results), &weight) in strategy_results.iter().zip(weights.iter()) {
            for result in results {
                document_scores.entry(result.id.clone())
                    .and_modify(|(_, score)| *score += result.score * weight)
                    .or_insert((result.clone(), result.score * weight));
            }
        }

        let mut fused_results: Vec<_> = document_scores.into_iter()
            .map(|(_, (mut result, score))| {
                result.score = score;
                result.source_strategy = "weighted_fused".to_string();
                result
            })
            .collect();

        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(fused_results)
    }

    async fn adaptive_fusion(&self, strategy_results: Vec<(String, Vec<SearchResult>)>) -> Result<Vec<SearchResult>> {
        // Adaptive fusion based on query characteristics and strategy performance
        // For now, fallback to score fusion with dynamic weighting
        self.score_fusion(strategy_results).await
    }

    async fn normalize_scores(&self, scores: &[f32]) -> Result<Vec<f32>> {
        if scores.is_empty() {
            return Ok(vec![]);
        }

        match self.config.hybrid.normalization {
            NormalizationMethod::MinMax => {
                let min_score = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let range = max_score - min_score;

                if range == 0.0 {
                    Ok(vec![1.0; scores.len()])
                } else {
                    Ok(scores.iter().map(|&score| (score - min_score) / range).collect())
                }
            }
            NormalizationMethod::ZScore => {
                let mean = scores.iter().sum::<f32>() / scores.len() as f32;
                let variance = scores.iter().map(|&score| (score - mean).powi(2)).sum::<f32>() / scores.len() as f32;
                let std_dev = variance.sqrt();

                if std_dev == 0.0 {
                    Ok(vec![0.0; scores.len()])
                } else {
                    Ok(scores.iter().map(|&score| (score - mean) / std_dev).collect())
                }
            }
            NormalizationMethod::Sigmoid => {
                Ok(scores.iter().map(|&score| 1.0 / (1.0 + (-score).exp())).collect())
            }
            NormalizationMethod::Softmax => {
                let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_scores: Vec<f32> = scores.iter().map(|&score| (score - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                Ok(exp_scores.iter().map(|&exp_score| exp_score / sum_exp).collect())
            }
        }
    }

    fn find_result_by_id(&self, strategy_results: &[(String, Vec<SearchResult>)], doc_id: &str) -> Option<&SearchResult> {
        for (_strategy, results) in strategy_results {
            if let Some(result) = results.iter().find(|r| r.id == doc_id) {
                return Some(result);
            }
        }
        None
    }

    async fn rank_and_filter(&self, mut results: Vec<SearchResult>, limit: usize) -> Result<Vec<SearchResult>> {
        // Filter by minimum score threshold
        results.retain(|r| r.score >= self.config.hybrid.min_score);

        // Final ranking and deduplication
        let mut seen_ids = HashSet::new();
        let mut final_results = Vec::new();

        for result in results {
            if !seen_ids.contains(&result.id) {
                seen_ids.insert(result.id.clone());
                final_results.push(result);

                if final_results.len() >= limit {
                    break;
                }
            }
        }

        Ok(final_results)
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!(similarity > 0.9); // These vectors are very similar
    }

    #[tokio::test]
    async fn test_keyword_search_strategy() {
        let strategy = KeywordSearchStrategy::new(1.0);

        // Add test document
        strategy.add_document(
            "doc1".to_string(),
            "This is a test document about machine learning".to_string(),
            HashMap::new(),
        ).await.unwrap();

        // Create mock processed query
        let query = ProcessedQuery {
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
        };

        let results = strategy.search(&query, 10).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_result_type_conversion() {
        use crate::graph::NodeType;

        let doc_type = match NodeType::Document {
            NodeType::Document => ResultType::Document,
            NodeType::Entity => ResultType::Entity,
            NodeType::Concept => ResultType::Concept,
            _ => ResultType::Document,
        };

        assert_eq!(doc_type, ResultType::Document);
    }

    #[tokio::test]
    async fn test_score_normalization() {
        let config = SearchConfig::default();
        let query_processor = Arc::new(
            crate::query::QueryProcessor::new(crate::config::QueryConfig::default(), None).unwrap()
        );
        let engine = SearchEngine::new(config, query_processor);

        let scores = vec![0.1, 0.5, 0.9, 0.3];
        let normalized = engine.normalize_scores(&scores).await.unwrap();

        // Check that min-max normalization produces values between 0 and 1
        assert!(normalized.iter().all(|&score| score >= 0.0 && score <= 1.0));
        assert_eq!(normalized.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)), 1.0);
        assert_eq!(normalized.iter().fold(f32::INFINITY, |a, &b| a.min(b)), 0.0);
    }
}
"