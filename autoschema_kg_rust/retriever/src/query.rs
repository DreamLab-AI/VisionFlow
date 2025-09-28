//! Query processing, expansion, and rewriting for enhanced retrieval
//!
//! Advanced query processing pipeline with semantic expansion and intelligent rewriting

use crate::error::{Result, RetrieverError};
use crate::config::{QueryConfig, ExpansionStrategy, RewritingStrategy, PreprocessingConfig};
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use stemmer::Stemmer;
use stop_words::{get, LANGUAGE};
use unicode_segmentation::UnicodeSegmentation;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Processed query with expanded and rewritten variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedQuery {
    /// Original query text
    pub original: String,

    /// Preprocessed query
    pub preprocessed: String,

    /// Expanded query variants
    pub expanded: Vec<ExpandedQuery>,

    /// Rewritten query variants
    pub rewritten: Vec<RewrittenQuery>,

    /// Query features
    pub features: QueryFeatures,

    /// Processing metadata
    pub metadata: QueryMetadata,
}

/// Expanded query variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedQuery {
    /// Expanded query text
    pub text: String,

    /// Expansion strategy used
    pub strategy: String,

    /// Expansion confidence score
    pub confidence: f32,

    /// Added terms
    pub added_terms: Vec<String>,

    /// Term weights
    pub term_weights: HashMap<String, f32>,
}

/// Rewritten query variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewrittenQuery {
    /// Rewritten query text
    pub text: String,

    /// Rewriting strategy used
    pub strategy: String,

    /// Rewriting confidence score
    pub confidence: f32,

    /// Transformation type
    pub transformation: String,
}

/// Query features extracted for ranking and filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    /// Query length in tokens
    pub token_count: usize,

    /// Query length in characters
    pub char_count: usize,

    /// Entity mentions
    pub entities: Vec<String>,

    /// Key phrases
    pub key_phrases: Vec<String>,

    /// Query type classification
    pub query_type: QueryType,

    /// Intent classification
    pub intent: QueryIntent,

    /// Complexity score
    pub complexity: f32,

    /// Ambiguity score
    pub ambiguity: f32,
}

/// Query type classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryType {
    Factual,
    Definitional,
    ListQuery,
    Comparison,
    Procedural,
    Causal,
    Temporal,
    Spatial,
    Unknown,
}

/// Query intent classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryIntent {
    Information,
    Navigation,
    Transaction,
    Learning,
    Verification,
    Exploration,
    Unknown,
}

/// Query processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    /// Processing timestamp
    pub timestamp: i64,

    /// Processing duration in milliseconds
    pub processing_time_ms: u64,

    /// Language detected
    pub language: String,

    /// Processing steps applied
    pub steps: Vec<String>,

    /// Warnings encountered
    pub warnings: Vec<String>,
}

/// Query preprocessing pipeline
pub struct QueryPreprocessor {
    config: PreprocessingConfig,
    stemmer: Stemmer,
    stop_words: HashSet<String>,
    regex_patterns: HashMap<String, Regex>,
}

impl QueryPreprocessor {
    /// Create new query preprocessor
    pub fn new(config: PreprocessingConfig) -> Result<Self> {
        let stemmer = Stemmer::new(stemmer::Algorithm::English)
            .map_err(|e| RetrieverError::query_processing(format!("Failed to create stemmer: {:?}", e)))?;

        let stop_words = get(LANGUAGE::English)
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let mut regex_patterns = HashMap::new();

        // Compile common regex patterns
        regex_patterns.insert(
            "punctuation".to_string(),
            Regex::new(r"[^\w\s]").map_err(|e| RetrieverError::query_processing(format!("Regex compilation failed: {}", e)))?
        );

        regex_patterns.insert(
            "multiple_spaces".to_string(),
            Regex::new(r"\s+").map_err(|e| RetrieverError::query_processing(format!("Regex compilation failed: {}", e)))?
        );

        Ok(Self {
            config,
            stemmer,
            stop_words,
            regex_patterns,
        })
    }

    /// Preprocess query text
    pub fn preprocess(&self, text: &str) -> Result<String> {
        let mut processed = text.to_string();

        // Unicode normalization
        if self.config.unicode_normalization {
            processed = processed.nfc().collect();
        }

        // Lowercase
        if self.config.lowercase {
            processed = processed.to_lowercase();
        }

        // Remove punctuation
        if self.config.remove_punctuation {
            if let Some(regex) = self.regex_patterns.get("punctuation") {
                processed = regex.replace_all(&processed, " ").to_string();
            }
        }

        // Remove extra spaces
        if let Some(regex) = self.regex_patterns.get("multiple_spaces") {
            processed = regex.replace_all(&processed, " ").to_string();
        }

        // Tokenize and process
        let tokens: Vec<String> = processed
            .unicode_words()
            .map(|word| {
                let mut token = word.to_string();

                // Remove stop words
                if self.config.remove_stopwords && self.stop_words.contains(&token) {
                    return None;
                }

                // Stemming
                if self.config.stemming {
                    token = self.stemmer.stem(&token).to_string();
                }

                Some(token)
            })
            .filter_map(|t| t)
            .collect();

        Ok(tokens.join(" ").trim().to_string())
    }

    /// Extract basic features from query
    pub fn extract_features(&self, text: &str) -> QueryFeatures {
        let tokens: Vec<&str> = text.unicode_words().collect();

        QueryFeatures {
            token_count: tokens.len(),
            char_count: text.chars().count(),
            entities: self.extract_entities(text),
            key_phrases: self.extract_key_phrases(text),
            query_type: self.classify_query_type(text),
            intent: self.classify_intent(text),
            complexity: self.calculate_complexity(text, &tokens),
            ambiguity: self.calculate_ambiguity(text, &tokens),
        }
    }

    fn extract_entities(&self, text: &str) -> Vec<String> {
        // Simple entity extraction based on capitalization patterns
        let mut entities = Vec::new();

        for word in text.unicode_words() {
            if word.chars().next().map_or(false, |c| c.is_uppercase()) && word.len() > 1 {
                entities.push(word.to_string());
            }
        }

        entities
    }

    fn extract_key_phrases(&self, text: &str) -> Vec<String> {
        // Extract n-grams as key phrases
        let tokens: Vec<&str> = text.unicode_words().collect();
        let mut phrases = Vec::new();

        // Bigrams
        for window in tokens.windows(2) {
            phrases.push(window.join(" "));
        }

        // Trigrams
        for window in tokens.windows(3) {
            phrases.push(window.join(" "));
        }

        phrases
    }

    fn classify_query_type(&self, text: &str) -> QueryType {
        let text_lower = text.to_lowercase();

        if text_lower.starts_with("what is") || text_lower.starts_with("define") {
            QueryType::Definitional
        } else if text_lower.starts_with("list") || text_lower.starts_with("name") {
            QueryType::ListQuery
        } else if text_lower.contains("compare") || text_lower.contains("difference") {
            QueryType::Comparison
        } else if text_lower.starts_with("how to") || text_lower.starts_with("how do") {
            QueryType::Procedural
        } else if text_lower.contains("why") || text_lower.contains("because") {
            QueryType::Causal
        } else if text_lower.contains("when") || text_lower.contains("time") {
            QueryType::Temporal
        } else if text_lower.contains("where") || text_lower.contains("location") {
            QueryType::Spatial
        } else {
            QueryType::Factual
        }
    }

    fn classify_intent(&self, text: &str) -> QueryIntent {
        let text_lower = text.to_lowercase();

        if text_lower.contains("learn") || text_lower.contains("understand") {
            QueryIntent::Learning
        } else if text_lower.contains("find") || text_lower.contains("navigate") {
            QueryIntent::Navigation
        } else if text_lower.contains("buy") || text_lower.contains("purchase") {
            QueryIntent::Transaction
        } else if text_lower.contains("verify") || text_lower.contains("check") {
            QueryIntent::Verification
        } else if text_lower.contains("explore") || text_lower.contains("discover") {
            QueryIntent::Exploration
        } else {
            QueryIntent::Information
        }
    }

    fn calculate_complexity(&self, text: &str, tokens: &[&str]) -> f32 {
        let avg_word_length = tokens.iter().map(|t| t.len()).sum::<usize>() as f32 / tokens.len().max(1) as f32;
        let sentence_count = text.matches('.').count().max(1) as f32;
        let words_per_sentence = tokens.len() as f32 / sentence_count;

        // Normalize to 0-1 range
        ((avg_word_length - 3.0) / 10.0 + (words_per_sentence - 5.0) / 20.0).clamp(0.0, 1.0)
    }

    fn calculate_ambiguity(&self, text: &str, tokens: &[&str]) -> f32 {
        let question_words = ["what", "how", "why", "when", "where", "which", "who"];
        let question_count = tokens.iter().filter(|t| question_words.contains(&t.to_lowercase().as_str())).count();

        let pronoun_words = ["it", "this", "that", "they", "them"];
        let pronoun_count = tokens.iter().filter(|t| pronoun_words.contains(&t.to_lowercase().as_str())).count();

        // Normalize to 0-1 range
        ((question_count + pronoun_count) as f32 / tokens.len().max(1) as f32).clamp(0.0, 1.0)
    }
}

/// Query expansion engine
pub struct QueryExpander {
    config: QueryConfig,
    embeddings: Option<Arc<dyn EmbeddingModel>>,
    synonym_dict: Arc<RwLock<HashMap<String, Vec<String>>>>,
    cooccurrence_matrix: Arc<RwLock<HashMap<String, HashMap<String, f32>>>>,
}

#[async_trait]
trait EmbeddingModel: Send + Sync {
    async fn get_similar_terms(&self, term: &str, k: usize) -> Result<Vec<(String, f32)>>;
}

impl QueryExpander {
    /// Create new query expander
    pub fn new(config: QueryConfig, embeddings: Option<Arc<dyn EmbeddingModel>>) -> Self {
        Self {
            config,
            embeddings,
            synonym_dict: Arc::new(RwLock::new(HashMap::new())),
            cooccurrence_matrix: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Expand query using configured strategies
    pub async fn expand(&self, query: &str) -> Result<Vec<ExpandedQuery>> {
        if !self.config.expansion.enabled {
            return Ok(vec![]);
        }

        let mut expanded_queries = Vec::new();

        for strategy in &self.config.expansion.strategies {
            match strategy {
                ExpansionStrategy::Synonym => {
                    if let Ok(expanded) = self.expand_with_synonyms(query).await {
                        expanded_queries.push(expanded);
                    }
                }
                ExpansionStrategy::WordNet => {
                    if let Ok(expanded) = self.expand_with_wordnet(query).await {
                        expanded_queries.push(expanded);
                    }
                }
                ExpansionStrategy::Embedding { similarity_threshold } => {
                    if let Ok(expanded) = self.expand_with_embeddings(query, *similarity_threshold).await {
                        expanded_queries.push(expanded);
                    }
                }
                ExpansionStrategy::CoOccurrence { window_size } => {
                    if let Ok(expanded) = self.expand_with_cooccurrence(query, *window_size).await {
                        expanded_queries.push(expanded);
                    }
                }
                ExpansionStrategy::Ngram { n } => {
                    if let Ok(expanded) = self.expand_with_ngrams(query, *n).await {
                        expanded_queries.push(expanded);
                    }
                }
            }
        }

        // Limit number of expanded queries
        expanded_queries.truncate(self.config.expansion.num_terms);

        Ok(expanded_queries)
    }

    async fn expand_with_synonyms(&self, query: &str) -> Result<ExpandedQuery> {
        let tokens: Vec<&str> = query.unicode_words().collect();
        let mut expanded_terms = Vec::new();
        let mut term_weights = HashMap::new();
        let synonym_dict = self.synonym_dict.read().await;

        for token in &tokens {
            term_weights.insert(token.to_string(), 1.0);

            if let Some(synonyms) = synonym_dict.get(*token) {
                for synonym in synonyms.iter().take(2) {
                    expanded_terms.push(synonym.clone());
                    term_weights.insert(synonym.clone(), 0.8);
                }
            }
        }

        let expanded_text = if expanded_terms.is_empty() {
            query.to_string()
        } else {
            format!("{} {}", query, expanded_terms.join(" "))
        };

        Ok(ExpandedQuery {
            text: expanded_text,
            strategy: "synonym".to_string(),
            confidence: if expanded_terms.is_empty() { 0.5 } else { 0.8 },
            added_terms: expanded_terms,
            term_weights,
        })
    }

    async fn expand_with_wordnet(&self, query: &str) -> Result<ExpandedQuery> {
        // Placeholder for WordNet integration
        // In practice, you'd integrate with a WordNet library
        self.expand_with_synonyms(query).await
    }

    async fn expand_with_embeddings(&self, query: &str, similarity_threshold: f32) -> Result<ExpandedQuery> {
        let tokens: Vec<&str> = query.unicode_words().collect();
        let mut expanded_terms = Vec::new();
        let mut term_weights = HashMap::new();

        if let Some(embeddings) = &self.embeddings {
            for token in &tokens {
                term_weights.insert(token.to_string(), 1.0);

                let similar_terms = embeddings.get_similar_terms(token, 3).await?;
                for (term, similarity) in similar_terms {
                    if similarity >= similarity_threshold {
                        expanded_terms.push(term.clone());
                        term_weights.insert(term, similarity);
                    }
                }
            }
        }

        let expanded_text = if expanded_terms.is_empty() {
            query.to_string()
        } else {
            format!("{} {}", query, expanded_terms.join(" "))
        };

        Ok(ExpandedQuery {
            text: expanded_text,
            strategy: "embedding".to_string(),
            confidence: 0.9,
            added_terms: expanded_terms,
            term_weights,
        })
    }

    async fn expand_with_cooccurrence(&self, query: &str, window_size: usize) -> Result<ExpandedQuery> {
        let tokens: Vec<&str> = query.unicode_words().collect();
        let mut expanded_terms = Vec::new();
        let mut term_weights = HashMap::new();
        let cooccurrence = self.cooccurrence_matrix.read().await;

        for token in &tokens {
            term_weights.insert(token.to_string(), 1.0);

            if let Some(cooccurring) = cooccurrence.get(*token) {
                let mut sorted_terms: Vec<_> = cooccurring.iter().collect();
                sorted_terms.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (term, score) in sorted_terms.iter().take(3) {
                    if *score >= self.config.expansion.min_frequency {
                        expanded_terms.push(term.to_string());
                        term_weights.insert(term.to_string(), **score);
                    }
                }
            }
        }

        let expanded_text = if expanded_terms.is_empty() {
            query.to_string()
        } else {
            format!("{} {}", query, expanded_terms.join(" "))
        };

        Ok(ExpandedQuery {
            text: expanded_text,
            strategy: "cooccurrence".to_string(),
            confidence: 0.7,
            added_terms: expanded_terms,
            term_weights,
        })
    }

    async fn expand_with_ngrams(&self, query: &str, n: usize) -> Result<ExpandedQuery> {
        let tokens: Vec<&str> = query.unicode_words().collect();
        let mut expanded_terms = Vec::new();
        let mut term_weights = HashMap::new();

        // Generate n-grams
        for window in tokens.windows(n) {
            let ngram = window.join(" ");
            expanded_terms.push(ngram.clone());
            term_weights.insert(ngram, 0.6);
        }

        // Original tokens
        for token in &tokens {
            term_weights.insert(token.to_string(), 1.0);
        }

        let expanded_text = if expanded_terms.is_empty() {
            query.to_string()
        } else {
            format!("{} {}", query, expanded_terms.join(" "))
        };

        Ok(ExpandedQuery {
            text: expanded_text,
            strategy: format!("{}-gram", n),
            confidence: 0.6,
            added_terms: expanded_terms,
            term_weights,
        })
    }

    /// Load synonym dictionary from file or API
    pub async fn load_synonyms(&self, synonyms: HashMap<String, Vec<String>>) -> Result<()> {
        let mut synonym_dict = self.synonym_dict.write().await;
        *synonym_dict = synonyms;
        Ok(())
    }

    /// Build co-occurrence matrix from corpus
    pub async fn build_cooccurrence_matrix(&self, corpus: &[String], window_size: usize) -> Result<()> {
        let mut matrix = HashMap::new();

        for document in corpus {
            let tokens: Vec<&str> = document.unicode_words().collect();

            for (i, token1) in tokens.iter().enumerate() {
                let entry = matrix.entry(token1.to_string()).or_insert_with(HashMap::new);

                for j in (i.saturating_sub(window_size)..=(i + window_size).min(tokens.len() - 1)) {
                    if i != j {
                        let token2 = tokens[j];
                        *entry.entry(token2.to_string()).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        // Normalize scores
        for (_, cooccurring) in matrix.iter_mut() {
            let total: f32 = cooccurring.values().sum();
            if total > 0.0 {
                for score in cooccurring.values_mut() {
                    *score /= total;
                }
            }
        }

        let mut cooccurrence_matrix = self.cooccurrence_matrix.write().await;
        *cooccurrence_matrix = matrix;

        Ok(())
    }
}

/// Query rewriting engine
pub struct QueryRewriter {
    config: QueryConfig,
    rewrite_patterns: HashMap<String, Vec<RewritePattern>>,
}

#[derive(Debug, Clone)]
struct RewritePattern {
    pattern: Regex,
    replacement: String,
    confidence: f32,
}

impl QueryRewriter {
    /// Create new query rewriter
    pub fn new(config: QueryConfig) -> Result<Self> {
        let mut rewriter = Self {
            config,
            rewrite_patterns: HashMap::new(),
        };

        // Initialize rewrite patterns
        rewriter.initialize_patterns()?;

        Ok(rewriter)
    }

    /// Rewrite query using configured strategies
    pub async fn rewrite(&self, query: &str) -> Result<Vec<RewrittenQuery>> {
        if !self.config.rewriting.enabled {
            return Ok(vec![]);
        }

        let mut rewritten_queries = Vec::new();

        for strategy in &self.config.rewriting.strategies {
            match strategy {
                RewritingStrategy::Paraphrase => {
                    if let Ok(rewritten) = self.paraphrase_query(query).await {
                        rewritten_queries.push(rewritten);
                    }
                }
                RewritingStrategy::Simplification => {
                    if let Ok(rewritten) = self.simplify_query(query).await {
                        rewritten_queries.push(rewritten);
                    }
                }
                RewritingStrategy::TermSubstitution => {
                    if let Ok(rewritten) = self.substitute_terms(query).await {
                        rewritten_queries.push(rewritten);
                    }
                }
                RewritingStrategy::GrammarCorrection => {
                    if let Ok(rewritten) = self.correct_grammar(query).await {
                        rewritten_queries.push(rewritten);
                    }
                }
                RewritingStrategy::ConceptualExpansion => {
                    if let Ok(rewritten) = self.expand_concepts(query).await {
                        rewritten_queries.push(rewritten);
                    }
                }
            }
        }

        // Limit number of rewrites
        rewritten_queries.truncate(self.config.rewriting.max_rewrites);

        Ok(rewritten_queries)
    }

    fn initialize_patterns(&mut self) -> Result<()> {
        // Paraphrase patterns
        let mut paraphrase_patterns = Vec::new();
        paraphrase_patterns.push(RewritePattern {
            pattern: Regex::new(r"how to (.+)").map_err(|e| RetrieverError::query_processing(format!("Regex error: {}", e)))?,
            replacement: "instructions for $1".to_string(),
            confidence: 0.8,
        });

        self.rewrite_patterns.insert("paraphrase".to_string(), paraphrase_patterns);

        // Simplification patterns
        let mut simplification_patterns = Vec::new();
        simplification_patterns.push(RewritePattern {
            pattern: Regex::new(r"can you tell me (.+)").map_err(|e| RetrieverError::query_processing(format!("Regex error: {}", e)))?,
            replacement: "$1".to_string(),
            confidence: 0.9,
        });

        self.rewrite_patterns.insert("simplification".to_string(), simplification_patterns);

        Ok(())
    }

    async fn paraphrase_query(&self, query: &str) -> Result<RewrittenQuery> {
        if let Some(patterns) = self.rewrite_patterns.get("paraphrase") {
            for pattern in patterns {
                if pattern.pattern.is_match(query) {
                    let rewritten = pattern.pattern.replace_all(query, &pattern.replacement);
                    return Ok(RewrittenQuery {
                        text: rewritten.to_string(),
                        strategy: "paraphrase".to_string(),
                        confidence: pattern.confidence,
                        transformation: "pattern_substitution".to_string(),
                    });
                }
            }
        }

        // Fallback: simple paraphrasing
        Ok(RewrittenQuery {
            text: query.to_string(),
            strategy: "paraphrase".to_string(),
            confidence: 0.5,
            transformation: "none".to_string(),
        })
    }

    async fn simplify_query(&self, query: &str) -> Result<RewrittenQuery> {
        if let Some(patterns) = self.rewrite_patterns.get("simplification") {
            for pattern in patterns {
                if pattern.pattern.is_match(query) {
                    let rewritten = pattern.pattern.replace_all(query, &pattern.replacement);
                    return Ok(RewrittenQuery {
                        text: rewritten.to_string(),
                        strategy: "simplification".to_string(),
                        confidence: pattern.confidence,
                        transformation: "pattern_simplification".to_string(),
                    });
                }
            }
        }

        // Remove filler words
        let filler_words = ["please", "could you", "can you", "I would like to", "I want to"];
        let mut simplified = query.to_string();

        for filler in &filler_words {
            simplified = simplified.replace(filler, "").trim().to_string();
        }

        Ok(RewrittenQuery {
            text: simplified,
            strategy: "simplification".to_string(),
            confidence: 0.7,
            transformation: "filler_removal".to_string(),
        })
    }

    async fn substitute_terms(&self, query: &str) -> Result<RewrittenQuery> {
        // Simple term substitution
        let substitutions = [
            ("automobile", "car"),
            ("purchase", "buy"),
            ("location", "place"),
            ("obtain", "get"),
        ];

        let mut substituted = query.to_string();
        let mut transformations = Vec::new();

        for (from, to) in &substitutions {
            if substituted.contains(from) {
                substituted = substituted.replace(from, to);
                transformations.push(format!("{} -> {}", from, to));
            }
        }

        Ok(RewrittenQuery {
            text: substituted,
            strategy: "term_substitution".to_string(),
            confidence: if transformations.is_empty() { 0.5 } else { 0.8 },
            transformation: transformations.join(", "),
        })
    }

    async fn correct_grammar(&self, query: &str) -> Result<RewrittenQuery> {
        // Simple grammar corrections
        let corrections = [
            (r"\bis\s+are\b", "are"),
            (r"\bare\s+is\b", "is"),
            (r"\ba\s+([aeiou])", "an $1"),
        ];

        let mut corrected = query.to_string();
        let mut transformations = Vec::new();

        for (pattern, replacement) in &corrections {
            let regex = Regex::new(pattern).map_err(|e| RetrieverError::query_processing(format!("Regex error: {}", e)))?;
            if regex.is_match(&corrected) {
                corrected = regex.replace_all(&corrected, *replacement).to_string();
                transformations.push(format!("{} -> {}", pattern, replacement));
            }
        }

        Ok(RewrittenQuery {
            text: corrected,
            strategy: "grammar_correction".to_string(),
            confidence: if transformations.is_empty() { 0.5 } else { 0.9 },
            transformation: transformations.join(", "),
        })
    }

    async fn expand_concepts(&self, query: &str) -> Result<RewrittenQuery> {
        // Conceptual expansion
        let concepts = [
            ("machine learning", "machine learning artificial intelligence ML AI"),
            ("climate change", "climate change global warming environmental"),
            ("renewable energy", "renewable energy solar wind sustainable"),
        ];

        let mut expanded = query.to_string();
        let mut transformations = Vec::new();

        for (concept, expansion) in &concepts {
            if expanded.to_lowercase().contains(concept) {
                expanded = expanded.replace(concept, expansion);
                transformations.push(format!("{} -> {}", concept, expansion));
            }
        }

        Ok(RewrittenQuery {
            text: expanded,
            strategy: "conceptual_expansion".to_string(),
            confidence: if transformations.is_empty() { 0.5 } else { 0.8 },
            transformation: transformations.join(", "),
        })
    }
}

/// Main query processor orchestrating all processing steps
pub struct QueryProcessor {
    preprocessor: QueryPreprocessor,
    expander: QueryExpander,
    rewriter: QueryRewriter,
    config: QueryConfig,
}

impl QueryProcessor {
    /// Create new query processor
    pub fn new(config: QueryConfig, embeddings: Option<Arc<dyn EmbeddingModel>>) -> Result<Self> {
        Ok(Self {
            preprocessor: QueryPreprocessor::new(config.preprocessing.clone())?,
            expander: QueryExpander::new(config.clone(), embeddings),
            rewriter: QueryRewriter::new(config.clone())?,
            config,
        })
    }

    /// Process query through full pipeline
    pub async fn process(&self, query: &str) -> Result<ProcessedQuery> {
        let start_time = std::time::Instant::now();
        let mut steps = Vec::new();
        let mut warnings = Vec::new();

        // Validate input
        if query.trim().is_empty() {
            return Err(RetrieverError::invalid_input("Query cannot be empty"));
        }

        if query.len() > self.config.max_length {
            warnings.push(format!("Query truncated from {} to {} characters", query.len(), self.config.max_length));
        }

        let truncated_query = if query.len() > self.config.max_length {
            &query[..self.config.max_length]
        } else {
            query
        };

        // Preprocessing
        steps.push("preprocessing".to_string());
        let preprocessed = self.preprocessor.preprocess(truncated_query)?;

        // Feature extraction
        steps.push("feature_extraction".to_string());
        let features = self.preprocessor.extract_features(&preprocessed);

        // Query expansion
        steps.push("expansion".to_string());
        let expanded = self.expander.expand(&preprocessed).await?;

        // Query rewriting
        steps.push("rewriting".to_string());
        let rewritten = self.rewriter.rewrite(&preprocessed).await?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ProcessedQuery {
            original: query.to_string(),
            preprocessed,
            expanded,
            rewritten,
            features,
            metadata: QueryMetadata {
                timestamp: chrono::Utc::now().timestamp(),
                processing_time_ms: processing_time,
                language: "en".to_string(), // Simplified language detection
                steps,
                warnings,
            },
        })
    }

    /// Process multiple queries in parallel
    pub async fn process_batch(&self, queries: &[String]) -> Result<Vec<ProcessedQuery>> {
        let results: Result<Vec<_>> = stream::iter(queries)
            .map(|query| async move {
                self.process(query).await
            })
            .buffer_unordered(self.config.expansion.num_terms)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_preprocessing() {
        let config = PreprocessingConfig::default();
        let preprocessor = QueryPreprocessor::new(config).unwrap();

        let query = "What is Machine Learning?";
        let processed = preprocessor.preprocess(query).unwrap();

        assert!(!processed.is_empty());
        assert!(!processed.contains("?"));
    }

    #[tokio::test]
    async fn test_query_features() {
        let config = PreprocessingConfig::default();
        let preprocessor = QueryPreprocessor::new(config).unwrap();

        let query = "How to implement machine learning algorithms?";
        let features = preprocessor.extract_features(query);

        assert_eq!(features.query_type, QueryType::Procedural);
        assert_eq!(features.intent, QueryIntent::Learning);
        assert!(features.token_count > 0);
    }

    #[tokio::test]
    async fn test_query_expansion() {
        let config = QueryConfig::default();
        let expander = QueryExpander::new(config, None);

        let query = "machine learning";
        let expanded = expander.expand(query).await.unwrap();

        // Should create expanded queries even without external resources
        assert!(!expanded.is_empty());
    }

    #[tokio::test]
    async fn test_query_rewriting() {
        let config = QueryConfig::default();
        let rewriter = QueryRewriter::new(config).unwrap();

        let query = "Can you tell me about artificial intelligence?";
        let rewritten = rewriter.rewrite(query).await.unwrap();

        assert!(!rewritten.is_empty());
        // Should simplify the query
        assert!(rewritten.iter().any(|r| r.strategy == "simplification"));
    }

    #[test]
    fn test_query_type_classification() {
        let config = PreprocessingConfig::default();
        let preprocessor = QueryPreprocessor::new(config).unwrap();

        assert_eq!(preprocessor.classify_query_type("What is AI?"), QueryType::Definitional);
        assert_eq!(preprocessor.classify_query_type("How to learn programming?"), QueryType::Procedural);
        assert_eq!(preprocessor.classify_query_type("List all programming languages"), QueryType::ListQuery);
        assert_eq!(preprocessor.classify_query_type("Compare Python and Java"), QueryType::Comparison);
    }
}
"