//! Advanced semantic analysis service for knowledge graph enhancement

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use chrono::{DateTime, Utc};
use crate::models::metadata::Metadata;

/// Semantic features extracted from content and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFeatures {
    /// File identifier (path or unique ID)
    pub id: String,
    /// Topic distribution (topic_name -> weight)
    pub topics: HashMap<String, f32>,
    /// Knowledge domain classification
    pub domains: Vec<KnowledgeDomain>,
    /// Temporal features
    pub temporal: TemporalFeatures,
    /// Structural features
    pub structural: StructuralFeatures,
    /// Content-based features
    pub content: ContentFeatures,
    /// Agent communication patterns (if applicable)
    pub agent_patterns: Option<AgentCommunicationPatterns>,
    /// Overall importance score
    pub importance_score: f32,
}

/// Knowledge domain classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeDomain {
    Mathematics,
    Physics,
    ComputerScience,
    Biology,
    Chemistry,
    Engineering,
    DataScience,
    MachineLearning,
    WebDevelopment,
    SystemsProgramming,
    DevOps,
    Security,
    Documentation,
    Configuration,
    Testing,
    UserInterface,
    Database,
    Networking,
    CloudComputing,
    Other(String),
}

/// Temporal features for co-evolution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatures {
    /// Creation timestamp
    pub created_at: Option<DateTime<Utc>>,
    /// Last modification timestamp
    pub modified_at: Option<DateTime<Utc>>,
    /// Modification frequency (changes per day)
    pub modification_frequency: f32,
    /// Co-evolution score with other files
    pub co_evolution_score: f32,
    /// Temporal cluster ID
    pub temporal_cluster: Option<u32>,
}

/// Structural features from file/code structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralFeatures {
    /// File type/extension
    pub file_type: String,
    /// Depth in directory hierarchy
    pub directory_depth: u32,
    /// Number of dependencies/imports
    pub dependency_count: u32,
    /// Complexity metrics
    pub complexity_score: f32,
    /// Lines of code (if applicable)
    pub loc: Option<u32>,
    /// Module/namespace hierarchy
    pub module_path: Vec<String>,
}

/// Content-based semantic features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFeatures {
    /// Primary language (programming or natural)
    pub language: String,
    /// Key terms/tokens extracted
    pub key_terms: Vec<String>,
    /// Semantic embeddings (if computed)
    pub embeddings: Option<Vec<f32>>,
    /// Content hash for deduplication
    pub content_hash: String,
    /// Readability/documentation score
    pub documentation_score: f32,
}

/// Agent communication pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCommunicationPatterns {
    /// Message send frequency
    pub send_frequency: f32,
    /// Message receive frequency
    pub receive_frequency: f32,
    /// Common communication partners (agent_id -> frequency)
    pub communication_partners: HashMap<String, f32>,
    /// Message types used
    pub message_types: HashSet<String>,
    /// Communication clustering coefficient
    pub clustering_coefficient: f32,
    /// Role in communication network
    pub network_role: NetworkRole,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NetworkRole {
    Hub,        // High degree centrality
    Bridge,     // High betweenness centrality
    Peripheral, // Low centrality
    Isolated,   // No connections
}

/// Configuration for semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalyzerConfig {
    /// Enable topic modeling
    pub enable_topics: bool,
    /// Number of topics to extract
    pub num_topics: usize,
    /// Enable temporal analysis
    pub enable_temporal: bool,
    /// Enable agent pattern analysis
    pub enable_agent_patterns: bool,
    /// Minimum term frequency for key terms
    pub min_term_frequency: f32,
    /// Maximum features to extract
    pub max_features: usize,
    /// Cache analysis results
    pub enable_caching: bool,
}

impl Default for SemanticAnalyzerConfig {
    fn default() -> Self {
        Self {
            enable_topics: true,
            num_topics: 10,
            enable_temporal: true,
            enable_agent_patterns: false,
            min_term_frequency: 0.01,
            max_features: 100,
            enable_caching: true,
        }
    }
}

/// Main semantic analyzer service
pub struct SemanticAnalyzer {
    config: SemanticAnalyzerConfig,
    feature_cache: HashMap<String, SemanticFeatures>,
    domain_patterns: HashMap<KnowledgeDomain, Vec<String>>,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer
    pub fn new(config: SemanticAnalyzerConfig) -> Self {
        let mut analyzer = Self {
            config,
            feature_cache: HashMap::new(),
            domain_patterns: HashMap::new(),
        };
        analyzer.initialize_domain_patterns();
        analyzer
    }

    /// Initialize domain-specific patterns for classification
    fn initialize_domain_patterns(&mut self) {
        self.domain_patterns.insert(
            KnowledgeDomain::Mathematics,
            vec!["theorem", "proof", "equation", "matrix", "vector", "calculus", "algebra"]
                .into_iter().map(String::from).collect()
        );
        
        self.domain_patterns.insert(
            KnowledgeDomain::MachineLearning,
            vec!["model", "training", "neural", "network", "tensor", "gradient", "optimizer"]
                .into_iter().map(String::from).collect()
        );
        
        self.domain_patterns.insert(
            KnowledgeDomain::WebDevelopment,
            vec!["react", "vue", "angular", "html", "css", "javascript", "frontend", "backend"]
                .into_iter().map(String::from).collect()
        );
        
        self.domain_patterns.insert(
            KnowledgeDomain::SystemsProgramming,
            vec!["kernel", "memory", "pointer", "thread", "mutex", "syscall", "buffer"]
                .into_iter().map(String::from).collect()
        );
        
        self.domain_patterns.insert(
            KnowledgeDomain::Database,
            vec!["sql", "query", "table", "index", "transaction", "schema", "relation"]
                .into_iter().map(String::from).collect()
        );
        
        self.domain_patterns.insert(
            KnowledgeDomain::Security,
            vec!["encryption", "authentication", "vulnerability", "exploit", "firewall", "ssl", "tls"]
                .into_iter().map(String::from).collect()
        );
    }

    /// Analyze metadata to extract semantic features
    pub fn analyze_metadata(&mut self, metadata: &Metadata) -> SemanticFeatures {
        let id = metadata.path.clone();
        
        // Check cache
        if self.config.enable_caching {
            if let Some(cached) = self.feature_cache.get(&id) {
                return cached.clone();
            }
        }
        
        // Extract features
        let topics = self.extract_topics(metadata);
        let domains = self.classify_domains(&topics, &metadata.path);
        let temporal = self.extract_temporal_features(metadata);
        let structural = self.extract_structural_features(metadata);
        let content = self.extract_content_features(metadata);
        let importance_score = self.calculate_importance_score(&topics, &temporal, &structural);
        
        let features = SemanticFeatures {
            id: id.clone(),
            topics,
            domains,
            temporal,
            structural,
            content,
            agent_patterns: None,
            importance_score,
        };
        
        // Cache result
        if self.config.enable_caching {
            self.feature_cache.insert(id, features.clone());
        }
        
        features
    }

    /// Extract topic distribution from metadata
    fn extract_topics(&self, metadata: &Metadata) -> HashMap<String, f32> {
        let mut topics = HashMap::new();
        
        if !self.config.enable_topics {
            return topics;
        }
        
        // Use existing topic counts as base
        for (topic, &count) in &metadata.topic_counts {
            let weight = (count as f32).ln() + 1.0;
            topics.insert(topic.clone(), weight);
        }
        
        // Normalize weights
        let total: f32 = topics.values().sum();
        if total > 0.0 {
            for weight in topics.values_mut() {
                *weight /= total;
            }
        }
        
        topics
    }

    /// Classify content into knowledge domains
    fn classify_domains(&self, topics: &HashMap<String, f32>, path: &str) -> Vec<KnowledgeDomain> {
        let mut domains = Vec::new();
        let mut domain_scores: HashMap<KnowledgeDomain, f32> = HashMap::new();
        
        // Check file extension
        let extension = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        
        match extension {
            "py" => domain_scores.insert(KnowledgeDomain::DataScience, 0.3),
            "rs" => domain_scores.insert(KnowledgeDomain::SystemsProgramming, 0.4),
            "js" | "jsx" | "ts" | "tsx" => domain_scores.insert(KnowledgeDomain::WebDevelopment, 0.4),
            "sql" => domain_scores.insert(KnowledgeDomain::Database, 0.5),
            "cu" | "cuda" => domain_scores.insert(KnowledgeDomain::Engineering, 0.4),
            _ => None,
        };
        
        // Check topic patterns
        for (domain, patterns) in &self.domain_patterns {
            let mut score = 0.0;
            for pattern in patterns {
                if let Some(&weight) = topics.get(pattern) {
                    score += weight;
                }
            }
            if score > 0.0 {
                *domain_scores.entry(domain.clone()).or_insert(0.0) += score;
            }
        }
        
        // Select top domains
        let mut scored_domains: Vec<_> = domain_scores.into_iter().collect();
        scored_domains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for (domain, score) in scored_domains.iter().take(3) {
            if *score > 0.1 {
                domains.push(domain.clone());
            }
        }
        
        if domains.is_empty() {
            domains.push(KnowledgeDomain::Other(extension.to_string()));
        }
        
        domains
    }

    /// Extract temporal features
    fn extract_temporal_features(&self, metadata: &Metadata) -> TemporalFeatures {
        TemporalFeatures {
            created_at: None, // Would need git history
            modified_at: None, // Would need git history
            modification_frequency: metadata.update_count as f32 / 30.0, // Rough estimate
            co_evolution_score: 0.0, // Would need cross-file analysis
            temporal_cluster: None,
        }
    }

    /// Extract structural features
    fn extract_structural_features(&self, metadata: &Metadata) -> StructuralFeatures {
        let path = Path::new(&metadata.path);
        let directory_depth = path.components().count() as u32;
        let file_type = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        StructuralFeatures {
            file_type,
            directory_depth,
            dependency_count: metadata.dependencies.len() as u32,
            complexity_score: (metadata.topic_counts.len() as f32).ln() + 1.0,
            loc: Some(metadata.total_count as u32),
            module_path: path.parent()
                .map(|p| p.to_string_lossy().split('/').map(String::from).collect())
                .unwrap_or_default(),
        }
    }

    /// Extract content-based features
    fn extract_content_features(&self, metadata: &Metadata) -> ContentFeatures {
        let mut key_terms: Vec<_> = metadata.topic_counts.keys().cloned().collect();
        key_terms.sort_by_key(|k| std::cmp::Reverse(metadata.topic_counts[k]));
        key_terms.truncate(20);
        
        ContentFeatures {
            language: self.detect_language(&metadata.path),
            key_terms,
            embeddings: None, // Would require actual content analysis
            content_hash: format!("{:x}", md5::compute(&metadata.path)),
            documentation_score: self.calculate_documentation_score(metadata),
        }
    }

    /// Detect programming/natural language from path
    fn detect_language(&self, path: &str) -> String {
        let extension = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        
        match extension {
            "py" => "Python",
            "rs" => "Rust",
            "js" | "jsx" => "JavaScript",
            "ts" | "tsx" => "TypeScript",
            "java" => "Java",
            "cpp" | "cc" | "cxx" => "C++",
            "c" | "h" => "C",
            "go" => "Go",
            "rb" => "Ruby",
            "php" => "PHP",
            "swift" => "Swift",
            "kt" => "Kotlin",
            "scala" => "Scala",
            "r" => "R",
            "m" => "MATLAB",
            "cu" | "cuda" => "CUDA",
            "md" => "Markdown",
            "txt" => "Text",
            "json" => "JSON",
            "yaml" | "yml" => "YAML",
            "toml" => "TOML",
            "xml" => "XML",
            "html" => "HTML",
            "css" => "CSS",
            "sql" => "SQL",
            _ => "Unknown",
        }.to_string()
    }

    /// Calculate documentation score based on metadata
    fn calculate_documentation_score(&self, metadata: &Metadata) -> f32 {
        let mut score = 0.0;
        
        // Check for documentation-related terms
        let doc_terms = ["readme", "doc", "comment", "description", "example", "usage", "api"];
        for term in doc_terms {
            if metadata.topic_counts.contains_key(term) {
                score += 0.2;
            }
        }
        
        // Markdown files get bonus
        if metadata.path.ends_with(".md") {
            score += 0.3;
        }
        
        score.min(1.0)
    }

    /// Calculate overall importance score
    fn calculate_importance_score(
        &self,
        topics: &HashMap<String, f32>,
        temporal: &TemporalFeatures,
        structural: &StructuralFeatures,
    ) -> f32 {
        let mut score = 0.0;
        
        // Topic diversity
        let topic_entropy = -topics.values()
            .filter(|&&v| v > 0.0)
            .map(|&v| v * v.ln())
            .sum::<f32>();
        score += topic_entropy.min(1.0) * 0.3;
        
        // Modification frequency
        score += temporal.modification_frequency.min(1.0) * 0.2;
        
        // Dependency importance
        score += (structural.dependency_count as f32 / 10.0).min(1.0) * 0.3;
        
        // Complexity
        score += (structural.complexity_score / 5.0).min(1.0) * 0.2;
        
        score.min(1.0)
    }

    /// Analyze agent communication patterns
    pub fn analyze_agent_patterns(
        &mut self,
        agent_id: &str,
        messages: &[(String, String, DateTime<Utc>)], // (from, to, timestamp)
    ) -> AgentCommunicationPatterns {
        let mut send_count = 0;
        let mut receive_count = 0;
        let mut partners: HashMap<String, f32> = HashMap::new();
        let mut message_types = HashSet::new();
        
        for (from, to, _timestamp) in messages {
            if from == agent_id {
                send_count += 1;
                *partners.entry(to.clone()).or_insert(0.0) += 1.0;
            }
            if to == agent_id {
                receive_count += 1;
                *partners.entry(from.clone()).or_insert(0.0) += 1.0;
            }
            // Extract message type from content if available
            message_types.insert("default".to_string());
        }
        
        let total_messages = (send_count + receive_count) as f32;
        let send_frequency = send_count as f32 / total_messages.max(1.0);
        let receive_frequency = receive_count as f32 / total_messages.max(1.0);
        
        // Determine network role
        let degree = partners.len();
        let network_role = if degree == 0 {
            NetworkRole::Isolated
        } else if degree > 10 {
            NetworkRole::Hub
        } else if degree > 5 {
            NetworkRole::Bridge
        } else {
            NetworkRole::Peripheral
        };
        
        AgentCommunicationPatterns {
            send_frequency,
            receive_frequency,
            communication_partners: partners,
            message_types,
            clustering_coefficient: 0.0, // Would need full network analysis
            network_role,
        }
    }

    /// Compute similarity between two semantic features
    pub fn compute_similarity(&self, features1: &SemanticFeatures, features2: &SemanticFeatures) -> f32 {
        let mut similarity = 0.0;
        
        // Topic similarity (cosine similarity)
        let topic_sim = self.cosine_similarity(&features1.topics, &features2.topics);
        similarity += topic_sim * 0.4;
        
        // Domain overlap
        let domain_overlap = features1.domains.iter()
            .filter(|d| features2.domains.contains(d))
            .count() as f32;
        let domain_sim = domain_overlap / (features1.domains.len().max(features2.domains.len()) as f32).max(1.0);
        similarity += domain_sim * 0.2;
        
        // Structural similarity
        if features1.structural.file_type == features2.structural.file_type {
            similarity += 0.1;
        }
        
        let depth_diff = (features1.structural.directory_depth as f32 - features2.structural.directory_depth as f32).abs();
        similarity += (1.0 / (1.0 + depth_diff)) * 0.1;
        
        // Temporal similarity
        let temporal_sim = 1.0 / (1.0 + (features1.temporal.modification_frequency - features2.temporal.modification_frequency).abs());
        similarity += temporal_sim * 0.1;
        
        // Importance similarity
        let importance_diff = (features1.importance_score - features2.importance_score).abs();
        similarity += (1.0 - importance_diff) * 0.1;
        
        similarity.min(1.0)
    }

    /// Compute cosine similarity between topic distributions
    fn cosine_similarity(&self, topics1: &HashMap<String, f32>, topics2: &HashMap<String, f32>) -> f32 {
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        
        let all_topics: HashSet<_> = topics1.keys().chain(topics2.keys()).collect();
        
        for topic in all_topics {
            let v1 = topics1.get(topic.as_str()).unwrap_or(&0.0);
            let v2 = topics2.get(topic.as_str()).unwrap_or(&0.0);
            
            dot_product += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        } else {
            0.0
        }
    }

    /// Get cached features
    pub fn get_cached_features(&self) -> &HashMap<String, SemanticFeatures> {
        &self.feature_cache
    }

    /// Clear feature cache
    pub fn clear_cache(&mut self) {
        self.feature_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_analyzer_creation() {
        let config = SemanticAnalyzerConfig::default();
        let analyzer = SemanticAnalyzer::new(config);
        assert!(analyzer.domain_patterns.len() > 0);
    }

    #[test]
    fn test_domain_classification() {
        let analyzer = SemanticAnalyzer::new(SemanticAnalyzerConfig::default());
        
        let mut topics = HashMap::new();
        topics.insert("neural".to_string(), 0.3);
        topics.insert("network".to_string(), 0.2);
        topics.insert("training".to_string(), 0.4);
        
        let domains = analyzer.classify_domains(&topics, "model.py");
        assert!(domains.contains(&KnowledgeDomain::MachineLearning) || 
                domains.contains(&KnowledgeDomain::DataScience));
    }

    #[test]
    fn test_language_detection() {
        let analyzer = SemanticAnalyzer::new(SemanticAnalyzerConfig::default());
        
        assert_eq!(analyzer.detect_language("test.py"), "Python");
        assert_eq!(analyzer.detect_language("main.rs"), "Rust");
        assert_eq!(analyzer.detect_language("app.js"), "JavaScript");
        assert_eq!(analyzer.detect_language("kernel.cu"), "CUDA");
    }

    #[test]
    fn test_similarity_computation() {
        let analyzer = SemanticAnalyzer::new(SemanticAnalyzerConfig::default());
        
        let mut topics1 = HashMap::new();
        topics1.insert("test".to_string(), 0.5);
        topics1.insert("unit".to_string(), 0.5);
        
        let mut topics2 = HashMap::new();
        topics2.insert("test".to_string(), 0.4);
        topics2.insert("integration".to_string(), 0.6);
        
        let similarity = analyzer.cosine_similarity(&topics1, &topics2);
        assert!(similarity > 0.0 && similarity < 1.0);
    }
}