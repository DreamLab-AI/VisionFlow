//! Advanced edge generation service with multi-modal similarity computation

use crate::models::edge::Edge;
use crate::services::semantic_analyzer::SemanticFeatures;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeGenerationConfig {
    
    pub similarity_threshold: f32,
    
    pub weights: SimilarityWeights,
    
    pub max_edges_per_node: usize,
    
    pub enable_semantic: bool,
    
    pub enable_structural: bool,
    
    pub enable_temporal: bool,
    
    pub enable_agent_communication: bool,
    
    pub enable_pruning: bool,
    
    pub classify_edge_types: bool,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityWeights {
    pub semantic: f32,
    pub structural: f32,
    pub temporal: f32,
    pub communication: f32,
}

impl Default for SimilarityWeights {
    fn default() -> Self {
        Self {
            semantic: 0.4,
            structural: 0.3,
            temporal: 0.2,
            communication: 0.1,
        }
    }
}

impl Default for EdgeGenerationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.1, 
            weights: SimilarityWeights::default(),
            max_edges_per_node: 20,
            enable_semantic: true,
            enable_structural: true,
            enable_temporal: true,
            enable_agent_communication: false,
            enable_pruning: true,
            classify_edge_types: true,
        }
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedEdge {
    
    pub source: String,
    
    pub target: String,
    
    pub weight: f32,
    
    pub semantic_similarity: f32,
    
    pub structural_similarity: f32,
    
    pub temporal_similarity: f32,
    
    pub communication_strength: f32,
    
    pub edge_type: EdgeType,
    
    pub bidirectional: bool,
    
    pub metadata: HashMap<String, serde_json::Value>,
}

///
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EdgeType {
    
    Semantic,
    
    Dependency,
    
    Temporal,
    
    Communication,
    
    Hierarchical,
    
    Reference,
    
    Similarity,
    
    Composite,
}

///
pub struct AdvancedEdgeGenerator {
    config: EdgeGenerationConfig,
    edge_cache: HashMap<(String, String), EnhancedEdge>,
}

impl AdvancedEdgeGenerator {
    
    pub fn new(config: EdgeGenerationConfig) -> Self {
        Self {
            config,
            edge_cache: HashMap::new(),
        }
    }

    
    pub fn generate(&mut self, features: &HashMap<String, SemanticFeatures>) -> Vec<EnhancedEdge> {
        let mut edges = Vec::new();
        let node_ids: Vec<_> = features.keys().cloned().collect();

        
        for i in 0..node_ids.len() {
            let mut node_edges = Vec::new();

            for j in i + 1..node_ids.len() {
                let id1 = &node_ids[i];
                let id2 = &node_ids[j];

                
                let cache_key = (id1.clone(), id2.clone());
                if let Some(cached_edge) = self.edge_cache.get(&cache_key) {
                    node_edges.push(cached_edge.clone());
                    continue;
                }

                
                let features1 = &features[id1];
                let features2 = &features[id2];

                let semantic_sim = if self.config.enable_semantic {
                    self.compute_semantic_similarity(features1, features2)
                } else {
                    0.0
                };

                let structural_sim = if self.config.enable_structural {
                    self.compute_structural_similarity(features1, features2)
                } else {
                    0.0
                };

                let temporal_sim = if self.config.enable_temporal {
                    self.compute_temporal_similarity(features1, features2)
                } else {
                    0.0
                };

                let comm_strength = if self.config.enable_agent_communication {
                    self.compute_communication_strength(features1, features2)
                } else {
                    0.0
                };

                
                let weight = self.compute_weighted_similarity(
                    semantic_sim,
                    structural_sim,
                    temporal_sim,
                    comm_strength,
                );

                
                if weight >= self.config.similarity_threshold {
                    let edge_type = self.classify_edge_type(
                        semantic_sim,
                        structural_sim,
                        temporal_sim,
                        comm_strength,
                        features1,
                        features2,
                    );

                    let edge = EnhancedEdge {
                        source: id1.clone(),
                        target: id2.clone(),
                        weight,
                        semantic_similarity: semantic_sim,
                        structural_similarity: structural_sim,
                        temporal_similarity: temporal_sim,
                        communication_strength: comm_strength,
                        edge_type,
                        bidirectional: true,
                        metadata: HashMap::new(),
                    };

                    
                    self.edge_cache.insert(cache_key, edge.clone());
                    node_edges.push(edge);
                }
            }

            
            node_edges.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
            node_edges.truncate(self.config.max_edges_per_node);
            edges.extend(node_edges);
        }

        
        if self.config.enable_pruning {
            edges = self.prune_redundant_edges(edges);
        }

        edges
    }

    
    fn compute_semantic_similarity(
        &self,
        features1: &SemanticFeatures,
        features2: &SemanticFeatures,
    ) -> f32 {
        let mut similarity = 0.0;

        
        let topic_sim = self.cosine_similarity(&features1.topics, &features2.topics);
        similarity += topic_sim * 0.5;

        
        let domain_overlap = features1
            .domains
            .iter()
            .filter(|d| features2.domains.contains(d))
            .count() as f32;
        let max_domains = features1.domains.len().max(features2.domains.len()) as f32;
        if max_domains > 0.0 {
            similarity += (domain_overlap / max_domains) * 0.3;
        }

        
        let terms1: HashSet<_> = features1.content.key_terms.iter().collect();
        let terms2: HashSet<_> = features2.content.key_terms.iter().collect();
        let intersection = terms1.intersection(&terms2).count() as f32;
        let union = terms1.union(&terms2).count() as f32;
        if union > 0.0 {
            similarity += (intersection / union) * 0.2;
        }

        similarity.min(1.0)
    }

    
    fn compute_structural_similarity(
        &self,
        features1: &SemanticFeatures,
        features2: &SemanticFeatures,
    ) -> f32 {
        let mut similarity = 0.0;

        
        if features1.structural.file_type == features2.structural.file_type {
            similarity += 0.3;
        } else if self.are_compatible_types(
            &features1.structural.file_type,
            &features2.structural.file_type,
        ) {
            similarity += 0.15;
        }

        
        let depth_diff = (features1.structural.directory_depth as i32
            - features2.structural.directory_depth as i32)
            .abs();
        similarity += (1.0 / (1.0 + depth_diff as f32)) * 0.2;

        
        let path_sim = self.compute_path_similarity(
            &features1.structural.module_path,
            &features2.structural.module_path,
        );
        similarity += path_sim * 0.3;

        
        let complexity_diff =
            (features1.structural.complexity_score - features2.structural.complexity_score).abs();
        similarity += (1.0 / (1.0 + complexity_diff)) * 0.1;

        
        if let (Some(loc1), Some(loc2)) = (features1.structural.loc, features2.structural.loc) {
            let size_ratio = (loc1.min(loc2) as f32) / (loc1.max(loc2) as f32).max(1.0);
            similarity += size_ratio * 0.1;
        }

        similarity.min(1.0)
    }

    
    fn compute_temporal_similarity(
        &self,
        features1: &SemanticFeatures,
        features2: &SemanticFeatures,
    ) -> f32 {
        let mut similarity = 0.0;

        
        let freq_diff = (features1.temporal.modification_frequency
            - features2.temporal.modification_frequency)
            .abs();
        similarity += (1.0 / (1.0 + freq_diff)) * 0.4;

        
        let co_evo_avg =
            (features1.temporal.co_evolution_score + features2.temporal.co_evolution_score) / 2.0;
        similarity += co_evo_avg * 0.3;

        
        if let (Some(cluster1), Some(cluster2)) = (
            features1.temporal.temporal_cluster,
            features2.temporal.temporal_cluster,
        ) {
            if cluster1 == cluster2 {
                similarity += 0.3;
            }
        }

        similarity.min(1.0)
    }

    
    fn compute_communication_strength(
        &self,
        features1: &SemanticFeatures,
        features2: &SemanticFeatures,
    ) -> f32 {
        if let (Some(patterns1), Some(patterns2)) =
            (&features1.agent_patterns, &features2.agent_patterns)
        {
            
            let comm1to2 = patterns1
                .communication_partners
                .get(&features2.id)
                .unwrap_or(&0.0);
            let comm2to1 = patterns2
                .communication_partners
                .get(&features1.id)
                .unwrap_or(&0.0);

            
            let direct_comm = (comm1to2 + comm2to1) / 2.0;

            
            let partners1: HashSet<_> = patterns1.communication_partners.keys().collect();
            let partners2: HashSet<_> = patterns2.communication_partners.keys().collect();
            let shared = partners1.intersection(&partners2).count() as f32;
            let total = partners1.union(&partners2).count() as f32;
            let indirect_comm = if total > 0.0 { shared / total } else { 0.0 };

            (direct_comm * 0.7 + indirect_comm * 0.3).min(1.0)
        } else {
            0.0
        }
    }

    
    fn compute_weighted_similarity(
        &self,
        semantic: f32,
        structural: f32,
        temporal: f32,
        communication: f32,
    ) -> f32 {
        let weights = &self.config.weights;
        let total_weight =
            weights.semantic + weights.structural + weights.temporal + weights.communication;

        if total_weight > 0.0 {
            (semantic * weights.semantic
                + structural * weights.structural
                + temporal * weights.temporal
                + communication * weights.communication)
                / total_weight
        } else {
            0.0
        }
    }

    
    fn classify_edge_type(
        &self,
        semantic: f32,
        structural: f32,
        temporal: f32,
        communication: f32,
        features1: &SemanticFeatures,
        features2: &SemanticFeatures,
    ) -> EdgeType {
        if !self.config.classify_edge_types {
            return EdgeType::Composite;
        }

        
        let max_sim = semantic.max(structural).max(temporal).max(communication);

        if communication > 0.5 && communication == max_sim {
            EdgeType::Communication
        } else if semantic == max_sim && semantic > 0.5 {
            EdgeType::Semantic
        } else if structural == max_sim && structural > 0.5 {
            
            if self.is_dependency_relationship(features1, features2) {
                EdgeType::Dependency
            } else if self.is_hierarchical_relationship(features1, features2) {
                EdgeType::Hierarchical
            } else {
                EdgeType::Similarity
            }
        } else if temporal == max_sim && temporal > 0.5 {
            EdgeType::Temporal
        } else if semantic > 0.3 && structural > 0.3 {
            EdgeType::Reference
        } else {
            EdgeType::Composite
        }
    }

    
    fn are_compatible_types(&self, type1: &str, type2: &str) -> bool {
        let web_types = ["js", "jsx", "ts", "tsx", "html", "css"];
        let system_types = ["c", "cpp", "h", "hpp", "rs"];
        let data_types = ["json", "yaml", "xml", "toml"];

        (web_types.contains(&type1) && web_types.contains(&type2))
            || (system_types.contains(&type1) && system_types.contains(&type2))
            || (data_types.contains(&type1) && data_types.contains(&type2))
    }

    
    fn compute_path_similarity(&self, path1: &[String], path2: &[String]) -> f32 {
        if path1.is_empty() || path2.is_empty() {
            return 0.0;
        }

        let mut common_prefix = 0;
        for (p1, p2) in path1.iter().zip(path2.iter()) {
            if p1 == p2 {
                common_prefix += 1;
            } else {
                break;
            }
        }

        let max_len = path1.len().max(path2.len()) as f32;
        common_prefix as f32 / max_len
    }

    
    fn is_dependency_relationship(
        &self,
        _features1: &SemanticFeatures,
        _features2: &SemanticFeatures,
    ) -> bool {
        
        false
    }

    
    fn is_hierarchical_relationship(
        &self,
        features1: &SemanticFeatures,
        features2: &SemanticFeatures,
    ) -> bool {
        
        let path1 = &features1.structural.module_path;
        let path2 = &features2.structural.module_path;

        if path1.len() == path2.len() {
            return false;
        }

        let (shorter, longer) = if path1.len() < path2.len() {
            (path1, path2)
        } else {
            (path2, path1)
        };

        longer.starts_with(shorter.as_slice())
    }

    
    fn prune_redundant_edges(&self, edges: Vec<EnhancedEdge>) -> Vec<EnhancedEdge> {
        if edges.len() <= 2 {
            return edges;
        }

        
        let mut adjacency: HashMap<String, HashSet<String>> = HashMap::new();
        for edge in &edges {
            adjacency
                .entry(edge.source.clone())
                .or_insert_with(HashSet::new)
                .insert(edge.target.clone());
            adjacency
                .entry(edge.target.clone())
                .or_insert_with(HashSet::new)
                .insert(edge.source.clone());
        }

        
        let mut to_remove = HashSet::new();
        for (i, edge) in edges.iter().enumerate() {
            if let Some(source_neighbors) = adjacency.get(&edge.source) {
                if let Some(target_neighbors) = adjacency.get(&edge.target) {
                    let common: Vec<_> = source_neighbors.intersection(target_neighbors).collect();

                    
                    for &common_node in &common {
                        let mut triangle_edges = vec![
                            (&edge.source, &edge.target, edge.weight),
                            (&edge.source, common_node, 0.0),
                            (&edge.target, common_node, 0.0),
                        ];

                        
                        for other_edge in &edges {
                            if (other_edge.source == edge.source
                                && other_edge.target == *common_node)
                                || (other_edge.target == edge.source
                                    && other_edge.source == *common_node)
                            {
                                triangle_edges[1].2 = other_edge.weight;
                            }
                            if (other_edge.source == edge.target
                                && other_edge.target == *common_node)
                                || (other_edge.target == edge.target
                                    && other_edge.source == *common_node)
                            {
                                triangle_edges[2].2 = other_edge.weight;
                            }
                        }

                        
                        if triangle_edges[1].2 > 0.0 && triangle_edges[2].2 > 0.0 {
                            let avg_other = (triangle_edges[1].2 + triangle_edges[2].2) / 2.0;
                            if edge.weight < avg_other * 0.5 {
                                to_remove.insert(i);
                            }
                        }
                    }
                }
            }
        }

        
        edges
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !to_remove.contains(i))
            .map(|(_, edge)| edge)
            .collect()
    }

    
    fn cosine_similarity(
        &self,
        topics1: &HashMap<String, f32>,
        topics2: &HashMap<String, f32>,
    ) -> f32 {
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

    
    pub fn to_basic_edges(
        &self,
        enhanced_edges: Vec<EnhancedEdge>,
        node_id_map: &HashMap<String, u32>,
    ) -> Vec<Edge> {
        enhanced_edges
            .into_iter()
            .filter_map(|edge| {
                
                let source_idx = node_id_map.get(&edge.source)?;
                let target_idx = node_id_map.get(&edge.target)?;

                if edge.bidirectional {
                    Some(vec![
                        Edge::new(*source_idx, *target_idx, edge.weight),
                        Edge::new(*target_idx, *source_idx, edge.weight),
                    ])
                } else {
                    Some(vec![Edge::new(*source_idx, *target_idx, edge.weight)])
                }
            })
            .flatten()
            .collect()
    }

    
    pub fn create_node_id_mapping(node_ids: &[String]) -> HashMap<String, u32> {
        node_ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.clone(), idx as u32))
            .collect()
    }

    
    pub fn generate_with_mapping(
        &mut self,
        features: &HashMap<String, SemanticFeatures>,
    ) -> (Vec<Edge>, HashMap<String, u32>) {
        
        let enhanced_edges = self.generate(features);

        
        let mut all_node_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for edge in &enhanced_edges {
            all_node_ids.insert(edge.source.clone());
            all_node_ids.insert(edge.target.clone());
        }

        
        for node_id in features.keys() {
            all_node_ids.insert(node_id.clone());
        }

        let node_ids: Vec<String> = all_node_ids.into_iter().collect();
        let node_id_map = Self::create_node_id_mapping(&node_ids);

        
        let basic_edges = self.to_basic_edges(enhanced_edges, &node_id_map);

        (basic_edges, node_id_map)
    }

    
    pub fn clear_cache(&mut self) {
        self.edge_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::semantic_analyzer::{
        ContentFeatures, KnowledgeDomain, StructuralFeatures, TemporalFeatures,
    };

    fn create_test_features(id: &str, topics: HashMap<String, f32>) -> SemanticFeatures {
        SemanticFeatures {
            id: id.to_string(),
            topics,
            domains: vec![KnowledgeDomain::ComputerScience],
            temporal: TemporalFeatures {
                created_at: None,
                modified_at: None,
                modification_frequency: 1.0,
                co_evolution_score: 0.5,
                temporal_cluster: Some(1),
            },
            structural: StructuralFeatures {
                file_type: "rs".to_string(),
                directory_depth: 2,
                dependency_count: 5,
                complexity_score: 3.0,
                loc: Some(100),
                module_path: vec!["src".to_string(), "models".to_string()],
            },
            content: ContentFeatures {
                language: "Rust".to_string(),
                key_terms: vec!["graph".to_string(), "node".to_string()],
                embeddings: None,
                content_hash: "abc123".to_string(),
                documentation_score: 0.7,
            },
            agent_patterns: None,
            importance_score: 0.6,
        }
    }

    #[test]
    fn test_edge_generator_creation() {
        let config = EdgeGenerationConfig::default();
        let generator = AdvancedEdgeGenerator::new(config);
        assert!(generator.edge_cache.is_empty());
    }

    #[test]
    fn test_semantic_similarity() {
        let generator = AdvancedEdgeGenerator::new(EdgeGenerationConfig::default());

        let mut topics1 = HashMap::new();
        topics1.insert("graph".to_string(), 0.5);
        topics1.insert("node".to_string(), 0.5);

        let mut topics2 = HashMap::new();
        topics2.insert("graph".to_string(), 0.4);
        topics2.insert("edge".to_string(), 0.6);

        let features1 = create_test_features("file1", topics1);
        let features2 = create_test_features("file2", topics2);

        let similarity = generator.compute_semantic_similarity(&features1, &features2);
        assert!(similarity > 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_structural_similarity() {
        let generator = AdvancedEdgeGenerator::new(EdgeGenerationConfig::default());

        let features1 = create_test_features("file1", HashMap::new());
        let features2 = create_test_features("file2", HashMap::new());

        let similarity = generator.compute_structural_similarity(&features1, &features2);
        assert!(similarity > 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_edge_generation() {
        let mut generator = AdvancedEdgeGenerator::new(EdgeGenerationConfig {
            similarity_threshold: 0.1,
            ..Default::default()
        });

        let mut features = HashMap::new();

        let mut topics1 = HashMap::new();
        topics1.insert("test".to_string(), 0.5);
        features.insert("file1".to_string(), create_test_features("file1", topics1));

        let mut topics2 = HashMap::new();
        topics2.insert("test".to_string(), 0.4);
        features.insert("file2".to_string(), create_test_features("file2", topics2));

        let edges = generator.generate(&features);
        assert!(!edges.is_empty());
        assert_eq!(edges[0].source, "file1");
        assert_eq!(edges[0].target, "file2");
    }

    #[test]
    fn test_edge_type_classification() {
        let generator = AdvancedEdgeGenerator::new(EdgeGenerationConfig::default());

        let features1 = create_test_features("file1", HashMap::new());
        let features2 = create_test_features("file2", HashMap::new());

        let edge_type = generator.classify_edge_type(0.8, 0.2, 0.1, 0.0, &features1, &features2);

        assert_eq!(edge_type, EdgeType::Semantic);
    }
}
