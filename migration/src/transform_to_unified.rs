use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};

// Import types from export scripts
#[derive(Debug, Deserialize)]
struct KnowledgeGraphExport {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    clustering: Vec<ClusterResult>,
    pathfinding_cache: Vec<PathfindingCache>,
    #[allow(dead_code)]
    nodes_sha1: String,
    #[allow(dead_code)]
    edges_sha1: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct GraphNode {
    id: i64,
    metadata_id: String,
    label: String,
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    mass: f64,
    node_type: Option<String>,
    category: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GraphEdge {
    id: i64,
    source_id: i64,
    target_id: i64,
    relationship_type: String,
    weight: f64,
    metadata: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClusterResult {
    node_id: i64,
    cluster_id: i32,
    algorithm: String,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PathfindingCache {
    source_id: i64,
    target_id: i64,
    path_nodes: String,
    distance: f64,
    cached_at: String,
}

#[derive(Debug, Deserialize)]
struct OntologyExport {
    owl_classes: Vec<OwlClass>,
    owl_axioms: Vec<OwlAxiom>,
    owl_properties: Vec<OwlProperty>,
    reasoning_cache: Vec<ReasoningCache>,
    #[allow(dead_code)]
    classes_sha1: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OwlClass {
    id: i64,
    iri: String,
    label: Option<String>,
    parent_class_iri: Option<String>,
    markdown_content: Option<String>,
    file_path: Option<String>,
    file_sha1: Option<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OwlAxiom {
    id: i64,
    axiom_type: String,
    subject_iri: Option<String>,
    object_iri: Option<String>,
    property_iri: Option<String>,
    strength: f64,
    priority: i32,
    user_defined: bool,
    created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OwlProperty {
    id: i64,
    iri: String,
    property_type: String,
    label: Option<String>,
    domain_iri: Option<String>,
    range_iri: Option<String>,
    functional: bool,
    inverse_functional: bool,
    transitive: bool,
    symmetric: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReasoningCache {
    ontology_checksum: String,
    inferred_axiom_type: String,
    subject_iri: String,
    object_iri: String,
    cached_at: String,
}

// Unified schema types
#[derive(Debug, Serialize)]
struct UnifiedNode {
    id: i64,
    metadata_id: String,
    label: String,
    // Physics state (from knowledge graph)
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    mass: f64,
    // Ontology linkage
    owl_class_iri: Option<String>,
    // Metadata
    node_type: Option<String>,
    category: Option<String>,
}

#[derive(Debug, Serialize)]
struct TransformedData {
    transform_timestamp: String,
    unified_nodes: Vec<UnifiedNode>,
    unified_edges: Vec<GraphEdge>,
    owl_classes: Vec<OwlClass>,
    owl_axioms: Vec<OwlAxiom>,
    owl_properties: Vec<OwlProperty>,
    clustering: Vec<ClusterResult>,
    pathfinding_cache: Vec<PathfindingCache>,
    reasoning_cache: Vec<ReasoningCache>,
    // Mapping statistics
    stats: TransformStats,
    checksum: String,
}

#[derive(Debug, Clone, Serialize)]
struct TransformStats {
    original_nodes: usize,
    original_owl_classes: usize,
    matched_nodes: usize,
    merged_duplicates: usize,
    unmatched_nodes: usize,
    overlap_percentage: f64,
}

/// Match graph nodes to OWL classes via metadata_id
fn match_nodes_to_classes(
    nodes: &[GraphNode],
    owl_classes: &[OwlClass],
) -> (Vec<UnifiedNode>, TransformStats) {
    println!("\n🔗 Matching nodes to OWL classes...");

    // Build IRI lookup by label (simplified matching)
    let mut iri_by_label: HashMap<String, String> = HashMap::new();
    for owl_class in owl_classes {
        if let Some(label) = &owl_class.label {
            iri_by_label.insert(label.to_lowercase(), owl_class.iri.clone());
        }
        // Also index by IRI basename
        if let Some(basename) = owl_class.iri.split('/').last() {
            iri_by_label.insert(basename.to_lowercase(), owl_class.iri.clone());
        }
    }

    let mut unified_nodes = Vec::new();
    let mut matched = 0;
    let mut unmatched = 0;
    let mut seen_metadata_ids = HashSet::new();
    let mut merged_duplicates = 0;

    for node in nodes {
        // Skip duplicates (40-60% overlap)
        if seen_metadata_ids.contains(&node.metadata_id) {
            merged_duplicates += 1;
            continue;
        }
        seen_metadata_ids.insert(node.metadata_id.clone());

        // Try to match to OWL class
        let owl_class_iri = iri_by_label
            .get(&node.label.to_lowercase())
            .or_else(|| iri_by_label.get(&node.metadata_id.to_lowercase()))
            .cloned();

        if owl_class_iri.is_some() {
            matched += 1;
        } else {
            unmatched += 1;
        }

        unified_nodes.push(UnifiedNode {
            id: node.id,
            metadata_id: node.metadata_id.clone(),
            label: node.label.clone(),
            x: node.x,
            y: node.y,
            z: node.z,
            vx: node.vx,
            vy: node.vy,
            vz: node.vz,
            mass: node.mass,
            owl_class_iri,
            node_type: node.node_type.clone(),
            category: node.category.clone(),
        });
    }

    let stats = TransformStats {
        original_nodes: nodes.len(),
        original_owl_classes: owl_classes.len(),
        matched_nodes: matched,
        merged_duplicates,
        unmatched_nodes: unmatched,
        overlap_percentage: (merged_duplicates as f64 / nodes.len() as f64) * 100.0,
    };

    println!("   ✅ Matched: {} nodes to OWL classes", matched);
    println!("   ⚠️  Unmatched: {} nodes (will be orphaned)", unmatched);
    println!("   🔄 Merged duplicates: {} ({:.1}% overlap)",
             merged_duplicates, stats.overlap_percentage);

    (unified_nodes, stats)
}

fn compute_sha1<T: Serialize>(data: &T) -> Result<String> {
    let json = serde_json::to_string(data)?;
    let mut hasher = Sha1::new();
    hasher.update(json.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 VisionFlow Unified Transform");
    println!("================================\n");

    // Load knowledge graph export
    let kg_path = "/home/devuser/workspace/project/migration/knowledge_graph_export.json";
    println!("📖 Loading knowledge graph export from: {}", kg_path);
    let mut kg_file = File::open(kg_path)
        .context("Failed to open knowledge graph export")?;
    let mut kg_json = String::new();
    kg_file.read_to_string(&mut kg_json)?;
    let kg_export: KnowledgeGraphExport = serde_json::from_str(&kg_json)
        .context("Failed to parse knowledge graph export")?;
    println!("   ✅ Loaded {} nodes, {} edges",
             kg_export.nodes.len(), kg_export.edges.len());

    // Load ontology export
    let ont_path = "/home/devuser/workspace/project/migration/ontology_export.json";
    println!("📖 Loading ontology export from: {}", ont_path);
    let mut ont_file = File::open(ont_path)
        .context("Failed to open ontology export")?;
    let mut ont_json = String::new();
    ont_file.read_to_string(&mut ont_json)?;
    let ont_export: OntologyExport = serde_json::from_str(&ont_json)
        .context("Failed to parse ontology export")?;
    println!("   ✅ Loaded {} OWL classes, {} axioms",
             ont_export.owl_classes.len(), ont_export.owl_axioms.len());

    // Transform: Match nodes to OWL classes
    let (unified_nodes, stats) = match_nodes_to_classes(&kg_export.nodes, &ont_export.owl_classes);

    // Validate referential integrity
    println!("\n🔍 Validating referential integrity...");
    let node_ids: HashSet<i64> = unified_nodes.iter().map(|n| n.id).collect();
    let mut invalid_edges = 0;
    for edge in &kg_export.edges {
        if !node_ids.contains(&edge.source_id) || !node_ids.contains(&edge.target_id) {
            invalid_edges += 1;
        }
    }
    if invalid_edges > 0 {
        println!("   ⚠️  Warning: {} edges reference non-existent nodes", invalid_edges);
    } else {
        println!("   ✅ All edges valid");
    }

    // Create transformed data structure
    let transformed = TransformedData {
        transform_timestamp: format!("{}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")),
        unified_nodes,
        unified_edges: kg_export.edges,
        owl_classes: ont_export.owl_classes,
        owl_axioms: ont_export.owl_axioms,
        owl_properties: ont_export.owl_properties,
        clustering: kg_export.clustering,
        pathfinding_cache: kg_export.pathfinding_cache,
        reasoning_cache: ont_export.reasoning_cache,
        stats: stats.clone(),
        checksum: String::new(), // Computed after serialization
    };

    // Compute checksum
    let checksum = compute_sha1(&transformed)?;
    let mut transformed_with_checksum = transformed;
    transformed_with_checksum.checksum = checksum.clone();

    // Write transformed data
    let output_path = "/home/devuser/workspace/project/migration/unified_transform.json";
    println!("\n💾 Writing transformed data to: {}", output_path);

    let json = serde_json::to_string_pretty(&transformed_with_checksum)
        .context("Failed to serialize transformed data")?;

    let mut file = File::create(output_path)
        .context("Failed to create output file")?;

    file.write_all(json.as_bytes())
        .context("Failed to write output file")?;

    println!("\n✅ Transform complete!");
    println!("   Checksum: {}", checksum);
    println!("   Output: {}", output_path);
    println!("   Size: {} bytes", json.len());
    println!("\n📊 Statistics:");
    println!("   Original nodes: {}", stats.original_nodes);
    println!("   Unified nodes: {}", stats.original_nodes - stats.merged_duplicates);
    println!("   Matched to OWL: {}", stats.matched_nodes);
    println!("   Overlap: {:.1}%", stats.overlap_percentage);

    Ok(())
}
