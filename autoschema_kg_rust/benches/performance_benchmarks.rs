//! Performance benchmarks for AutoSchema KG system

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

// Benchmark utilities
mod common {
    use super::*;

    pub fn generate_test_documents(count: usize) -> Vec<String> {
        (0..count)
            .map(|i| {
                format!(
                    "Document {} contains information about Company{} which was founded by Person{} in the year {}. \
                     The company operates in the {} sector and has {} employees.",
                    i,
                    i % 100,
                    i % 50,
                    1950 + (i % 70),
                    ["Technology", "Healthcare", "Finance", "Manufacturing"][i % 4],
                    100 + (i % 10000)
                )
            })
            .collect()
    }

    pub fn generate_test_embeddings(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                (0..dimensions)
                    .map(|j| ((i * j) as f32 * 0.01) % 1.0)
                    .collect()
            })
            .collect()
    }

    pub fn generate_csv_data(rows: usize) -> String {
        let mut csv = String::from("id,name,company,role,salary\n");
        for i in 0..rows {
            csv.push_str(&format!(
                "{},Person{},Company{},Role{},{}\n",
                i,
                i % 1000,
                i % 100,
                i % 20,
                30000 + (i % 100000)
            ));
        }
        csv
    }
}

use common::*;

// Utils benchmarks
fn benchmark_csv_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_processing");

    for size in [100, 1000, 10000].iter() {
        let csv_data = generate_csv_data(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("parse_csv", size),
            &csv_data,
            |b, csv| {
                b.iter(|| {
                    let processor = utils::CsvProcessor::from_string(black_box(csv)).unwrap();
                    let records = processor.parse_all().unwrap();
                    black_box(records)
                })
            },
        );
    }

    group.finish();
}

fn benchmark_json_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_processing");

    for size in [100, 1000, 5000].iter() {
        let json_data = serde_json::json!({
            "items": (0..*size).map(|i| serde_json::json!({
                "id": i,
                "name": format!("Item {}", i),
                "value": i as f64 * 1.5,
                "active": i % 2 == 0
            })).collect::<Vec<_>>()
        }).to_string();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("parse_json", size),
            &json_data,
            |b, json| {
                b.iter(|| {
                    let processor = utils::JsonProcessor::new();
                    let parsed = processor.parse(black_box(json)).unwrap();
                    black_box(parsed)
                })
            },
        );
    }

    group.finish();
}

fn benchmark_text_cleaning(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_cleaning");

    let test_texts = vec![
        ("small", "Hello, World! This is a test.".repeat(10)),
        ("medium", "Hello, World! This is a test.".repeat(100)),
        ("large", "Hello, World! This is a test.".repeat(1000)),
    ];

    for (name, text) in test_texts {
        group.bench_function(format!("normalize_case_{}", name), |b| {
            let cleaner = utils::TextCleaner::new();
            b.iter(|| {
                cleaner.normalize_case(black_box(&text))
            })
        });

        group.bench_function(format!("remove_whitespace_{}", name), |b| {
            let cleaner = utils::TextCleaner::new();
            b.iter(|| {
                cleaner.remove_extra_whitespace(black_box(&text))
            })
        });

        group.bench_function(format!("remove_special_chars_{}", name), |b| {
            let cleaner = utils::TextCleaner::new();
            b.iter(|| {
                cleaner.remove_special_characters(black_box(&text))
            })
        });
    }

    group.finish();
}

fn benchmark_hash_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_operations");

    let test_data = vec![
        ("small", "Hello, World!"),
        ("medium", &"Hello, World! ".repeat(100)),
        ("large", &"Hello, World! ".repeat(1000)),
    ];

    for (name, data) in test_data {
        group.bench_function(format!("sha256_{}", name), |b| {
            b.iter(|| {
                utils::hash_sha256(black_box(data))
            })
        });

        group.bench_function(format!("md5_{}", name), |b| {
            b.iter(|| {
                utils::hash_md5(black_box(data))
            })
        });
    }

    group.finish();
}

// Vector operations benchmarks
fn benchmark_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    for dimensions in [128, 384, 768, 1536].iter() {
        let vector1 = (0..*dimensions).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
        let vector2 = (0..*dimensions).map(|i| (i as f32) * 0.02).collect::<Vec<f32>>();

        group.bench_function(format!("cosine_similarity_{}", dimensions), |b| {
            b.iter(|| {
                cosine_similarity(black_box(&vector1), black_box(&vector2))
            })
        });

        group.bench_function(format!("normalize_vector_{}", dimensions), |b| {
            b.iter(|| {
                let mut vec = vector1.clone();
                normalize_vector(black_box(&mut vec));
                black_box(vec)
            })
        });
    }

    group.finish();
}

fn benchmark_vector_index_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_index");
    group.sample_size(10); // Reduce sample size for expensive operations

    for dimensions in [128, 384].iter() {
        let vectors = generate_test_embeddings(1000, *dimensions);

        group.bench_function(format!("build_index_{}", dimensions), |b| {
            b.iter(|| {
                let config = MockVectorIndexConfig::new(*dimensions, "hnsw");
                let mut index = MockVectorIndex::new(config);

                for (i, vector) in vectors.iter().enumerate() {
                    index.add_vector(&format!("doc_{}", i), vector);
                }

                black_box(index)
            })
        });

        // Benchmark search performance
        let config = MockVectorIndexConfig::new(*dimensions, "hnsw");
        let mut index = MockVectorIndex::new(config);

        for (i, vector) in vectors.iter().take(100).enumerate() {
            index.add_vector(&format!("doc_{}", i), vector);
        }

        let query = &vectors[0];

        group.bench_function(format!("search_index_{}", dimensions), |b| {
            b.iter(|| {
                index.search(black_box(query), black_box(10))
            })
        });
    }

    group.finish();
}

// LLM operations benchmarks
fn benchmark_token_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_counting");

    let test_texts = vec![
        ("small", "Hello, World!"),
        ("medium", &"The quick brown fox jumps over the lazy dog. ".repeat(50)),
        ("large", &"The quick brown fox jumps over the lazy dog. ".repeat(500)),
    ];

    for (name, text) in test_texts {
        group.bench_function(format!("count_tokens_{}", name), |b| {
            let counter = MockTokenCounter::new("gpt-3.5-turbo");
            b.iter(|| {
                counter.count_tokens(black_box(text))
            })
        });
    }

    group.finish();
}

fn benchmark_prompt_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_processing");

    let template = "Hello {{name}}, you are {{age}} years old and work at {{company}}. Today is {{date}}.";
    let variables = std::collections::HashMap::from([
        ("name".to_string(), "Alice".to_string()),
        ("age".to_string(), "30".to_string()),
        ("company".to_string(), "TechCorp".to_string()),
        ("date".to_string(), "2024-01-15".to_string()),
    ]);

    group.bench_function("render_template", |b| {
        let prompt_template = MockPromptTemplate::new(template);
        b.iter(|| {
            prompt_template.render(black_box(&variables))
        })
    });

    group.finish();
}

// Knowledge graph benchmarks
fn benchmark_kg_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("knowledge_graph");
    group.sample_size(10);

    for size in [100, 500, 1000].iter() {
        group.bench_function(format!("build_graph_{}", size), |b| {
            b.iter(|| {
                let mut kg = MockKnowledgeGraph::new();

                // Add entities
                for i in 0..*size {
                    let entity = MockEntity::new(&format!("Entity{}", i), "TEST", 0.9);
                    kg.add_entity(entity);
                }

                // Add relationships (create a connected graph)
                for i in 0..(*size / 2) {
                    let rel = MockRelationship::new(
                        &format!("Entity{}", i),
                        "connected_to",
                        &format!("Entity{}", (i + 1) % size),
                        0.8,
                    );
                    kg.add_relationship(rel);
                }

                black_box(kg)
            })
        });
    }

    // Benchmark graph traversal
    let mut kg = MockKnowledgeGraph::new();
    for i in 0..1000 {
        let entity = MockEntity::new(&format!("Entity{}", i), "TEST", 0.9);
        kg.add_entity(entity);
    }

    for i in 0..500 {
        let rel = MockRelationship::new(
            &format!("Entity{}", i),
            "connected_to",
            &format!("Entity{}", (i + 1) % 1000),
            0.8,
        );
        kg.add_relationship(rel);
    }

    for hops in [1, 2, 3, 5].iter() {
        group.bench_function(format!("traverse_{}hops", hops), |b| {
            b.iter(|| {
                kg.get_neighbors(black_box("Entity0"), black_box(*hops))
            })
        });
    }

    group.finish();
}

// Integration pipeline benchmarks
fn benchmark_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for doc_count in [10, 50, 100].iter() {
        let documents = generate_test_documents(*doc_count);

        group.throughput(Throughput::Elements(*doc_count as u64));
        group.bench_function(format!("process_documents_{}", doc_count), |b| {
            b.iter(|| {
                // Simulate full pipeline processing
                let mut total_entities = 0;
                let mut total_relationships = 0;

                for doc in black_box(&documents) {
                    // Simulate entity extraction
                    let entities = simulate_entity_extraction(doc);
                    total_entities += entities.len();

                    // Simulate relationship extraction
                    let relationships = simulate_relationship_extraction(doc, &entities);
                    total_relationships += relationships.len();

                    // Simulate embedding generation
                    let _embedding = simulate_embedding_generation(doc);
                }

                black_box((total_entities, total_relationships))
            })
        });
    }

    group.finish();
}

// Helper functions for benchmarks
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        0.0
    } else {
        dot_product / (magnitude_a * magnitude_b)
    }
}

fn normalize_vector(vector: &mut [f32]) {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for x in vector.iter_mut() {
            *x /= magnitude;
        }
    }
}

fn simulate_entity_extraction(text: &str) -> Vec<String> {
    // Simple simulation: extract words that start with capital letters
    text.split_whitespace()
        .filter(|word| word.chars().next().map_or(false, |c| c.is_uppercase()))
        .map(|s| s.to_string())
        .collect()
}

fn simulate_relationship_extraction(text: &str, entities: &[String]) -> Vec<(String, String, String)> {
    // Simple simulation: create relationships between consecutive entities
    let mut relationships = Vec::new();
    for i in 0..entities.len().saturating_sub(1) {
        relationships.push((
            entities[i].clone(),
            "related_to".to_string(),
            entities[i + 1].clone(),
        ));
    }
    relationships
}

fn simulate_embedding_generation(_text: &str) -> Vec<f32> {
    // Generate a mock 384-dimensional embedding
    (0..384).map(|i| (i as f32) * 0.01).collect()
}

// Mock types for benchmarking
struct MockVectorIndexConfig {
    dimensions: usize,
}

impl MockVectorIndexConfig {
    fn new(dimensions: usize, _index_type: &str) -> Self {
        Self { dimensions }
    }
}

struct MockVectorIndex {
    vectors: std::collections::HashMap<String, Vec<f32>>,
    dimensions: usize,
}

impl MockVectorIndex {
    fn new(config: MockVectorIndexConfig) -> Self {
        Self {
            vectors: std::collections::HashMap::new(),
            dimensions: config.dimensions,
        }
    }

    fn add_vector(&mut self, id: &str, vector: &[f32]) {
        self.vectors.insert(id.to_string(), vector.to_vec());
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut results: Vec<_> = self
            .vectors
            .iter()
            .map(|(id, vec)| (id.clone(), cosine_similarity(query, vec)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(k).collect()
    }
}

struct MockTokenCounter {
    _model: String,
}

impl MockTokenCounter {
    fn new(model: &str) -> Self {
        Self {
            _model: model.to_string(),
        }
    }

    fn count_tokens(&self, text: &str) -> usize {
        // Simple approximation: 4 characters per token
        (text.len() / 4).max(1)
    }
}

struct MockPromptTemplate {
    template: String,
}

impl MockPromptTemplate {
    fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
        }
    }

    fn render(&self, variables: &std::collections::HashMap<String, String>) -> String {
        let mut result = self.template.clone();
        for (key, value) in variables {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }
        result
    }
}

struct MockEntity {
    text: String,
    entity_type: String,
    confidence: f64,
}

impl MockEntity {
    fn new(text: &str, entity_type: &str, confidence: f64) -> Self {
        Self {
            text: text.to_string(),
            entity_type: entity_type.to_string(),
            confidence,
        }
    }
}

struct MockRelationship {
    subject: String,
    predicate: String,
    object: String,
    confidence: f64,
}

impl MockRelationship {
    fn new(subject: &str, predicate: &str, object: &str, confidence: f64) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence,
        }
    }
}

struct MockKnowledgeGraph {
    entities: std::collections::HashMap<String, MockEntity>,
    relationships: Vec<MockRelationship>,
}

impl MockKnowledgeGraph {
    fn new() -> Self {
        Self {
            entities: std::collections::HashMap::new(),
            relationships: Vec::new(),
        }
    }

    fn add_entity(&mut self, entity: MockEntity) {
        self.entities.insert(entity.text.clone(), entity);
    }

    fn add_relationship(&mut self, relationship: MockRelationship) {
        self.relationships.push(relationship);
    }

    fn get_neighbors(&self, start: &str, max_hops: usize) -> Vec<String> {
        let mut visited = std::collections::HashSet::new();
        let mut current_level = std::collections::HashSet::new();
        current_level.insert(start.to_string());
        visited.insert(start.to_string());

        let mut neighbors = Vec::new();

        for _hop in 0..max_hops {
            let mut next_level = std::collections::HashSet::new();

            for node in &current_level {
                for rel in &self.relationships {
                    if &rel.subject == node && !visited.contains(&rel.object) {
                        next_level.insert(rel.object.clone());
                        neighbors.push(rel.object.clone());
                        visited.insert(rel.object.clone());
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }
            current_level = next_level;
        }

        neighbors
    }
}

// Benchmark groups
criterion_group!(
    utils_benches,
    benchmark_csv_processing,
    benchmark_json_processing,
    benchmark_text_cleaning,
    benchmark_hash_operations
);

criterion_group!(
    vector_benches,
    benchmark_vector_operations,
    benchmark_vector_index_operations
);

criterion_group!(
    llm_benches,
    benchmark_token_counting,
    benchmark_prompt_processing
);

criterion_group!(
    kg_benches,
    benchmark_kg_operations
);

criterion_group!(
    pipeline_benches,
    benchmark_full_pipeline
);

criterion_main!(
    utils_benches,
    vector_benches,
    llm_benches,
    kg_benches,
    pipeline_benches
);