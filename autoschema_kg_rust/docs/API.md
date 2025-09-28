# AutoSchema KG Rust API Documentation

This document provides comprehensive API documentation for all modules in the AutoSchema KG Rust project.

## Table of Contents

- [Knowledge Graph Construction](#knowledge-graph-construction)
- [LLM Generator](#llm-generator)
- [Retriever](#retriever)
- [Vector Store](#vectorstore)
- [Utilities](#utilities)
- [Error Handling](#error-handling)
- [Configuration](#configuration)

## Knowledge Graph Construction

The `kg_construction` module provides tools for building and managing knowledge graphs with Neo4j integration.

### Core Types

#### `KnowledgeGraphBuilder`

Main interface for knowledge graph construction operations.

```rust
use kg_construction::{KnowledgeGraphBuilder, GraphConfig};

// Create a new builder instance
let config = GraphConfig::builder()
    .neo4j_uri("bolt://localhost:7687")
    .username("neo4j")
    .password("password")
    .max_connections(10)
    .connection_timeout(Duration::from_secs(30))
    .build();

let builder = KnowledgeGraphBuilder::new(config).await?;
```

#### Methods

##### `process_dataset`

Process a dataset and construct knowledge graph.

```rust
pub async fn process_dataset(
    &self,
    dataset_path: &Path,
    config: ProcessingConfig,
) -> Result<ProcessingStats, KgError>
```

**Parameters:**
- `dataset_path`: Path to the dataset file (CSV, JSON, or GraphML)
- `config`: Processing configuration options

**Returns:**
- `ProcessingStats`: Statistics about the processing operation

**Example:**
```rust
let config = ProcessingConfig::builder()
    .batch_size(1000)
    .parallel_workers(4)
    .enable_validation(true)
    .deduplicate_entities(true)
    .build();

let stats = builder.process_dataset(
    Path::new("data/entities.csv"),
    config
).await?;

println!("Processed {} entities and {} relationships",
         stats.entities_processed,
         stats.relationships_created);
```

##### `add_entity`

Add a single entity to the knowledge graph.

```rust
pub async fn add_entity(
    &self,
    entity: Entity,
) -> Result<String, KgError>
```

**Example:**
```rust
use kg_construction::{Entity, EntityType};

let entity = Entity::builder()
    .id("person_123")
    .entity_type(EntityType::Person)
    .name("John Doe")
    .property("age", 30)
    .property("email", "john@example.com")
    .build();

let entity_id = builder.add_entity(entity).await?;
```

##### `add_relationship`

Create a relationship between two entities.

```rust
pub async fn add_relationship(
    &self,
    relationship: Relationship,
) -> Result<String, KgError>
```

**Example:**
```rust
use kg_construction::{Relationship, RelationType};

let relationship = Relationship::builder()
    .from_entity("person_123")
    .to_entity("company_456")
    .relationship_type(RelationType::WorksFor)
    .property("start_date", "2020-01-01")
    .property("position", "Software Engineer")
    .build();

let rel_id = builder.add_relationship(relationship).await?;
```

### Schema Management

#### `SchemaInferrer`

Automatically infer schema from data.

```rust
use kg_construction::SchemaInferrer;

let inferrer = SchemaInferrer::new();
let schema = inferrer.infer_from_file("data/sample.csv").await?;

println!("Detected {} entity types and {} relationship types",
         schema.entity_types.len(),
         schema.relationship_types.len());
```

#### `SchemaValidator`

Validate data against a schema.

```rust
use kg_construction::{SchemaValidator, ValidationConfig};

let validator = SchemaValidator::new(schema);
let config = ValidationConfig::builder()
    .strict_mode(true)
    .allow_unknown_types(false)
    .build();

let validation_result = validator.validate_dataset(
    "data/entities.csv",
    config
).await?;

if !validation_result.is_valid() {
    for error in validation_result.errors {
        eprintln!("Validation error: {}", error);
    }
}
```

## LLM Generator

The `llm_generator` module provides unified interface for multiple LLM providers.

### Core Types

#### `LLMGenerator`

Main interface for LLM generation operations.

```rust
use llm_generator::{LLMGenerator, OpenAIProvider, GenerationConfig};

let provider = OpenAIProvider::new("your-api-key");
let config = GenerationConfig::builder()
    .model("gpt-4")
    .temperature(0.1)
    .max_tokens(2048)
    .top_p(0.9)
    .frequency_penalty(0.0)
    .presence_penalty(0.0)
    .build();

let generator = LLMGenerator::new(provider, config);
```

#### Methods

##### `generate`

Generate text from a prompt.

```rust
pub async fn generate(
    &self,
    prompt: &str,
) -> Result<GenerationResponse, LLMError>
```

**Example:**
```rust
let prompt = "Explain the concept of knowledge graphs in simple terms.";
let response = generator.generate(prompt).await?;

println!("Response: {}", response.text);
println!("Tokens used: {}", response.usage.total_tokens);
```

##### `generate_with_context`

Generate text with additional context.

```rust
pub async fn generate_with_context(
    &self,
    prompt: &str,
    context: &[String],
) -> Result<GenerationResponse, LLMError>
```

**Example:**
```rust
let context = vec![
    "Neo4j is a graph database".to_string(),
    "Graphs consist of nodes and relationships".to_string(),
];

let response = generator.generate_with_context(
    "How do graph databases work?",
    &context
).await?;
```

##### `generate_stream`

Generate text with streaming response.

```rust
pub async fn generate_stream(
    &self,
    prompt: &str,
) -> Result<impl Stream<Item = Result<String, LLMError>>, LLMError>
```

**Example:**
```rust
use futures::StreamExt;

let mut stream = generator.generate_stream("Write a story about AI").await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(text) => print!("{}", text),
        Err(e) => eprintln!("Stream error: {}", e),
    }
}
```

### Providers

#### `OpenAIProvider`

OpenAI GPT models integration.

```rust
use llm_generator::OpenAIProvider;

let provider = OpenAIProvider::builder()
    .api_key("your-api-key")
    .base_url("https://api.openai.com/v1") // Optional custom base URL
    .organization("your-org-id") // Optional organization ID
    .timeout(Duration::from_secs(60))
    .max_retries(3)
    .build();
```

#### `AnthropicProvider`

Anthropic Claude models integration.

```rust
use llm_generator::AnthropicProvider;

let provider = AnthropicProvider::builder()
    .api_key("your-api-key")
    .version("2023-06-01")
    .timeout(Duration::from_secs(60))
    .build();
```

#### `LocalProvider`

Local model integration (Ollama, etc.).

```rust
use llm_generator::LocalProvider;

let provider = LocalProvider::builder()
    .endpoint("http://localhost:11434")
    .model("llama2")
    .timeout(Duration::from_secs(120))
    .build();
```

### Batch Processing

#### `BatchProcessor`

Process multiple requests in batches.

```rust
use llm_generator::{BatchProcessor, BatchRequest};

let processor = BatchProcessor::new(generator, 10); // Batch size of 10

let requests = vec![
    BatchRequest::new("1", "What is AI?"),
    BatchRequest::new("2", "Explain machine learning"),
    BatchRequest::new("3", "What are neural networks?"),
];

let results = processor.process_batch(requests).await?;

for result in results {
    println!("Request {}: {}", result.id, result.response.text);
}
```

### Rate Limiting

#### `RateLimiter`

Control request rate to APIs.

```rust
use llm_generator::{RateLimiter, RateConfig};

let rate_config = RateConfig::builder()
    .requests_per_minute(60)
    .tokens_per_minute(50000)
    .concurrent_requests(5)
    .build();

let rate_limiter = RateLimiter::new(rate_config);

// Use with generator
let generator_with_limits = generator.with_rate_limiter(rate_limiter);
```

## Retriever

The `retriever` module provides advanced retrieval capabilities for RAG systems.

### Core Types

#### `MultiHopRetriever`

Main retrieval interface with graph traversal capabilities.

```rust
use retriever::{MultiHopRetriever, RetrieverConfig};

let config = RetrieverConfig::builder()
    .max_hops(3)
    .top_k(10)
    .similarity_threshold(0.7)
    .enable_reranking(true)
    .cache_size(1000)
    .build();

let retriever = MultiHopRetriever::new(
    config,
    vector_store,
    graph_client,
).await?;
```

#### Methods

##### `retrieve`

Perform retrieval with optional context.

```rust
pub async fn retrieve(
    &self,
    query: &str,
    context: Option<&str>,
) -> Result<Vec<RetrievalResult>, RetrieverError>
```

**Example:**
```rust
let query = "What are the applications of machine learning in healthcare?";
let results = retriever.retrieve(query, None).await?;

for result in results {
    println!("Score: {:.3}, Text: {}", result.score, result.text);
    println!("Source: {} (Hop: {})", result.source, result.hop_count);
}
```

##### `multi_hop_retrieve`

Perform multi-hop graph traversal retrieval.

```rust
pub async fn multi_hop_retrieve(
    &self,
    query: &str,
    starting_entities: Vec<String>,
    max_hops: usize,
) -> Result<MultiHopResult, RetrieverError>
```

**Example:**
```rust
let starting_entities = vec!["machine_learning".to_string()];
let result = retriever.multi_hop_retrieve(
    "healthcare applications",
    starting_entities,
    3
).await?;

println!("Found {} paths across {} hops",
         result.paths.len(),
         result.max_hops_reached);
```

### Query Processing

#### `QueryExpander`

Expand queries for better retrieval.

```rust
use retriever::QueryExpander;

let expander = QueryExpander::new(llm_generator);
let expanded = expander.expand_query(
    "machine learning healthcare",
    3 // Number of expansions
).await?;

println!("Original: machine learning healthcare");
println!("Expanded: {:?}", expanded);
```

#### `QueryRewriter`

Rewrite queries for optimization.

```rust
use retriever::QueryRewriter;

let rewriter = QueryRewriter::new();
let rewritten = rewriter.rewrite_for_graph_search(
    "What are ML applications in medicine?"
).await?;

println!("Rewritten: {}", rewritten);
```

### Ranking

#### `RankingModel`

Rank and score retrieval results.

```rust
use retriever::{RankingModel, RankingStrategy};

let ranking_model = RankingModel::new(RankingStrategy::Fusion);
let ranked_results = ranking_model.rank_results(
    query,
    initial_results,
    context
).await?;
```

## Vector Store

The `vectorstore` module provides high-performance vector storage and search.

### Core Types

#### `VectorStore`

Main vector storage interface.

```rust
use vectorstore::{VectorStore, EmbeddingConfig};

let config = EmbeddingConfig::builder()
    .model("sentence-transformers/all-MiniLM-L6-v2")
    .dimension(384)
    .device("cuda") // or "cpu"
    .batch_size(32)
    .build();

let store = VectorStore::new(config).await?;
```

#### Methods

##### `add_documents`

Add documents to the vector store.

```rust
pub async fn add_documents(
    &mut self,
    documents: Vec<Document>,
) -> Result<Vec<String>, VectorStoreError>
```

**Example:**
```rust
use vectorstore::Document;

let documents = vec![
    Document::new("doc1", "Machine learning is a subset of AI"),
    Document::new("doc2", "Neural networks are inspired by the brain"),
    Document::new("doc3", "Deep learning uses multiple layers"),
];

let doc_ids = store.add_documents(documents).await?;
```

##### `search`

Search for similar documents.

```rust
pub async fn search(
    &self,
    query: &str,
    top_k: usize,
    threshold: Option<f32>,
) -> Result<Vec<SearchResult>, VectorStoreError>
```

**Example:**
```rust
let results = store.search(
    "artificial intelligence applications",
    5,
    Some(0.7)
).await?;

for result in results {
    println!("Score: {:.3}, Doc: {}", result.score, result.document_id);
}
```

##### `batch_search`

Perform multiple searches in batch.

```rust
let queries = vec![
    "machine learning",
    "neural networks",
    "deep learning",
];

let batch_results = store.batch_search(queries, 5).await?;
```

### Embedding Models

#### `EmbeddingModel`

Interface for different embedding models.

```rust
use vectorstore::{EmbeddingModel, SentenceTransformerModel};

// Sentence Transformers model
let model = SentenceTransformerModel::new(
    "sentence-transformers/all-MiniLM-L6-v2"
).await?;

// Custom model
let embeddings = model.encode_batch(&texts).await?;
```

### Index Management

#### `VectorIndex`

Manage vector indices for fast search.

```rust
use vectorstore::{VectorIndex, IndexConfig};

let index_config = IndexConfig::builder()
    .index_type("hnsw")
    .m(16)
    .ef_construction(200)
    .ef_search(100)
    .build();

let index = VectorIndex::new(index_config);
```

## Utilities

The `utils` module provides comprehensive data processing utilities.

### File Processing

#### CSV Processing

```rust
use utils::csv_processing::{CsvProcessor, CsvConfig};

let config = CsvConfig::builder()
    .delimiter(',')
    .has_headers(true)
    .quote_char('"')
    .escape_char('\\')
    .build();

let processor = CsvProcessor::new(config);
let records = processor.process_file("data.csv").await?;
```

#### JSON Processing

```rust
use utils::json_processing::{JsonProcessor, JsonConfig};

let processor = JsonProcessor::new(JsonConfig::default());
let data = processor.process_file("data.json").await?;
```

#### GraphML Processing

```rust
use utils::graph_conversion::{GraphMLProcessor, GraphConfig};

let processor = GraphMLProcessor::new();
let graph_data = processor.convert_to_kg_format("graph.graphml").await?;
```

### Text Processing

#### Text Cleaning

```rust
use utils::text_cleaning::{TextCleaner, CleaningConfig};

let config = CleaningConfig::builder()
    .remove_html(true)
    .normalize_whitespace(true)
    .remove_special_chars(false)
    .lowercase(true)
    .build();

let cleaner = TextCleaner::new(config);
let cleaned = cleaner.clean_text("  <p>Hello   World!</p>  ");
```

### Hash Utilities

```rust
use utils::hash_utils::{HashGenerator, HashAlgorithm};

let generator = HashGenerator::new(HashAlgorithm::Sha256);
let hash = generator.hash_text("content to hash");
let hash_hex = generator.hash_text_hex("content to hash");
```

## Error Handling

All modules use comprehensive error types with detailed information.

### Common Error Types

```rust
use kg_construction::KgError;
use llm_generator::LLMError;
use retriever::RetrieverError;
use vectorstore::VectorStoreError;
use utils::UtilsError;

// Error handling pattern
match result {
    Ok(value) => println!("Success: {:?}", value),
    Err(KgError::ConnectionFailed(msg)) => {
        eprintln!("Database connection failed: {}", msg);
    },
    Err(KgError::ValidationFailed(errors)) => {
        eprintln!("Validation errors: {:?}", errors);
    },
    Err(e) => eprintln!("Other error: {}", e),
}
```

### Error Conversion

Errors implement standard traits for easy conversion:

```rust
use anyhow::Result;

async fn process_pipeline() -> Result<()> {
    let builder = KnowledgeGraphBuilder::new(config).await?;
    let stats = builder.process_dataset(path, proc_config).await?;
    Ok(())
}
```

## Configuration

### Configuration Loading

```rust
use serde::Deserialize;
use config::{Config, File, Environment};

#[derive(Deserialize)]
struct AppConfig {
    knowledge_graph: KgConfig,
    retriever: RetrieverConfig,
    llm: LLMConfig,
    vector_store: VectorStoreConfig,
}

let config = Config::builder()
    .add_source(File::with_name("config"))
    .add_source(Environment::with_prefix("APP"))
    .build()?
    .try_deserialize::<AppConfig>()?;
```

### Environment Variables

```bash
# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

# LLM API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Logging
export RUST_LOG=info
export RUST_BACKTRACE=1
```

## Complete Example

Here's a complete example demonstrating the API usage:

```rust
use autoschema_kg_rust::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Set up vector store
    let vector_config = EmbeddingConfig::builder()
        .model("sentence-transformers/all-MiniLM-L6-v2")
        .dimension(384)
        .build();
    let mut vector_store = VectorStore::new(vector_config).await?;

    // Set up knowledge graph
    let kg_config = GraphConfig::builder()
        .neo4j_uri("bolt://localhost:7687")
        .username("neo4j")
        .password("password")
        .build();
    let kg_builder = KnowledgeGraphBuilder::new(kg_config).await?;

    // Process dataset
    let processing_config = ProcessingConfig::builder()
        .batch_size(1000)
        .parallel_workers(4)
        .build();

    let stats = kg_builder.process_dataset(
        Path::new("data/entities.csv"),
        processing_config
    ).await?;

    println!("Processed {} entities", stats.entities_processed);

    // Set up retriever
    let retriever_config = RetrieverConfig::builder()
        .max_hops(3)
        .top_k(10)
        .build();

    let retriever = MultiHopRetriever::new(
        retriever_config,
        vector_store,
        kg_builder.graph_client(),
    ).await?;

    // Set up LLM generator
    let llm_config = GenerationConfig::builder()
        .model("gpt-4")
        .temperature(0.1)
        .build();

    let llm = LLMGenerator::new(
        OpenAIProvider::new(&std::env::var("OPENAI_API_KEY")?),
        llm_config,
    );

    // Perform RAG query
    let query = "What are the main applications of machine learning?";
    let context = retriever.retrieve(query, None).await?;
    let response = llm.generate_with_context(query, &context).await?;

    println!("Query: {}", query);
    println!("Response: {}", response.text);
    println!("Context used: {} documents", context.len());

    Ok(())
}
```

This API documentation provides comprehensive coverage of all major components and their usage patterns. For more specific examples and advanced usage, refer to the examples directory in the repository.