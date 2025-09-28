//! Mock implementations for testing

use mockall::mock;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

// Mock LLM Provider
mock! {
    pub LLMProvider {}

    #[async_trait]
    impl llm_generator::providers::Provider for LLMProvider {
        async fn generate(&self, prompt: &str) -> llm_generator::Result<String>;
        async fn generate_with_config(&self, prompt: &str, config: &llm_generator::GenerationConfig) -> llm_generator::Result<llm_generator::GenerationResponse>;
        fn supports_streaming(&self) -> bool;
        async fn generate_stream(&self, prompt: &str) -> llm_generator::Result<Box<dyn futures::Stream<Item = llm_generator::Result<String>> + Send + Unpin>>;
    }
}

// Mock Vector Store
mock! {
    pub VectorStore {}

    #[async_trait]
    impl VectorStoreInterface for VectorStore {
        async fn store_embedding(&mut self, id: &str, embedding: &[f32]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
        async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error + Send + Sync>>;
        async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>>;
        async fn delete_embedding(&mut self, id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>;
    }
}

#[async_trait]
pub trait VectorStoreInterface {
    async fn store_embedding(&mut self, id: &str, embedding: &[f32]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>>;
    async fn delete_embedding(&mut self, id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>;
}

// Mock Neo4j Client
mock! {
    pub Neo4jClient {}

    #[async_trait]
    impl Neo4jInterface for Neo4jClient {
        async fn create_node(&self, node: &Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
        async fn create_relationship(&self, rel: &Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
        async fn find_nodes(&self, label: &str, properties: &HashMap<String, Value>) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>>;
        async fn run_cypher(&self, query: &str, params: &HashMap<String, Value>) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>>;
    }
}

#[async_trait]
pub trait Neo4jInterface {
    async fn create_node(&self, node: &Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn create_relationship(&self, rel: &Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn find_nodes(&self, label: &str, properties: &HashMap<String, Value>) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>>;
    async fn run_cypher(&self, query: &str, params: &HashMap<String, Value>) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>>;
}

// Mock HTTP Client for testing external API calls
mock! {
    pub HttpClient {}

    #[async_trait]
    impl HttpClientInterface for HttpClient {
        async fn get(&self, url: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
        async fn post(&self, url: &str, body: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
        async fn post_json(&self, url: &str, json: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;
    }
}

#[async_trait]
pub trait HttpClientInterface {
    async fn get(&self, url: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn post(&self, url: &str, body: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    async fn post_json(&self, url: &str, json: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;
}

// Mock File System for testing file operations
mock! {
    pub FileSystem {}

    #[async_trait]
    impl FileSystemInterface for FileSystem {
        async fn read_file(&self, path: &str) -> Result<String, std::io::Error>;
        async fn write_file(&self, path: &str, content: &str) -> Result<(), std::io::Error>;
        async fn file_exists(&self, path: &str) -> bool;
        async fn list_files(&self, dir: &str) -> Result<Vec<String>, std::io::Error>;
    }
}

#[async_trait]
pub trait FileSystemInterface {
    async fn read_file(&self, path: &str) -> Result<String, std::io::Error>;
    async fn write_file(&self, path: &str, content: &str) -> Result<(), std::io::Error>;
    async fn file_exists(&self, path: &str) -> bool;
    async fn list_files(&self, dir: &str) -> Result<Vec<String>, std::io::Error>;
}

/// Helper function to create mock LLM response
pub fn mock_llm_response(content: &str) -> Value {
    serde_json::json!({
        "id": "test_response",
        "object": "text_completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    })
}

/// Helper function to create mock embeddings
pub fn mock_embeddings(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| (0..dimensions).map(|j| (i * j) as f32 * 0.01).collect())
        .collect()
}

/// Setup function for creating all mocks together
pub fn setup_mocks() -> (MockLLMProvider, MockVectorStore, MockNeo4jClient, MockHttpClient, MockFileSystem) {
    (
        MockLLMProvider::new(),
        MockVectorStore::new(),
        MockNeo4jClient::new(),
        MockHttpClient::new(),
        MockFileSystem::new()
    )
}