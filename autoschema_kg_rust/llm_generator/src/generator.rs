use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, LLMError, TokenUsage, UsageStats};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub model: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: bool,
    pub timeout: Option<std::time::Duration>,
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            stream: false,
            timeout: Some(std::time::Duration::from_secs(30)),
            custom_params: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub text: String,
    pub model: String,
    pub usage: TokenUsage,
    pub finish_reason: String,
    pub response_time: std::time::Duration,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            metadata: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            metadata: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            metadata: None,
        }
    }
}

#[async_trait]
pub trait LLMGenerator: Send + Sync {
    /// Generate text from a prompt
    async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<GenerationResponse>;

    /// Generate text from a conversation
    async fn generate_chat(&self, messages: &[Message], config: &GenerationConfig) -> Result<GenerationResponse>;

    /// Stream generation with callback (note: callback is boxed to make trait object-safe)
    async fn generate_stream(&self, prompt: &str, config: &GenerationConfig, callback: Box<dyn Fn(String) -> Result<()> + Send + Sync>) -> Result<GenerationResponse>;

    /// Get available models
    async fn list_models(&self) -> Result<Vec<String>>;

    /// Validate configuration
    fn validate_config(&self, config: &GenerationConfig) -> Result<()>;

    /// Get provider name
    fn provider_name(&self) -> &str;

    /// Check if provider supports streaming
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if provider supports chat
    fn supports_chat(&self) -> bool {
        true
    }

    /// Get usage statistics
    async fn get_usage_stats(&self) -> Result<UsageStats>;

    /// Reset usage statistics
    async fn reset_usage_stats(&self) -> Result<()>;
}