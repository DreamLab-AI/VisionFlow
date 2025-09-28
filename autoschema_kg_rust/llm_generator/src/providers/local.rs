use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    Result, LLMError, LLMGenerator, GenerationConfig, GenerationResponse,
    Message, TokenUsage, UsageTracker, UsageStats, LocalCostCalculator,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalRequest {
    prompt: Option<String>,
    messages: Option<Vec<LocalMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalResponse {
    text: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    usage: Option<LocalUsage>,
    #[serde(default)]
    finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaRequest {
    model: String,
    prompt: Option<String>,
    messages: Option<Vec<LocalMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaResponse {
    model: String,
    #[serde(default)]
    response: String,
    #[serde(default)]
    message: Option<LocalMessage>,
    done: bool,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    load_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum LocalProviderType {
    Generic,
    Ollama,
    LlamaCpp,
    Oobabooga,
}

pub struct LocalProvider {
    base_url: String,
    client: Client,
    usage_tracker: Arc<UsageTracker>,
    provider_type: LocalProviderType,
    model: String,
    timeout: Duration,
}

impl LocalProvider {
    pub fn new(base_url: impl Into<String>, provider_type: LocalProviderType) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // Longer timeout for local models
            .build()
            .map_err(LLMError::Http)?;

        let usage_tracker = Arc::new(UsageTracker::new(
            Box::new(LocalCostCalculator)
        ));

        Ok(Self {
            base_url: base_url.into(),
            client,
            usage_tracker,
            provider_type,
            model: "default".to_string(),
            timeout: Duration::from_secs(300),
        })
    }

    pub fn ollama(base_url: impl Into<String>) -> Result<Self> {
        Self::new(base_url, LocalProviderType::Ollama)
    }

    pub fn llama_cpp(base_url: impl Into<String>) -> Result<Self> {
        Self::new(base_url, LocalProviderType::LlamaCpp)
    }

    pub fn oobabooga(base_url: impl Into<String>) -> Result<Self> {
        Self::new(base_url, LocalProviderType::Oobabooga)
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap();
        self
    }

    fn convert_messages(&self, messages: &[Message]) -> Vec<LocalMessage> {
        messages
            .iter()
            .map(|msg| LocalMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            })
            .collect()
    }

    async fn make_ollama_request(&self, messages: &[Message], config: &GenerationConfig) -> Result<GenerationResponse> {
        let url = format!("{}/api/chat", self.base_url);
        let start_time = Instant::now();

        let options = OllamaOptions {
            temperature: config.temperature,
            top_p: config.top_p,
            num_predict: config.max_tokens,
            stop: config.stop_sequences.clone(),
        };

        let request = OllamaRequest {
            model: config.model.clone(),
            prompt: None,
            messages: Some(self.convert_messages(messages)),
            options: Some(options),
            stream: false,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(LLMError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::Provider(format!(
                "Ollama error {}: {}",
                status,
                error_text
            )));
        }

        let ollama_response: OllamaResponse = response.json().await.map_err(LLMError::Http)?;
        let response_time = start_time.elapsed();

        let text = if let Some(message) = ollama_response.message {
            message.content
        } else {
            ollama_response.response
        };

        let usage = TokenUsage::new(
            ollama_response.prompt_eval_count.unwrap_or(text.len() / 4),
            ollama_response.eval_count.unwrap_or(text.len() / 4),
        );

        self.usage_tracker
            .record_request(usage.clone(), response_time, true, &config.model)
            .await;

        Ok(GenerationResponse {
            text,
            model: ollama_response.model,
            usage,
            finish_reason: "stop".to_string(),
            response_time,
            metadata: std::collections::HashMap::from([
                ("total_duration".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(ollama_response.total_duration.unwrap_or(0))
                )),
                ("load_duration".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(ollama_response.load_duration.unwrap_or(0))
                )),
            ]),
        })
    }

    async fn make_generic_request(&self, messages: &[Message], config: &GenerationConfig) -> Result<GenerationResponse> {
        let url = match self.provider_type {
            LocalProviderType::LlamaCpp => format!("{}/v1/chat/completions", self.base_url),
            LocalProviderType::Oobabooga => format!("{}/v1/chat/completions", self.base_url),
            _ => format!("{}/generate", self.base_url),
        };

        let start_time = Instant::now();

        let request = if messages.len() == 1 && messages[0].role == "user" {
            // Simple prompt-based request
            LocalRequest {
                prompt: Some(messages[0].content.clone()),
                messages: None,
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                top_p: config.top_p,
                stop: config.stop_sequences.clone(),
                stream: Some(false),
            }
        } else {
            // Chat-based request
            LocalRequest {
                prompt: None,
                messages: Some(self.convert_messages(messages)),
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                top_p: config.top_p,
                stop: config.stop_sequences.clone(),
                stream: Some(false),
            }
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(LLMError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::Provider(format!(
                "Local provider error {}: {}",
                status,
                error_text
            )));
        }

        let local_response: LocalResponse = response.json().await.map_err(LLMError::Http)?;
        let response_time = start_time.elapsed();

        let usage = if let Some(usage) = local_response.usage {
            TokenUsage::new(usage.prompt_tokens, usage.completion_tokens)
        } else {
            // Estimate tokens if not provided
            TokenUsage::new(
                local_response.text.len() / 4,
                local_response.text.len() / 4,
            )
        };

        self.usage_tracker
            .record_request(usage.clone(), response_time, true, &config.model)
            .await;

        Ok(GenerationResponse {
            text: local_response.text,
            model: if local_response.model.is_empty() {
                config.model.clone()
            } else {
                local_response.model
            },
            usage,
            finish_reason: if local_response.finish_reason.is_empty() {
                "stop".to_string()
            } else {
                local_response.finish_reason
            },
            response_time,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn stream_ollama_request(
        &self,
        messages: &[Message],
        config: &GenerationConfig,
        callback: Box<dyn Fn(String) -> Result<()> + Send + Sync>,
    ) -> Result<GenerationResponse> {
        let url = format!("{}/api/chat", self.base_url);
        let start_time = Instant::now();

        let options = OllamaOptions {
            temperature: config.temperature,
            top_p: config.top_p,
            num_predict: config.max_tokens,
            stop: config.stop_sequences.clone(),
        };

        let request = OllamaRequest {
            model: config.model.clone(),
            prompt: None,
            messages: Some(self.convert_messages(messages)),
            options: Some(options),
            stream: true,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(LLMError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::Provider(format!(
                "Ollama error {}: {}",
                status,
                error_text
            )));
        }

        let mut stream = response.bytes_stream();
        use futures::StreamExt;

        let mut accumulated_text = String::new();
        let mut model_name = config.model.clone();
        let mut usage = TokenUsage::empty();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(LLMError::Http)?;
            let chunk_str = String::from_utf8_lossy(&chunk);

            for line in chunk_str.lines() {
                if let Ok(ollama_response) = serde_json::from_str::<OllamaResponse>(line) {
                    if let Some(message) = ollama_response.message {
                        if !message.content.is_empty() {
                            accumulated_text.push_str(&message.content);
                            callback(message.content)?;
                        }
                    } else if !ollama_response.response.is_empty() {
                        accumulated_text.push_str(&ollama_response.response);
                        callback(ollama_response.response)?;
                    }

                    model_name = ollama_response.model;

                    if ollama_response.done {
                        usage = TokenUsage::new(
                            ollama_response.prompt_eval_count.unwrap_or(accumulated_text.len() / 4),
                            ollama_response.eval_count.unwrap_or(accumulated_text.len() / 4),
                        );
                        break;
                    }
                }
            }
        }

        let response_time = start_time.elapsed();

        self.usage_tracker
            .record_request(usage.clone(), response_time, true, &config.model)
            .await;

        Ok(GenerationResponse {
            text: accumulated_text,
            model: model_name,
            usage,
            finish_reason: "stop".to_string(),
            response_time,
            metadata: std::collections::HashMap::new(),
        })
    }
}

#[async_trait]
impl LLMGenerator for LocalProvider {
    async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<GenerationResponse> {
        let messages = vec![Message::user(prompt)];
        self.generate_chat(&messages, config).await
    }

    async fn generate_chat(&self, messages: &[Message], config: &GenerationConfig) -> Result<GenerationResponse> {
        self.validate_config(config)?;

        match self.provider_type {
            LocalProviderType::Ollama => self.make_ollama_request(messages, config).await,
            _ => self.make_generic_request(messages, config).await,
        }
    }

    async fn generate_stream(&self, prompt: &str, config: &GenerationConfig, callback: Box<dyn Fn(String) -> Result<()> + Send + Sync>) -> Result<GenerationResponse> {
        let messages = vec![Message::user(prompt)];

        match self.provider_type {
            LocalProviderType::Ollama => self.stream_ollama_request(&messages, config, callback).await,
            _ => {
                // For non-Ollama providers, fall back to non-streaming
                log::warn!("Streaming not supported for this local provider, using non-streaming");
                let response = self.generate_chat(&messages, config).await?;
                callback(response.text.clone())?;
                Ok(response)
            }
        }
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        match self.provider_type {
            LocalProviderType::Ollama => {
                let url = format!("{}/api/tags", self.base_url);

                #[derive(Deserialize)]
                struct OllamaModelsResponse {
                    models: Vec<OllamaModelInfo>,
                }

                #[derive(Deserialize)]
                struct OllamaModelInfo {
                    name: String,
                }

                let response = self
                    .client
                    .get(&url)
                    .send()
                    .await
                    .map_err(LLMError::Http)?;

                if response.status().is_success() {
                    let models_response: OllamaModelsResponse = response.json().await.map_err(LLMError::Http)?;
                    Ok(models_response.models.into_iter().map(|m| m.name).collect())
                } else {
                    Ok(vec![self.model.clone()])
                }
            }
            _ => {
                // For other local providers, return the configured model
                Ok(vec![self.model.clone()])
            }
        }
    }

    fn validate_config(&self, config: &GenerationConfig) -> Result<()> {
        if config.model.is_empty() {
            return Err(LLMError::Config("Model cannot be empty".to_string()));
        }

        if let Some(temp) = config.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(LLMError::Config("Temperature must be between 0.0 and 2.0".to_string()));
            }
        }

        if let Some(top_p) = config.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(LLMError::Config("top_p must be between 0.0 and 1.0".to_string()));
            }
        }

        Ok(())
    }

    fn provider_name(&self) -> &str {
        match self.provider_type {
            LocalProviderType::Ollama => "ollama",
            LocalProviderType::LlamaCpp => "llama_cpp",
            LocalProviderType::Oobabooga => "oobabooga",
            LocalProviderType::Generic => "local",
        }
    }

    fn supports_streaming(&self) -> bool {
        matches!(self.provider_type, LocalProviderType::Ollama)
    }

    fn supports_chat(&self) -> bool {
        true
    }

    async fn get_usage_stats(&self) -> Result<UsageStats> {
        Ok(self.usage_tracker.get_stats().await)
    }

    async fn reset_usage_stats(&self) -> Result<()> {
        self.usage_tracker.reset().await;
        Ok(())
    }
}