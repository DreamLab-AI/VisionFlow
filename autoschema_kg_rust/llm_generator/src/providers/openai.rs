use async_trait::async_trait;
use reqwest::{Client, header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE}};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

use crate::{
    Result, LLMError, LLMGenerator, GenerationConfig, GenerationResponse,
    Message, TokenUsage, UsageTracker, UsageStats, OpenAICostCalculator,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIChoice {
    index: usize,
    message: OpenAIMessage,
    finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIStreamChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIStreamChoice {
    index: usize,
    delta: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIModelsResponse {
    object: String,
    data: Vec<OpenAIModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIModel {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    param: Option<String>,
    code: Option<String>,
}

pub struct OpenAIProvider {
    api_key: String,
    base_url: String,
    client: Client,
    usage_tracker: Arc<UsageTracker>,
    organization: Option<String>,
    timeout: Duration,
}

impl OpenAIProvider {
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(LLMError::InvalidApiKey);
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(LLMError::Http)?;

        let usage_tracker = Arc::new(UsageTracker::new(
            Box::new(OpenAICostCalculator)
        ));

        Ok(Self {
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
            client,
            usage_tracker,
            organization: None,
            timeout: Duration::from_secs(120),
        })
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
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

    fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();

        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| LLMError::InvalidApiKey)?,
        );

        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        if let Some(org) = &self.organization {
            headers.insert(
                "OpenAI-Organization",
                HeaderValue::from_str(org)
                    .map_err(|_| LLMError::Config("Invalid organization header".to_string()))?,
            );
        }

        Ok(headers)
    }

    fn convert_messages(&self, messages: &[Message]) -> Vec<OpenAIMessage> {
        messages
            .iter()
            .map(|msg| OpenAIMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            })
            .collect()
    }

    fn build_request(&self, messages: &[Message], config: &GenerationConfig) -> OpenAIRequest {
        OpenAIRequest {
            model: config.model.clone(),
            messages: self.convert_messages(messages),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            stop: config.stop_sequences.clone(),
            stream: Some(config.stream),
        }
    }

    async fn make_request(&self, request: &OpenAIRequest) -> Result<OpenAIResponse> {
        let headers = self.build_headers()?;
        let url = format!("{}/chat/completions", self.base_url);

        let start_time = Instant::now();

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(request)
            .send()
            .await
            .map_err(LLMError::Http)?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();

            // Try to parse as OpenAI error
            if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&error_text) {
                let error_msg = format!("{}: {}", error_response.error.error_type, error_response.error.message);

                return Err(match status.as_u16() {
                    401 => LLMError::InvalidApiKey,
                    429 => LLMError::RateLimit,
                    _ => LLMError::Provider(error_msg),
                });
            }

            return Err(LLMError::Provider(format!("HTTP {}: {}", status, error_text)));
        }

        let response_text = response.text().await.map_err(LLMError::Http)?;
        let openai_response: OpenAIResponse = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::InvalidResponse(format!("JSON parse error: {}", e)))?;

        // Record usage
        let response_time = start_time.elapsed();
        let token_usage = TokenUsage::new(
            openai_response.usage.prompt_tokens,
            openai_response.usage.completion_tokens,
        );

        self.usage_tracker
            .record_request(token_usage, response_time, true, &request.model)
            .await;

        Ok(openai_response)
    }

    fn convert_response(&self, openai_response: OpenAIResponse, response_time: Duration) -> Result<GenerationResponse> {
        let choice = openai_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LLMError::InvalidResponse("No choices in response".to_string()))?;

        Ok(GenerationResponse {
            text: choice.message.content,
            model: openai_response.model,
            usage: TokenUsage::new(
                openai_response.usage.prompt_tokens,
                openai_response.usage.completion_tokens,
            ),
            finish_reason: choice.finish_reason,
            response_time,
            metadata: std::collections::HashMap::from([
                ("id".to_string(), serde_json::Value::String(openai_response.id)),
                ("object".to_string(), serde_json::Value::String(openai_response.object)),
                ("created".to_string(), serde_json::Value::Number(serde_json::Number::from(openai_response.created))),
            ]),
        })
    }
}

#[async_trait]
impl LLMGenerator for OpenAIProvider {
    async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<GenerationResponse> {
        let messages = vec![Message::user(prompt)];
        self.generate_chat(&messages, config).await
    }

    async fn generate_chat(&self, messages: &[Message], config: &GenerationConfig) -> Result<GenerationResponse> {
        self.validate_config(config)?;

        let request = self.build_request(messages, config);
        let start_time = Instant::now();

        let openai_response = self.make_request(&request).await?;
        let response_time = start_time.elapsed();

        self.convert_response(openai_response, response_time)
    }

    async fn generate_stream(&self, prompt: &str, config: &GenerationConfig, callback: Box<dyn Fn(String) -> Result<()> + Send + Sync>) -> Result<GenerationResponse> {
        let messages = vec![Message::user(prompt)];
        let mut stream_config = config.clone();
        stream_config.stream = true;

        let request = self.build_request(&messages, &stream_config);
        let headers = self.build_headers()?;
        let url = format!("{}/chat/completions", self.base_url);

        let start_time = Instant::now();
        let mut accumulated_text = String::new();
        let mut finish_reason = "stop".to_string();

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(LLMError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::Provider(format!(
                "HTTP {}: {}",
                status,
                error_text
            )));
        }

        let mut stream = response.bytes_stream();
        use futures::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(LLMError::Http)?;
            let chunk_str = String::from_utf8_lossy(&chunk);

            for line in chunk_str.lines() {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    if data == "[DONE]" {
                        break;
                    }

                    if let Ok(stream_chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                        if let Some(choice) = stream_chunk.choices.first() {
                            if let Some(content) = choice.delta.content.as_ref() {
                                if !content.is_empty() {
                                    accumulated_text.push_str(content);
                                    callback(content.clone())?;
                                }
                            }

                            if let Some(reason) = &choice.finish_reason {
                                finish_reason = reason.clone();
                            }
                        }
                    }
                }
            }
        }

        let response_time = start_time.elapsed();

        // Estimate token usage for streaming
        let estimated_usage = TokenUsage::new(
            accumulated_text.len() / 4, // Rough estimate: 4 chars per token
            accumulated_text.len() / 4,
        );

        self.usage_tracker
            .record_request(estimated_usage.clone(), response_time, true, &config.model)
            .await;

        Ok(GenerationResponse {
            text: accumulated_text,
            model: config.model.clone(),
            usage: estimated_usage,
            finish_reason,
            response_time,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let headers = self.build_headers()?;
        let url = format!("{}/models", self.base_url);

        let response = self
            .client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(LLMError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LLMError::Provider(format!(
                "HTTP {}: {}",
                status,
                error_text
            )));
        }

        let models_response: OpenAIModelsResponse = response.json().await.map_err(LLMError::Http)?;

        Ok(models_response.data.into_iter().map(|model| model.id).collect())
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

        if let Some(max_tokens) = config.max_tokens {
            if max_tokens == 0 {
                return Err(LLMError::Config("max_tokens must be greater than 0".to_string()));
            }
        }

        Ok(())
    }

    fn provider_name(&self) -> &str {
        "openai"
    }

    fn supports_streaming(&self) -> bool {
        true
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