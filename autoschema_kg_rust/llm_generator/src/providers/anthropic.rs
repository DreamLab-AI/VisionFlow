use async_trait::async_trait;
use reqwest::{Client, header::{HeaderMap, HeaderValue, CONTENT_TYPE}};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    Result, LLMError, LLMGenerator, GenerationConfig, GenerationResponse,
    Message, TokenUsage, UsageTracker, UsageStats, AnthropicCostCalculator,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    message: Option<AnthropicStreamMessage>,
    content_block: Option<AnthropicStreamContentBlock>,
    delta: Option<AnthropicStreamDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamMessage {
    id: String,
    #[serde(rename = "type")]
    message_type: String,
    role: String,
    content: Vec<serde_json::Value>,
    model: String,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamDelta {
    #[serde(rename = "type")]
    delta_type: String,
    text: Option<String>,
    stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    error_type: String,
    error: AnthropicErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    detail_type: String,
    message: String,
}

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: Client,
    usage_tracker: Arc<UsageTracker>,
    anthropic_version: String,
    timeout: Duration,
}

impl AnthropicProvider {
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
            Box::new(AnthropicCostCalculator)
        ));

        Ok(Self {
            api_key,
            base_url: "https://api.anthropic.com".to_string(),
            client,
            usage_tracker,
            anthropic_version: "2023-06-01".to_string(),
            timeout: Duration::from_secs(120),
        })
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
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
            "x-api-key",
            HeaderValue::from_str(&self.api_key)
                .map_err(|_| LLMError::InvalidApiKey)?,
        );

        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(&self.anthropic_version)
                .map_err(|_| LLMError::Config("Invalid version header".to_string()))?,
        );

        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        Ok(headers)
    }

    fn convert_messages(&self, messages: &[Message]) -> (Option<String>, Vec<AnthropicMessage>) {
        let mut system_message = None;
        let mut anthropic_messages = Vec::new();

        for message in messages {
            match message.role.as_str() {
                "system" => {
                    // Anthropic handles system messages separately
                    system_message = Some(message.content.clone());
                }
                "user" | "assistant" => {
                    anthropic_messages.push(AnthropicMessage {
                        role: message.role.clone(),
                        content: message.content.clone(),
                    });
                }
                _ => {
                    // Convert unknown roles to user messages
                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: message.content.clone(),
                    });
                }
            }
        }

        (system_message, anthropic_messages)
    }

    fn build_request(&self, messages: &[Message], config: &GenerationConfig) -> AnthropicRequest {
        let (system, anthropic_messages) = self.convert_messages(messages);

        AnthropicRequest {
            model: config.model.clone(),
            messages: anthropic_messages,
            max_tokens: config.max_tokens.or(Some(1000)), // Anthropic requires max_tokens
            temperature: config.temperature,
            top_p: config.top_p,
            stop_sequences: config.stop_sequences.clone(),
            stream: Some(config.stream),
            system,
        }
    }

    async fn make_request(&self, request: &AnthropicRequest) -> Result<AnthropicResponse> {
        let headers = self.build_headers()?;
        let url = format!("{}/v1/messages", self.base_url);

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

            // Try to parse as Anthropic error
            if let Ok(error_response) = serde_json::from_str::<AnthropicErrorResponse>(&error_text) {
                let error_msg = error_response.error.message;

                return Err(match status.as_u16() {
                    401 => LLMError::InvalidApiKey,
                    429 => LLMError::RateLimit,
                    _ => LLMError::Provider(error_msg),
                });
            }

            return Err(LLMError::Provider(format!("HTTP {}: {}", status, error_text)));
        }

        let response_text = response.text().await.map_err(LLMError::Http)?;
        let anthropic_response: AnthropicResponse = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::InvalidResponse(format!("JSON parse error: {}", e)))?;

        // Record usage
        let response_time = start_time.elapsed();
        let token_usage = TokenUsage::new(
            anthropic_response.usage.input_tokens,
            anthropic_response.usage.output_tokens,
        );

        self.usage_tracker
            .record_request(token_usage, response_time, true, &request.model)
            .await;

        Ok(anthropic_response)
    }

    fn convert_response(&self, anthropic_response: AnthropicResponse, response_time: Duration) -> Result<GenerationResponse> {
        let text = anthropic_response
            .content
            .into_iter()
            .filter(|content| content.content_type == "text")
            .map(|content| content.text)
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            return Err(LLMError::InvalidResponse("No text content in response".to_string()));
        }

        Ok(GenerationResponse {
            text,
            model: anthropic_response.model,
            usage: TokenUsage::new(
                anthropic_response.usage.input_tokens,
                anthropic_response.usage.output_tokens,
            ),
            finish_reason: anthropic_response.stop_reason,
            response_time,
            metadata: std::collections::HashMap::from([
                ("id".to_string(), serde_json::Value::String(anthropic_response.id)),
                ("type".to_string(), serde_json::Value::String(anthropic_response.response_type)),
                ("role".to_string(), serde_json::Value::String(anthropic_response.role)),
            ]),
        })
    }
}

#[async_trait]
impl LLMGenerator for AnthropicProvider {
    async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<GenerationResponse> {
        let messages = vec![Message::user(prompt)];
        self.generate_chat(&messages, config).await
    }

    async fn generate_chat(&self, messages: &[Message], config: &GenerationConfig) -> Result<GenerationResponse> {
        self.validate_config(config)?;

        let request = self.build_request(messages, config);
        let start_time = Instant::now();

        let anthropic_response = self.make_request(&request).await?;
        let response_time = start_time.elapsed();

        self.convert_response(anthropic_response, response_time)
    }

    async fn generate_stream(&self, prompt: &str, config: &GenerationConfig, callback: Box<dyn Fn(String) -> Result<()> + Send + Sync>) -> Result<GenerationResponse> {
        let messages = vec![Message::user(prompt)];
        let mut stream_config = config.clone();
        stream_config.stream = true;

        let request = self.build_request(&messages, &stream_config);
        let headers = self.build_headers()?;
        let url = format!("{}/v1/messages", self.base_url);

        let start_time = Instant::now();
        let mut accumulated_text = String::new();
        let mut finish_reason = "end_turn".to_string();

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

                    if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                        match event.event_type.as_str() {
                            "content_block_delta" => {
                                if let Some(delta) = event.delta {
                                    if let Some(text) = delta.text {
                                        accumulated_text.push_str(&text);
                                        callback(text)?;
                                    }
                                }
                            }
                            "message_stop" => {
                                if let Some(message) = event.message {
                                    if let Some(reason) = message.stop_reason {
                                        finish_reason = reason;
                                    }
                                }
                            }
                            _ => {} // Ignore other event types
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
        // Anthropic doesn't have a public models endpoint, return known models
        Ok(vec![
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
            "claude-2.1".to_string(),
            "claude-2.0".to_string(),
            "claude-instant-1.2".to_string(),
        ])
    }

    fn validate_config(&self, config: &GenerationConfig) -> Result<()> {
        if config.model.is_empty() {
            return Err(LLMError::Config("Model cannot be empty".to_string()));
        }

        if config.max_tokens.is_none() {
            return Err(LLMError::Config("max_tokens is required for Anthropic".to_string()));
        }

        if let Some(max_tokens) = config.max_tokens {
            if max_tokens == 0 || max_tokens > 4096 {
                return Err(LLMError::Config("max_tokens must be between 1 and 4096".to_string()));
            }
        }

        if let Some(temp) = config.temperature {
            if !(0.0..=1.0).contains(&temp) {
                return Err(LLMError::Config("Temperature must be between 0.0 and 1.0".to_string()));
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
        "anthropic"
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