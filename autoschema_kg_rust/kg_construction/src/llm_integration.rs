//! LLM integration for batched inference

use crate::{
    error::{KgConstructionError, Result},
    types::{LlmResponse, TokenUsage},
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// LLM Generator trait for different LLM backends
pub trait LlmGenerator: Send + Sync {
    async fn generate_response(
        &self,
        inputs: Vec<Vec<HashMap<String, String>>>,
        return_text_only: bool,
        max_workers: Option<usize>,
    ) -> Result<Vec<LlmResponse>>;
}

/// Mock LLM Generator for testing
pub struct MockLlmGenerator {
    delay_ms: u64,
}

impl MockLlmGenerator {
    pub fn new(delay_ms: u64) -> Self {
        Self { delay_ms }
    }
}

impl LlmGenerator for MockLlmGenerator {
    async fn generate_response(
        &self,
        inputs: Vec<Vec<HashMap<String, String>>>,
        return_text_only: bool,
        _max_workers: Option<usize>,
    ) -> Result<Vec<LlmResponse>> {
        tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;

        let mut responses = Vec::new();
        for input in inputs {
            let content = input
                .iter()
                .find(|msg| msg.get("role") == Some(&"user".to_string()))
                .and_then(|msg| msg.get("content"))
                .unwrap_or(&"".to_string())
                .clone();

            // Generate mock concepts based on input content
            let concepts = if content.contains("entity") {
                "person, object, place"
            } else if content.contains("event") {
                "action, occurrence, happening"
            } else if content.contains("relation") {
                "connection, link, association"
            } else {
                "concept, idea, notion"
            };

            let usage = if !return_text_only {
                Some(TokenUsage {
                    prompt_tokens: content.len() / 4, // Rough estimation
                    completion_tokens: concepts.len() / 4,
                    total_tokens: (content.len() + concepts.len()) / 4,
                })
            } else {
                None
            };

            responses.push(LlmResponse {
                text: concepts.to_string(),
                usage,
            });
        }

        Ok(responses)
    }
}

/// HTTP-based LLM Generator (for OpenAI, Anthropic, etc.)
pub struct HttpLlmGenerator {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    max_concurrent_requests: usize,
}

impl HttpLlmGenerator {
    pub fn new(
        endpoint: String,
        api_key: String,
        model: String,
        max_concurrent_requests: usize,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint,
            api_key,
            model,
            max_concurrent_requests,
        }
    }

    async fn generate_single(
        &self,
        messages: Vec<HashMap<String, String>>,
        return_usage: bool,
    ) -> Result<LlmResponse> {
        let request_body = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.1,
        });

        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| KgConstructionError::LlmError(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(KgConstructionError::LlmError(format!(
                "API error {}: {}",
                response.status(),
                error_text
            )));
        }

        let response_json: Value = response
            .json()
            .await
            .map_err(|e| KgConstructionError::LlmError(format!("JSON parsing error: {}", e)))?;

        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| KgConstructionError::LlmError("No content in response".to_string()))?
            .to_string();

        let usage = if return_usage {
            response_json["usage"].as_object().map(|u| TokenUsage {
                prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as usize,
                total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as usize,
            })
        } else {
            None
        };

        Ok(LlmResponse { text: content, usage })
    }
}

impl LlmGenerator for HttpLlmGenerator {
    async fn generate_response(
        &self,
        inputs: Vec<Vec<HashMap<String, String>>>,
        return_text_only: bool,
        _max_workers: Option<usize>,
    ) -> Result<Vec<LlmResponse>> {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_requests));
        let tasks: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                let semaphore = semaphore.clone();
                let return_usage = !return_text_only;
                async move {
                    let _permit = semaphore.acquire().await.map_err(|e| {
                        KgConstructionError::ThreadingError(format!("Semaphore error: {}", e))
                    })?;
                    self.generate_single(input, return_usage).await
                }
            })
            .collect();

        let results = futures::future::try_join_all(tasks).await?;
        Ok(results)
    }
}

/// Batched inference function that processes inputs and returns structured results
pub async fn batched_inference<T: LlmGenerator>(
    model: &T,
    inputs: Vec<Vec<HashMap<String, String>>>,
    record: bool,
    max_workers: Option<usize>,
) -> Result<(Vec<Vec<String>>, Option<Vec<TokenUsage>>)> {
    let responses = model
        .generate_response(inputs, !record, max_workers)
        .await?;

    let mut answers = Vec::new();
    let mut usages = if record { Some(Vec::new()) } else { None };

    for response in responses {
        // Parse the response text and split by commas
        let answer: Vec<String> = response
            .text
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();

        answers.push(answer);

        if let Some(ref mut usage_vec) = usages {
            if let Some(usage) = response.usage {
                usage_vec.push(usage);
            } else {
                // Create a default usage entry if not provided
                usage_vec.push(TokenUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                });
            }
        }
    }

    Ok((answers, usages))
}

/// Rate limiter for API calls
pub struct RateLimiter {
    requests_per_minute: u32,
    last_request_times: std::sync::Mutex<std::collections::VecDeque<std::time::Instant>>,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            last_request_times: std::sync::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    pub async fn wait_if_needed(&self) -> Result<()> {
        let now = std::time::Instant::now();
        let one_minute_ago = now - std::time::Duration::from_secs(60);

        let mut times = self.last_request_times.lock().map_err(|e| {
            KgConstructionError::ThreadingError(format!("Mutex poisoned: {}", e))
        })?;

        // Remove old requests
        while let Some(&front_time) = times.front() {
            if front_time < one_minute_ago {
                times.pop_front();
            } else {
                break;
            }
        }

        // Check if we need to wait
        if times.len() >= self.requests_per_minute as usize {
            if let Some(&oldest) = times.front() {
                let wait_time = oldest + std::time::Duration::from_secs(60) - now;
                if wait_time > std::time::Duration::ZERO {
                    tokio::time::sleep(wait_time).await;
                }
            }
        }

        times.push_back(now);
        Ok(())
    }
}

/// Retry logic for failed requests
pub async fn retry_with_backoff<F, Fut, T>(
    mut operation: F,
    max_retries: usize,
    initial_delay_ms: u64,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut delay = initial_delay_ms;

    for attempt in 0..=max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt == max_retries {
                    return Err(e);
                }

                // Log the retry attempt
                eprintln!("Attempt {} failed: {}, retrying in {}ms", attempt + 1, e, delay);

                tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                delay *= 2; // Exponential backoff
            }
        }
    }

    unreachable!()
}

/// Clean text for LLM processing
pub fn clean_text(text: &str) -> String {
    text.replace('\n', " ")
        .replace('\r', " ")
        .replace('\t', " ")
        .replace('\x0b', " ") // vertical tab
        .replace('\x0c', " ") // form feed
        .replace('\x08', " ") // backspace
        .replace('\x07', " ") // bell
        .replace('\x1b', " ") // escape
        .replace(';', ",")
        .replace('\x00', "") // NUL character
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Remove NUL characters from text
pub fn remove_nul(text: &str) -> String {
    text.replace('\x00', "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_llm_generator() {
        let generator = MockLlmGenerator::new(10);
        let inputs = vec![vec![
            [("role".to_string(), "user".to_string()),
             ("content".to_string(), "Generate concepts for entity test".to_string())]
            .iter().cloned().collect()
        ]];

        let (answers, _) = batched_inference(&generator, inputs, false, None).await.unwrap();
        assert_eq!(answers.len(), 1);
        assert!(!answers[0].is_empty());
    }

    #[test]
    fn test_clean_text() {
        let dirty_text = "Hello\nWorld\t\x00Test;More";
        let clean = clean_text(dirty_text);
        assert_eq!(clean, "Hello World Test,More");
    }

    #[test]
    fn test_remove_nul() {
        let text_with_nul = "Hello\x00World";
        let clean = remove_nul(text_with_nul);
        assert_eq!(clean, "HelloWorld");
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(2);

        // First two requests should be immediate
        limiter.wait_if_needed().await.unwrap();
        limiter.wait_if_needed().await.unwrap();

        // Third request might need to wait (depending on timing)
        limiter.wait_if_needed().await.unwrap();
    }
}