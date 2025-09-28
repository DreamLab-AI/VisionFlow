use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};
use futures::future::join_all;
use uuid::Uuid;

use crate::{
    Result, LLMError, LLMGenerator, GenerationConfig, GenerationResponse,
    RateLimiter, TokenCounter, RetryManager, Message
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    pub id: String,
    pub prompt: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub config: GenerationConfig,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl BatchRequest {
    pub fn new_prompt(prompt: impl Into<String>, config: GenerationConfig) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            prompt: Some(prompt.into()),
            messages: None,
            config,
            metadata: HashMap::new(),
        }
    }

    pub fn new_chat(messages: Vec<Message>, config: GenerationConfig) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            prompt: None,
            messages: Some(messages),
            config,
            metadata: HashMap::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchResponse {
    pub id: String,
    pub result: Result<GenerationResponse>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub processing_time: std::time::Duration,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchResult {
    pub successful: Vec<BatchResponse>,
    pub failed: Vec<BatchResponse>,
    pub total_time: std::time::Duration,
    pub throughput: f64, // requests per second
    pub error_rate: f64,
}

impl BatchResult {
    pub fn success_rate(&self) -> f64 {
        let total = self.successful.len() + self.failed.len();
        if total > 0 {
            self.successful.len() as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn average_response_time(&self) -> std::time::Duration {
        let all_responses: Vec<_> = self.successful.iter().chain(self.failed.iter()).collect();
        if all_responses.is_empty() {
            return std::time::Duration::from_secs(0);
        }

        let total_ms: u64 = all_responses
            .iter()
            .map(|r| r.processing_time.as_millis() as u64)
            .sum();

        std::time::Duration::from_millis(total_ms / all_responses.len() as u64)
    }
}

pub struct BatchProcessor {
    generator: Arc<dyn LLMGenerator>,
    rate_limiter: Option<Arc<RateLimiter>>,
    retry_manager: Option<Arc<RetryManager>>,
    token_counter: Arc<TokenCounter>,
    max_concurrent: usize,
}

impl BatchProcessor {
    pub fn new(
        generator: Arc<dyn LLMGenerator>,
        token_counter: Arc<TokenCounter>,
    ) -> Self {
        Self {
            generator,
            rate_limiter: None,
            retry_manager: None,
            token_counter,
            max_concurrent: 10,
        }
    }

    pub fn with_rate_limiter(mut self, rate_limiter: Arc<RateLimiter>) -> Self {
        self.rate_limiter = Some(rate_limiter);
        self
    }

    pub fn with_retry_manager(mut self, retry_manager: Arc<RetryManager>) -> Self {
        self.retry_manager = Some(retry_manager);
        self
    }

    pub fn with_max_concurrent(mut self, max_concurrent: usize) -> Self {
        self.max_concurrent = max_concurrent;
        self
    }

    pub async fn process_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchResult> {
        if requests.is_empty() {
            return Ok(BatchResult {
                successful: Vec::new(),
                failed: Vec::new(),
                total_time: std::time::Duration::from_secs(0),
                throughput: 0.0,
                error_rate: 0.0,
            });
        }

        let start_time = std::time::Instant::now();
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent));

        // Create tasks for all requests
        let tasks: Vec<_> = requests
            .into_iter()
            .map(|request| {
                let generator = self.generator.clone();
                let rate_limiter = self.rate_limiter.clone();
                let retry_manager = self.retry_manager.clone();
                let token_counter = self.token_counter.clone();
                let semaphore = semaphore.clone();

                tokio::spawn(async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    Self::process_single_request(
                        request,
                        generator,
                        rate_limiter,
                        retry_manager,
                        token_counter,
                    ).await
                })
            })
            .collect();

        // Wait for all tasks to complete
        let results = join_all(tasks).await;
        let total_time = start_time.elapsed();

        // Collect results
        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for task_result in results {
            match task_result {
                Ok(response) => {
                    if response.result.is_ok() {
                        successful.push(response);
                    } else {
                        failed.push(response);
                    }
                }
                Err(e) => {
                    // Task panicked or was cancelled
                    failed.push(BatchResponse {
                        id: "unknown".to_string(),
                        result: Err(LLMError::Provider(format!("Task failed: {}", e))),
                        metadata: HashMap::new(),
                        processing_time: std::time::Duration::from_secs(0),
                    });
                }
            }
        }

        let total_requests = successful.len() + failed.len();
        let throughput = if total_time.as_secs_f64() > 0.0 {
            total_requests as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        let error_rate = if total_requests > 0 {
            failed.len() as f64 / total_requests as f64
        } else {
            0.0
        };

        Ok(BatchResult {
            successful,
            failed,
            total_time,
            throughput,
            error_rate,
        })
    }

    pub async fn process_batch_streaming<F>(
        &self,
        requests: Vec<BatchRequest>,
        callback: F,
    ) -> Result<BatchResult>
    where
        F: Fn(BatchResponse) -> Result<()> + Send + Sync + Clone + 'static,
    {
        if requests.is_empty() {
            return Ok(BatchResult {
                successful: Vec::new(),
                failed: Vec::new(),
                total_time: std::time::Duration::from_secs(0),
                throughput: 0.0,
                error_rate: 0.0,
            });
        }

        let start_time = std::time::Instant::now();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent));

        // Spawn tasks for all requests
        let tasks: Vec<_> = requests
            .into_iter()
            .map(|request| {
                let generator = self.generator.clone();
                let rate_limiter = self.rate_limiter.clone();
                let retry_manager = self.retry_manager.clone();
                let token_counter = self.token_counter.clone();
                let semaphore = semaphore.clone();
                let tx = tx.clone();

                tokio::spawn(async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    let response = Self::process_single_request(
                        request,
                        generator,
                        rate_limiter,
                        retry_manager,
                        token_counter,
                    ).await;

                    let _ = tx.send(response);
                })
            })
            .collect();

        // Drop the sender so the receiver knows when all tasks are done
        drop(tx);

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        // Process responses as they come in
        while let Some(response) = rx.recv().await {
            // Call the callback
            if let Err(e) = callback(response.clone()) {
                log::warn!("Callback error for response {}: {:?}", response.id, e);
            }

            // Collect the response
            if response.result.is_ok() {
                successful.push(response);
            } else {
                failed.push(response);
            }
        }

        // Wait for all tasks to complete
        let _ = join_all(tasks).await;

        let total_time = start_time.elapsed();
        let total_requests = successful.len() + failed.len();
        let throughput = if total_time.as_secs_f64() > 0.0 {
            total_requests as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        let error_rate = if total_requests > 0 {
            failed.len() as f64 / total_requests as f64
        } else {
            0.0
        };

        Ok(BatchResult {
            successful,
            failed,
            total_time,
            throughput,
            error_rate,
        })
    }

    async fn process_single_request(
        request: BatchRequest,
        generator: Arc<dyn LLMGenerator>,
        rate_limiter: Option<Arc<RateLimiter>>,
        retry_manager: Option<Arc<RetryManager>>,
        token_counter: Arc<TokenCounter>,
    ) -> BatchResponse {
        let start_time = std::time::Instant::now();

        let result = async {
            // Estimate tokens for rate limiting
            let estimated_tokens = if let Some(prompt) = &request.prompt {
                token_counter.count_tokens(prompt)
            } else if let Some(messages) = &request.messages {
                token_counter.count_message_tokens(messages)
            } else {
                1000 // Default estimate
            };

            // Apply rate limiting if configured
            let _rate_guard = if let Some(limiter) = &rate_limiter {
                Some(limiter.check_and_wait(estimated_tokens as u32).await?)
            } else {
                None
            };

            // Execute the request with optional retry
            let response = if let Some(retry_mgr) = &retry_manager {
                retry_mgr.retry_with_backoff(|| async {
                    Self::execute_request(&request, &generator).await
                }).await?
            } else {
                Self::execute_request(&request, &generator).await?
            };

            Ok(response)
        }.await;

        let processing_time = start_time.elapsed();

        BatchResponse {
            id: request.id.clone(),
            result,
            metadata: request.metadata.clone(),
            processing_time,
        }
    }

    async fn execute_request(
        request: &BatchRequest,
        generator: &Arc<dyn LLMGenerator>,
    ) -> Result<GenerationResponse> {
        if let Some(prompt) = &request.prompt {
            generator.generate(prompt, &request.config).await
        } else if let Some(messages) = &request.messages {
            generator.generate_chat(messages, &request.config).await
        } else {
            Err(LLMError::Config(
                "Request must have either prompt or messages".to_string(),
            ))
        }
    }

    pub async fn process_adaptive_batch(
        &self,
        requests: Vec<BatchRequest>,
        target_throughput: f64,
    ) -> Result<BatchResult> {
        let mut batch_size = self.max_concurrent.min(requests.len());
        let mut processed = 0;
        let mut all_successful = Vec::new();
        let mut all_failed = Vec::new();
        let start_time = std::time::Instant::now();

        while processed < requests.len() {
            let end_idx = (processed + batch_size).min(requests.len());
            let batch = requests[processed..end_idx].to_vec();

            let batch_start = std::time::Instant::now();
            let result = self.process_batch(batch).await?;
            let batch_time = batch_start.elapsed();

            // Extend results
            all_successful.extend(result.successful);
            all_failed.extend(result.failed);

            // Adaptive batch size adjustment
            let current_throughput = batch_size as f64 / batch_time.as_secs_f64();
            if current_throughput < target_throughput * 0.8 {
                batch_size = (batch_size * 2).min(self.max_concurrent);
            } else if current_throughput > target_throughput * 1.2 {
                batch_size = (batch_size / 2).max(1);
            }

            processed = end_idx;

            // Small delay to prevent overwhelming the API
            if processed < requests.len() {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }

        let total_time = start_time.elapsed();
        let total_requests = all_successful.len() + all_failed.len();
        let throughput = if total_time.as_secs_f64() > 0.0 {
            total_requests as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        let error_rate = if total_requests > 0 {
            all_failed.len() as f64 / total_requests as f64
        } else {
            0.0
        };

        Ok(BatchResult {
            successful: all_successful,
            failed: all_failed,
            total_time,
            throughput,
            error_rate,
        })
    }
}