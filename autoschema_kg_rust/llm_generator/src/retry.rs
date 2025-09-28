use backoff::{exponential::ExponentialBackoff, Error as BackoffError, ExponentialBackoffBuilder};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;

use crate::{Result, LLMError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub initial_interval: Duration,
    pub max_interval: Duration,
    pub multiplier: f64,
    pub max_elapsed_time: Duration,
    pub retry_on_rate_limit: bool,
    pub retry_on_timeout: bool,
    pub retry_on_server_error: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_interval: Duration::from_millis(100),
            max_interval: Duration::from_secs(60),
            multiplier: 2.0,
            max_elapsed_time: Duration::from_secs(300),
            retry_on_rate_limit: true,
            retry_on_timeout: true,
            retry_on_server_error: true,
        }
    }
}

impl RetryConfig {
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_interval: Duration::from_millis(50),
            max_interval: Duration::from_secs(30),
            multiplier: 1.5,
            max_elapsed_time: Duration::from_secs(180),
            retry_on_rate_limit: true,
            retry_on_timeout: true,
            retry_on_server_error: true,
        }
    }

    pub fn conservative() -> Self {
        Self {
            max_attempts: 2,
            initial_interval: Duration::from_secs(1),
            max_interval: Duration::from_secs(120),
            multiplier: 3.0,
            max_elapsed_time: Duration::from_secs(600),
            retry_on_rate_limit: false,
            retry_on_timeout: false,
            retry_on_server_error: true,
        }
    }

    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            initial_interval: Duration::from_secs(0),
            max_interval: Duration::from_secs(0),
            multiplier: 1.0,
            max_elapsed_time: Duration::from_secs(0),
            retry_on_rate_limit: false,
            retry_on_timeout: false,
            retry_on_server_error: false,
        }
    }
}

pub enum RetryStrategy {
    ExponentialBackoff,
    FixedInterval,
    LinearBackoff,
    Custom(fn(usize) -> Duration),
}

impl std::fmt::Debug for RetryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExponentialBackoff => write!(f, "ExponentialBackoff"),
            Self::FixedInterval => write!(f, "FixedInterval"),
            Self::LinearBackoff => write!(f, "LinearBackoff"),
            Self::Custom(_) => write!(f, "Custom"),
        }
    }
}

impl Clone for RetryStrategy {
    fn clone(&self) -> Self {
        match self {
            Self::ExponentialBackoff => Self::ExponentialBackoff,
            Self::FixedInterval => Self::FixedInterval,
            Self::LinearBackoff => Self::LinearBackoff,
            Self::Custom(f) => Self::Custom(*f),
        }
    }
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self::ExponentialBackoff
    }
}

pub struct RetryManager {
    config: RetryConfig,
    strategy: RetryStrategy,
}

impl RetryManager {
    pub fn new(config: RetryConfig, strategy: RetryStrategy) -> Self {
        Self { config, strategy }
    }

    pub async fn retry_with_backoff<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempts = 0;
        let start_time = std::time::Instant::now();

        let backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(self.config.initial_interval)
            .with_max_interval(self.config.max_interval)
            .with_multiplier(self.config.multiplier)
            .with_max_elapsed_time(Some(self.config.max_elapsed_time))
            .build();

        backoff::future::retry(backoff, || async {
            attempts += 1;

            if attempts > self.config.max_attempts {
                return Err(BackoffError::permanent(LLMError::RetryExhausted { attempts }));
            }

            if start_time.elapsed() > self.config.max_elapsed_time {
                return Err(BackoffError::permanent(LLMError::Timeout));
            }

            match operation().await {
                Ok(result) => Ok(result),
                Err(error) => {
                    if self.should_retry(&error) {
                        log::warn!("Operation failed (attempt {}), retrying: {:?}", attempts, error);
                        Err(BackoffError::transient(error))
                    } else {
                        log::error!("Operation failed permanently: {:?}", error);
                        Err(BackoffError::permanent(error))
                    }
                }
            }
        })
        .await
    }

    pub async fn retry_with_custom_strategy<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempts = 0;
        let start_time = std::time::Instant::now();

        loop {
            attempts += 1;

            if attempts > self.config.max_attempts {
                return Err(LLMError::RetryExhausted { attempts });
            }

            if start_time.elapsed() > self.config.max_elapsed_time {
                return Err(LLMError::Timeout);
            }

            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !self.should_retry(&error) {
                        return Err(error);
                    }

                    if attempts < self.config.max_attempts {
                        let delay = self.calculate_delay(attempts);
                        log::warn!(
                            "Operation failed (attempt {}), retrying in {:?}: {:?}",
                            attempts,
                            delay,
                            error
                        );
                        sleep(delay).await;
                    }
                }
            }
        }
    }

    fn should_retry(&self, error: &LLMError) -> bool {
        match error {
            LLMError::RateLimit => self.config.retry_on_rate_limit,
            LLMError::Timeout => self.config.retry_on_timeout,
            LLMError::Http(reqwest_error) => {
                if let Some(status) = reqwest_error.status() {
                    match status.as_u16() {
                        429 => self.config.retry_on_rate_limit, // Rate limit
                        500..=599 => self.config.retry_on_server_error, // Server errors
                        _ => false,
                    }
                } else {
                    // Network errors, connection errors, etc.
                    self.config.retry_on_timeout
                }
            }
            LLMError::Provider(_) => false, // Don't retry provider-specific errors
            LLMError::InvalidApiKey => false, // Don't retry auth errors
            LLMError::InvalidResponse(_) => false, // Don't retry parsing errors
            LLMError::TokenLimit { .. } => false, // Don't retry token limit errors
            LLMError::RetryExhausted { .. } => false, // Already exhausted
            LLMError::Config(_) => false, // Don't retry config errors
            LLMError::Validation(_) => false, // Don't retry validation errors
            LLMError::Io(_) => true, // Retry IO errors
            LLMError::UrlParse(_) => false, // Don't retry URL parsing errors
            LLMError::Json(_) => false, // Don't retry JSON errors
        }
    }

    fn calculate_delay(&self, attempt: usize) -> Duration {
        match &self.strategy {
            RetryStrategy::ExponentialBackoff => {
                let delay = self.config.initial_interval.as_millis() as f64
                    * self.config.multiplier.powi((attempt - 1) as i32);
                Duration::from_millis(delay as u64).min(self.config.max_interval)
            }
            RetryStrategy::FixedInterval => self.config.initial_interval,
            RetryStrategy::LinearBackoff => {
                let delay = self.config.initial_interval.as_millis() as u64 * attempt as u64;
                Duration::from_millis(delay).min(self.config.max_interval)
            }
            RetryStrategy::Custom(calc) => calc(attempt),
        }
    }

    pub fn get_config(&self) -> &RetryConfig {
        &self.config
    }

    pub fn update_config(&mut self, config: RetryConfig) {
        self.config = config;
    }
}

// Jitter utilities for more sophisticated retry strategies
pub fn add_jitter(duration: Duration, max_jitter_percent: f64) -> Duration {
    use rand::Rng;
    let jitter_factor = rand::thread_rng().gen_range(0.0..=max_jitter_percent);
    let jitter_ms = (duration.as_millis() as f64 * jitter_factor / 100.0) as u64;
    duration + Duration::from_millis(jitter_ms)
}

pub fn full_jitter(duration: Duration) -> Duration {
    use rand::Rng;
    let jitter_ms = rand::thread_rng().gen_range(0..=duration.as_millis() as u64);
    Duration::from_millis(jitter_ms)
}