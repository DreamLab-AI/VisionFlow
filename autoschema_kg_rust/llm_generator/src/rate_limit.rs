use governor::{
    clock::{Clock, DefaultClock, QuantaClock},
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter as GovernorRateLimiter,
};
use std::num::NonZeroU32;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;

use crate::{Result, LLMError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateConfig {
    pub requests_per_minute: u32,
    pub tokens_per_minute: u32,
    pub concurrent_requests: u32,
    pub burst_allowance: u32,
}

impl Default for RateConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 60000,
            concurrent_requests: 5,
            burst_allowance: 10,
        }
    }
}

impl RateConfig {
    pub fn openai_free() -> Self {
        Self {
            requests_per_minute: 3,
            tokens_per_minute: 40000,
            concurrent_requests: 1,
            burst_allowance: 5,
        }
    }

    pub fn openai_plus() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 90000,
            concurrent_requests: 5,
            burst_allowance: 20,
        }
    }

    pub fn anthropic_free() -> Self {
        Self {
            requests_per_minute: 5,
            tokens_per_minute: 25000,
            concurrent_requests: 1,
            burst_allowance: 5,
        }
    }

    pub fn anthropic_pro() -> Self {
        Self {
            requests_per_minute: 50,
            tokens_per_minute: 100000,
            concurrent_requests: 5,
            burst_allowance: 15,
        }
    }

    pub fn unlimited() -> Self {
        Self {
            requests_per_minute: u32::MAX,
            tokens_per_minute: u32::MAX,
            concurrent_requests: 100,
            burst_allowance: u32::MAX,
        }
    }
}

pub struct RateLimiter {
    request_limiter: GovernorRateLimiter<NotKeyed, InMemoryState, DefaultClock, governor::middleware::NoOpMiddleware>,
    token_limiter: GovernorRateLimiter<NotKeyed, InMemoryState, DefaultClock, governor::middleware::NoOpMiddleware>,
    concurrent_semaphore: tokio::sync::Semaphore,
    config: RateConfig,
}

impl RateLimiter {
    pub fn new(config: RateConfig) -> Result<Self> {
        let request_quota = Quota::per_minute(
            NonZeroU32::new(config.requests_per_minute.max(1)).unwrap()
        )
        .allow_burst(
            NonZeroU32::new(config.burst_allowance.max(1)).unwrap()
        );

        let token_quota = Quota::per_minute(
            NonZeroU32::new(config.tokens_per_minute.max(1)).unwrap()
        );

        Ok(Self {
            request_limiter: GovernorRateLimiter::direct(request_quota),
            token_limiter: GovernorRateLimiter::direct(token_quota),
            concurrent_semaphore: tokio::sync::Semaphore::new(config.concurrent_requests as usize),
            config,
        })
    }

    pub async fn acquire_request_permit(&self) -> Result<RateLimitGuard<'_>> {
        // Acquire concurrent request permit
        let permit = self
            .concurrent_semaphore
            .acquire()
            .await
            .map_err(|_| LLMError::RateLimit)?;

        // Check request rate limit
        if let Err(negative) = self.request_limiter.check() {
            let earliest = negative.earliest_possible();
            let clock = QuantaClock::default();
            let now = clock.now();
            let wait_time = earliest.duration_since(now).unwrap_or(Duration::from_secs(0));
            if wait_time > Duration::from_secs(60) {
                return Err(LLMError::RateLimit);
            }
            sleep(wait_time).await;

            // Try again after waiting
            self.request_limiter.check().map_err(|_| LLMError::RateLimit)?;
        }

        Ok(RateLimitGuard { _permit: permit })
    }

    pub async fn acquire_token_permit(&self, tokens: u32) -> Result<()> {
        // For token-based limiting, we might need to wait multiple times
        let mut remaining_tokens = tokens;

        while remaining_tokens > 0 {
            let tokens_to_check = remaining_tokens.min(self.config.tokens_per_minute);

            if let Err(negative) = self.token_limiter.check_n(
                NonZeroU32::new(tokens_to_check.max(1)).unwrap()
            ) {
                let earliest = negative.earliest_possible();
                let clock = QuantaClock::default();
                let now = clock.now();
                let wait_time = earliest.duration_since(now).unwrap_or(Duration::from_secs(0));
                if wait_time > Duration::from_secs(300) {
                    // Don't wait more than 5 minutes
                    return Err(LLMError::RateLimit);
                }
                sleep(wait_time).await;
                continue;
            }

            remaining_tokens = remaining_tokens.saturating_sub(tokens_to_check);
        }

        Ok(())
    }

    pub async fn check_and_wait(&self, estimated_tokens: u32) -> Result<RateLimitGuard<'_>> {
        // First acquire token permits
        self.acquire_token_permit(estimated_tokens).await?;

        // Then acquire request permit
        self.acquire_request_permit().await
    }

    pub fn get_config(&self) -> &RateConfig {
        &self.config
    }

    pub fn update_config(&mut self, config: RateConfig) -> Result<()> {
        *self = Self::new(config)?;
        Ok(())
    }

    pub async fn get_current_usage(&self) -> RateUsage {
        // This would require additional tracking in a real implementation
        RateUsage {
            requests_used: 0,
            tokens_used: 0,
            concurrent_requests: self.concurrent_semaphore.available_permits(),
        }
    }
}

pub struct RateLimitGuard<'a> {
    _permit: tokio::sync::SemaphorePermit<'a>,
}

#[derive(Debug, Clone)]
pub struct RateUsage {
    pub requests_used: u32,
    pub tokens_used: u32,
    pub concurrent_requests: usize,
}

// Adaptive rate limiting that adjusts based on response patterns
pub struct AdaptiveRateLimiter {
    base_limiter: RateLimiter,
    success_rate: f64,
    adjustment_factor: f64,
    last_adjustment: std::time::Instant,
}

impl AdaptiveRateLimiter {
    pub fn new(base_config: RateConfig) -> Result<Self> {
        Ok(Self {
            base_limiter: RateLimiter::new(base_config)?,
            success_rate: 1.0,
            adjustment_factor: 1.0,
            last_adjustment: std::time::Instant::now(),
        })
    }

    pub async fn acquire_permit(&self, estimated_tokens: u32) -> Result<RateLimitGuard<'_>> {
        self.base_limiter.check_and_wait(estimated_tokens).await
    }

    pub fn record_result(&mut self, success: bool) {
        // Update success rate with exponential moving average
        let alpha = 0.1;
        let new_success = if success { 1.0 } else { 0.0 };
        self.success_rate = alpha * new_success + (1.0 - alpha) * self.success_rate;

        // Adjust rate limits if success rate drops
        if self.success_rate < 0.8 && self.last_adjustment.elapsed() > Duration::from_secs(60) {
            self.adjustment_factor *= 0.8; // Reduce rate by 20%
            self.last_adjustment = std::time::Instant::now();
        } else if self.success_rate > 0.95 && self.adjustment_factor < 1.0 {
            self.adjustment_factor = (self.adjustment_factor * 1.1).min(1.0); // Increase rate by 10%
            self.last_adjustment = std::time::Instant::now();
        }
    }

    pub fn get_adjusted_config(&self) -> RateConfig {
        let base = self.base_limiter.get_config();
        RateConfig {
            requests_per_minute: ((base.requests_per_minute as f64) * self.adjustment_factor) as u32,
            tokens_per_minute: ((base.tokens_per_minute as f64) * self.adjustment_factor) as u32,
            concurrent_requests: base.concurrent_requests,
            burst_allowance: base.burst_allowance,
        }
    }
}