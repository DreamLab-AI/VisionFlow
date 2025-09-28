use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

use crate::TokenUsage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub total_requests: usize,
    pub total_tokens: TokenUsage,
    pub total_cost: f64,
    pub average_response_time: std::time::Duration,
    pub error_rate: f64,
    pub last_request: Option<DateTime<Utc>>,
    pub requests_per_minute: f64,
    pub peak_tokens_per_minute: usize,
}

impl Default for UsageStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_tokens: TokenUsage::empty(),
            total_cost: 0.0,
            average_response_time: std::time::Duration::from_secs(0),
            error_rate: 0.0,
            last_request: None,
            requests_per_minute: 0.0,
            peak_tokens_per_minute: 0,
        }
    }
}

#[derive(Debug)]
struct RequestRecord {
    timestamp: DateTime<Utc>,
    tokens: TokenUsage,
    cost: f64,
    response_time: std::time::Duration,
    success: bool,
}

pub struct UsageTracker {
    stats: Arc<RwLock<UsageStats>>,
    records: Arc<RwLock<Vec<RequestRecord>>>,
    cost_calculator: Box<dyn CostCalculator + Send + Sync>,
}

impl UsageTracker {
    pub fn new(cost_calculator: Box<dyn CostCalculator + Send + Sync>) -> Self {
        Self {
            stats: Arc::new(RwLock::new(UsageStats::default())),
            records: Arc::new(RwLock::new(Vec::new())),
            cost_calculator,
        }
    }

    pub async fn record_request(
        &self,
        tokens: TokenUsage,
        response_time: std::time::Duration,
        success: bool,
        model: &str,
    ) {
        let cost = self.cost_calculator.calculate_cost(&tokens, model);
        let now = Utc::now();

        let record = RequestRecord {
            timestamp: now,
            tokens: tokens.clone(),
            cost,
            response_time,
            success,
        };

        // Add to records
        {
            let mut records = self.records.write().await;
            records.push(record);

            // Keep only last 1000 records to prevent memory bloat
            if records.len() > 1000 {
                records.drain(0..records.len() - 1000);
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
            stats.total_tokens.add(&tokens);
            stats.total_cost += cost;
            stats.last_request = Some(now);

            // Update average response time
            let total_time = stats.average_response_time.as_millis() as f64 * (stats.total_requests - 1) as f64
                + response_time.as_millis() as f64;
            stats.average_response_time = std::time::Duration::from_millis((total_time / stats.total_requests as f64) as u64);

            // Calculate error rate
            let records = self.records.read().await;
            let errors = records.iter().filter(|r| !r.success).count();
            stats.error_rate = errors as f64 / stats.total_requests as f64;

            // Calculate requests per minute (last 60 seconds)
            let one_minute_ago = now - chrono::Duration::seconds(60);
            let recent_requests = records.iter().filter(|r| r.timestamp > one_minute_ago).count();
            stats.requests_per_minute = recent_requests as f64;

            // Calculate peak tokens per minute
            let recent_tokens: usize = records
                .iter()
                .filter(|r| r.timestamp > one_minute_ago)
                .map(|r| r.tokens.total_tokens)
                .sum();
            stats.peak_tokens_per_minute = stats.peak_tokens_per_minute.max(recent_tokens);
        }
    }

    pub async fn get_stats(&self) -> UsageStats {
        self.stats.read().await.clone()
    }

    pub async fn reset(&self) {
        let mut stats = self.stats.write().await;
        *stats = UsageStats::default();

        let mut records = self.records.write().await;
        records.clear();
    }

    pub async fn get_hourly_usage(&self) -> Vec<(DateTime<Utc>, usize, f64)> {
        let records = self.records.read().await;
        let mut hourly_stats = std::collections::HashMap::new();

        for record in records.iter() {
            let hour = record.timestamp.with_minute(0).unwrap().with_second(0).unwrap();
            let entry = hourly_stats.entry(hour).or_insert((0, 0.0));
            entry.0 += record.tokens.total_tokens;
            entry.1 += record.cost;
        }

        let mut result: Vec<_> = hourly_stats
            .into_iter()
            .map(|(hour, (tokens, cost))| (hour, tokens, cost))
            .collect();
        result.sort_by_key(|(hour, _, _)| *hour);
        result
    }
}

pub trait CostCalculator: Send + Sync {
    fn calculate_cost(&self, tokens: &TokenUsage, model: &str) -> f64;
}

pub struct OpenAICostCalculator;

impl CostCalculator for OpenAICostCalculator {
    fn calculate_cost(&self, tokens: &TokenUsage, model: &str) -> f64 {
        let (input_cost, output_cost) = match model {
            "gpt-3.5-turbo" => (0.0015, 0.002), // per 1K tokens
            "gpt-4" => (0.03, 0.06),
            "gpt-4-turbo" => (0.01, 0.03),
            _ => (0.001, 0.002), // Default
        };

        let input_cost_total = (tokens.prompt_tokens as f64 / 1000.0) * input_cost;
        let output_cost_total = (tokens.completion_tokens as f64 / 1000.0) * output_cost;

        input_cost_total + output_cost_total
    }
}

pub struct AnthropicCostCalculator;

impl CostCalculator for AnthropicCostCalculator {
    fn calculate_cost(&self, tokens: &TokenUsage, model: &str) -> f64 {
        let (input_cost, output_cost) = match model {
            "claude-3-haiku-20240307" => (0.00025, 0.00125),
            "claude-3-sonnet-20240229" => (0.003, 0.015),
            "claude-3-opus-20240229" => (0.015, 0.075),
            _ => (0.001, 0.005), // Default
        };

        let input_cost_total = (tokens.prompt_tokens as f64 / 1000.0) * input_cost;
        let output_cost_total = (tokens.completion_tokens as f64 / 1000.0) * output_cost;

        input_cost_total + output_cost_total
    }
}

pub struct LocalCostCalculator;

impl CostCalculator for LocalCostCalculator {
    fn calculate_cost(&self, _tokens: &TokenUsage, _model: &str) -> f64 {
        0.0 // Local models are free
    }
}