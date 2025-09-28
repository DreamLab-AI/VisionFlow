use serde::{Deserialize, Serialize};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use std::sync::Arc;
use once_cell::sync::Lazy;
use dashmap::DashMap;

use crate::{Result, LLMError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl TokenUsage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    pub fn empty() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }

    pub fn add(&mut self, other: &TokenUsage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
}

static ENCODER_CACHE: Lazy<DashMap<String, Arc<CoreBPE>>> = Lazy::new(|| DashMap::new());

pub struct TokenCounter {
    model: String,
    encoder: Arc<CoreBPE>,
}

impl TokenCounter {
    pub fn new(model: &str) -> Result<Self> {
        let encoder = if let Some(cached) = ENCODER_CACHE.get(model) {
            cached.clone()
        } else {
            let bpe = get_bpe_from_model(model)
                .map_err(|e| LLMError::Config(format!("Failed to get encoder for model {}: {}", model, e)))?;
            let encoder = Arc::new(bpe);
            ENCODER_CACHE.insert(model.to_string(), encoder.clone());
            encoder
        };

        Ok(Self {
            model: model.to_string(),
            encoder,
        })
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.encoder.encode_with_special_tokens(text).len()
    }

    pub fn count_message_tokens(&self, messages: &[crate::Message]) -> usize {
        let mut total = 0;

        for message in messages {
            // Add tokens for message structure (role, content delimiters)
            total += 4; // Rough estimate for message overhead
            total += self.count_tokens(&message.role);
            total += self.count_tokens(&message.content);
        }

        // Add tokens for conversation structure
        total += 2;

        total
    }

    pub fn estimate_completion_tokens(&self, max_tokens: Option<usize>, prompt_tokens: usize) -> usize {
        let model_max = self.get_model_context_limit();
        let available = model_max.saturating_sub(prompt_tokens);

        match max_tokens {
            Some(requested) => requested.min(available),
            None => available.min(1000), // Default reasonable limit
        }
    }

    pub fn validate_token_limit(&self, prompt_tokens: usize, max_tokens: Option<usize>) -> Result<()> {
        let model_max = self.get_model_context_limit();
        let estimated_completion = self.estimate_completion_tokens(max_tokens, prompt_tokens);
        let total_estimate = prompt_tokens + estimated_completion;

        if total_estimate > model_max {
            return Err(LLMError::TokenLimit {
                used: total_estimate,
                limit: model_max,
            });
        }

        Ok(())
    }

    fn get_model_context_limit(&self) -> usize {
        match self.model.as_str() {
            "gpt-3.5-turbo" => 4096,
            "gpt-3.5-turbo-16k" => 16384,
            "gpt-4" => 8192,
            "gpt-4-32k" => 32768,
            "gpt-4-turbo" | "gpt-4-turbo-preview" => 128000,
            "claude-3-haiku-20240307" => 200000,
            "claude-3-sonnet-20240229" => 200000,
            "claude-3-opus-20240229" => 200000,
            _ => 4096, // Conservative default
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}