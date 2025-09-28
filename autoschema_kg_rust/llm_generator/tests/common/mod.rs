//! Common test utilities for llm_generator

use llm_generator::*;
use mockall::mock;
use serde_json::Value;
use std::time::Duration;

/// Mock implementations for testing
pub mod mocks {
    use super::*;

    mock! {
        pub Provider {}

        #[async_trait::async_trait]
        impl llm_generator::providers::Provider for Provider {
            async fn generate(&self, prompt: &str) -> Result<String>;
            async fn generate_with_config(&self, prompt: &str, config: &GenerationConfig) -> Result<GenerationResponse>;
            fn supports_streaming(&self) -> bool;
        }
    }
}

/// Test fixtures
pub mod fixtures {
    use super::*;

    pub fn sample_openai_response() -> Value {
        serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0613",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19
            }
        })
    }

    pub fn sample_anthropic_response() -> Value {
        serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "The capital of France is Paris."}],
            "model": "claude-3-sonnet-20240229",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 7
            }
        })
    }

    pub fn sample_generation_config() -> GenerationConfig {
        GenerationConfig::new()
            .temperature(0.7)
            .max_tokens(1000)
            .top_p(0.9)
    }
}

/// Test helpers
pub mod helpers {
    use super::*;

    pub fn init_test_logging() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Debug)
            .try_init();
    }

    pub async fn with_timeout<F, T>(future: F, timeout: Duration) -> Result<T, tokio::time::error::Elapsed>
    where
        F: std::future::Future<Output = T>,
    {
        tokio::time::timeout(timeout, future).await
    }
}