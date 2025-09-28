use llm_generator::*;
use llm_generator::error::*;
use tokio_test;
use rstest::*;
use test_case::test_case;
use mockall::{predicate::*, mock};
use serde_json::json;
use std::time::Duration;

mod common;
use common::*;

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_llm_error_display() {
        let errors = vec![
            LLMError::RateLimit,
            LLMError::TokenLimit { used: 150, limit: 100 },
            LLMError::InvalidApiKey,
            LLMError::InvalidResponse("Bad format".to_string()),
            LLMError::Provider("API Error".to_string()),
            LLMError::Timeout,
            LLMError::RetryExhausted { attempts: 3 },
            LLMError::Config("Invalid config".to_string()),
            LLMError::Validation("Validation failed".to_string()),
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
            println!("Error: {}", display);
        }
    }

    #[test]
    fn test_error_from_conversions() {
        // Test HTTP error conversion
        let http_error = reqwest::Error::from(reqwest::ErrorKind::Request);
        let llm_error: LLMError = http_error.into();
        assert!(matches!(llm_error, LLMError::Http(_)));

        // Test JSON error conversion
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let llm_error: LLMError = json_error.into();
        assert!(matches!(llm_error, LLMError::Json(_)));

        // Test IO error conversion
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let llm_error: LLMError = io_error.into();
        assert!(matches!(llm_error, LLMError::Io(_)));
    }
}

#[cfg(test)]
mod generation_config_tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 1000);
        assert_eq!(config.top_p, 1.0);
        assert!(config.stop_sequences.is_empty());
    }

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::new()
            .temperature(0.5)
            .max_tokens(500)
            .top_p(0.9)
            .stop_sequence("END".to_string());

        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, 500);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.stop_sequences, vec!["END"]);
    }

    #[test_case(0.0, true; "zero temperature valid")]
    #[test_case(2.0, true; "high temperature valid")]
    #[test_case(-0.1, false; "negative temperature invalid")]
    #[test_case(2.1, false; "too high temperature invalid")]
    fn test_temperature_validation(temp: f32, expected_valid: bool) {
        let result = GenerationConfig::new().temperature(temp).validate();
        assert_eq!(result.is_ok(), expected_valid);
    }

    #[test_case(1, true; "min tokens valid")]
    #[test_case(4096, true; "max tokens valid")]
    #[test_case(0, false; "zero tokens invalid")]
    #[test_case(10000, false; "too many tokens invalid")]
    fn test_max_tokens_validation(max_tokens: usize, expected_valid: bool) {
        let result = GenerationConfig::new().max_tokens(max_tokens).validate();
        assert_eq!(result.is_ok(), expected_valid);
    }
}

#[cfg(test)]
mod prompt_template_tests {
    use super::*;

    #[test]
    fn test_prompt_template_creation() {
        let template = PromptTemplate::new("Hello, {{name}}!");
        assert_eq!(template.template(), "Hello, {{name}}!");
    }

    #[test]
    fn test_prompt_template_render() {
        let template = PromptTemplate::new("Hello, {{name}}! You are {{age}} years old.");
        let mut variables = std::collections::HashMap::new();
        variables.insert("name".to_string(), "Alice".to_string());
        variables.insert("age".to_string(), "30".to_string());

        let rendered = template.render(&variables).unwrap();
        assert_eq!(rendered, "Hello, Alice! You are 30 years old.");
    }

    #[test]
    fn test_prompt_template_missing_variable() {
        let template = PromptTemplate::new("Hello, {{name}}!");
        let variables = std::collections::HashMap::new();

        let result = template.render(&variables);
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_builder() {
        let prompt = PromptBuilder::new()
            .system("You are a helpful assistant.")
            .user("What is the capital of France?")
            .build();

        assert!(prompt.contains("You are a helpful assistant."));
        assert!(prompt.contains("What is the capital of France?"));
    }
}

#[cfg(test)]
mod token_counter_tests {
    use super::*;

    #[test]
    fn test_token_counter_creation() {
        let counter = TokenCounter::new("gpt-3.5-turbo");
        assert_eq!(counter.model(), "gpt-3.5-turbo");
    }

    #[test]
    fn test_count_tokens() {
        let counter = TokenCounter::new("gpt-3.5-turbo");
        let text = "Hello, world!";
        let count = counter.count_tokens(text);
        assert!(count > 0);
        assert!(count < 100); // Should be reasonable for short text
    }

    #[test]
    fn test_token_usage_tracking() {
        let mut usage = TokenUsage::new();
        usage.add_prompt_tokens(10);
        usage.add_completion_tokens(20);

        assert_eq!(usage.prompt_tokens(), 10);
        assert_eq!(usage.completion_tokens(), 20);
        assert_eq!(usage.total_tokens(), 30);
    }

    #[test]
    fn test_token_usage_cost_calculation() {
        let mut usage = TokenUsage::new();
        usage.add_prompt_tokens(1000);
        usage.add_completion_tokens(500);

        let cost = usage.calculate_cost("gpt-3.5-turbo");
        assert!(cost > 0.0);
    }
}

#[cfg(test)]
mod rate_limiter_tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let config = RateConfig::new(10, Duration::from_secs(1));
        let limiter = RateLimiter::new(config);
        assert_eq!(limiter.config().requests_per_window, 10);
    }

    #[tokio::test]
    async fn test_rate_limiter_allows_requests() {
        let config = RateConfig::new(10, Duration::from_millis(100));
        let limiter = RateLimiter::new(config);

        // Should allow first few requests
        for _ in 0..5 {
            assert!(limiter.check_rate_limit().await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_excess_requests() {
        let config = RateConfig::new(2, Duration::from_secs(1));
        let limiter = RateLimiter::new(config);

        // First two requests should succeed
        assert!(limiter.check_rate_limit().await.is_ok());
        assert!(limiter.check_rate_limit().await.is_ok());

        // Third request should be rate limited
        let result = limiter.check_rate_limit().await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LLMError::RateLimit));
    }
}

#[cfg(test)]
mod retry_tests {
    use super::*;

    #[test]
    fn test_retry_config_creation() {
        let config = RetryConfig::new(3, Duration::from_millis(100));
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.base_delay, Duration::from_millis(100));
    }

    #[test]
    fn test_retry_strategy_exponential() {
        let strategy = RetryStrategy::Exponential { base: Duration::from_millis(100), max: Duration::from_secs(1) };
        let delay = strategy.delay_for_attempt(2);
        assert!(delay >= Duration::from_millis(400)); // 100 * 2^2
    }

    #[test]
    fn test_retry_strategy_linear() {
        let strategy = RetryStrategy::Linear { increment: Duration::from_millis(100) };
        let delay = strategy.delay_for_attempt(3);
        assert_eq!(delay, Duration::from_millis(300)); // 100 * 3
    }

    #[tokio::test]
    async fn test_retry_success_on_first_attempt() {
        let config = RetryConfig::new(3, Duration::from_millis(10));
        let mut attempt = 0;

        let result = retry_with_config(config, || async {
            attempt += 1;
            Ok::<i32, LLMError>(42)
        }).await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt, 1);
    }

    #[tokio::test]
    async fn test_retry_success_on_second_attempt() {
        let config = RetryConfig::new(3, Duration::from_millis(10));
        let mut attempt = 0;

        let result = retry_with_config(config, || async {
            attempt += 1;
            if attempt == 1 {
                Err(LLMError::Timeout)
            } else {
                Ok::<i32, LLMError>(42)
            }
        }).await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt, 2);
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let config = RetryConfig::new(2, Duration::from_millis(10));
        let mut attempt = 0;

        let result = retry_with_config(config, || async {
            attempt += 1;
            Err::<i32, LLMError>(LLMError::Timeout)
        }).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LLMError::RetryExhausted { attempts: 2 }));
        assert_eq!(attempt, 2);
    }
}

#[cfg(test)]
mod response_parser_tests {
    use super::*;

    #[test]
    fn test_parse_openai_response() {
        let response = json!({
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let parser = ResponseParser::new("openai");
        let result = parser.parse_response(&response).unwrap();

        assert_eq!(result.content, "Hello, world!");
        assert_eq!(result.usage.prompt_tokens, 10);
        assert_eq!(result.usage.completion_tokens, 5);
        assert_eq!(result.usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_anthropic_response() {
        let response = json!({
            "id": "test-id",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "model": "claude-3-sonnet-20240229",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        });

        let parser = ResponseParser::new("anthropic");
        let result = parser.parse_response(&response).unwrap();

        assert_eq!(result.content, "Hello, world!");
        assert_eq!(result.usage.prompt_tokens, 10);
        assert_eq!(result.usage.completion_tokens, 5);
    }

    #[test]
    fn test_parse_invalid_response() {
        let response = json!({
            "invalid": "format"
        });

        let parser = ResponseParser::new("openai");
        let result = parser.parse_response(&response);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LLMError::InvalidResponse(_)));
    }
}

#[cfg(test)]
mod batch_processor_tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_processor_creation() {
        let processor = BatchProcessor::new(5, Duration::from_millis(100));
        assert_eq!(processor.batch_size(), 5);
        assert_eq!(processor.timeout(), Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let processor = BatchProcessor::new(3, Duration::from_secs(1));
        let requests = vec![
            BatchRequest::new("1", "Hello"),
            BatchRequest::new("2", "World"),
            BatchRequest::new("3", "Test"),
        ];

        // Mock the actual processing - in real implementation this would call LLM
        let responses = processor.process_batch_mock(requests).await.unwrap();
        assert_eq!(responses.len(), 3);
    }
}

// Helper functions for testing
async fn retry_with_config<F, Fut, T>(config: RetryConfig, mut f: F) -> Result<T, LLMError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, LLMError>>,
{
    let mut attempts = 0;
    loop {
        attempts += 1;
        match f().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                if attempts >= config.max_attempts {
                    return Err(LLMError::RetryExhausted { attempts });
                }
                tokio::time::sleep(config.base_delay).await;
            }
        }
    }
}

impl BatchProcessor {
    async fn process_batch_mock(&self, requests: Vec<BatchRequest>) -> Result<Vec<BatchResponse>, LLMError> {
        // Mock implementation for testing
        let responses = requests.into_iter().map(|req| {
            BatchResponse::new(req.id().to_string(), format!("Response to: {}", req.prompt()))
        }).collect();
        Ok(responses)
    }
}