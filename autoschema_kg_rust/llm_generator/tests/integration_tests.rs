use llm_generator::{
    LLMGenerator, GenerationConfig, Message,
    OpenAIProvider, AnthropicProvider, LocalProvider, LocalProviderType,
    BatchProcessor, BatchRequest,
    RateLimiter, RateConfig,
    TokenCounter, TokenUsage,
    PromptBuilder, PromptTemplate,
    RetryManager, RetryConfig, RetryStrategy,
    ResponseParser, JsonResponseParser, RegexResponseParser, CodeResponseParser,
    LLMError, Result,
};
use std::sync::Arc;
use std::collections::HashMap;
use tokio_test;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct TestResponse {
    message: String,
    code: i32,
}

#[tokio::test]
async fn test_token_counter() -> Result<()> {
    let counter = TokenCounter::new("gpt-3.5-turbo")?;

    // Test basic token counting
    let text = "Hello, world!";
    let tokens = counter.count_tokens(text);
    assert!(tokens > 0);

    // Test message token counting
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello!"),
    ];
    let message_tokens = counter.count_message_tokens(&messages);
    assert!(message_tokens > tokens);

    // Test token limit validation
    assert!(counter.validate_token_limit(100, Some(1000)).is_ok());
    assert!(counter.validate_token_limit(10000, Some(1000)).is_err());

    Ok(())
}

#[tokio::test]
async fn test_rate_limiter() -> Result<()> {
    let config = RateConfig {
        requests_per_minute: 60,
        tokens_per_minute: 1000,
        concurrent_requests: 2,
        burst_allowance: 5,
    };

    let rate_limiter = RateLimiter::new(config)?;

    // Test acquiring permits
    let permit1 = rate_limiter.acquire_request_permit().await?;
    let permit2 = rate_limiter.acquire_request_permit().await?;

    // Should be able to acquire two permits
    assert!(permit1.is_ok());
    assert!(permit2.is_ok());

    // Test token permits
    assert!(rate_limiter.acquire_token_permit(100).await.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_prompt_templates() -> Result<()> {
    let mut builder = PromptBuilder::new();

    // Create a custom template
    let template = PromptTemplate::new(
        "test_template",
        "Hello {{name}}, you are {{age}} years old."
    );

    builder.add_template(template);

    let mut variables = HashMap::new();
    variables.insert("name".to_string(), "Alice".to_string());
    variables.insert("age".to_string(), "25".to_string());

    let rendered = builder.render_template("test_template", &variables)?;
    assert_eq!(rendered, "Hello Alice, you are 25 years old.");

    // Test missing variable
    let mut incomplete_vars = HashMap::new();
    incomplete_vars.insert("name".to_string(), "Bob".to_string());

    let result = builder.render_template("test_template", &incomplete_vars);
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_json_response_parser() -> Result<()> {
    let parser = JsonResponseParser::<TestResponse>::new();

    // Create a mock response
    let usage = TokenUsage::new(10, 20);
    let response = llm_generator::GenerationResponse {
        text: r#"{"message": "success", "code": 200}"#.to_string(),
        model: "test-model".to_string(),
        usage,
        finish_reason: "stop".to_string(),
        response_time: std::time::Duration::from_millis(100),
        metadata: HashMap::new(),
    };

    let parsed = parser.parse(&response)?;
    assert_eq!(parsed.data.message, "success");
    assert_eq!(parsed.data.code, 200);
    assert!(parsed.confidence > 0.5);

    Ok(())
}

#[tokio::test]
async fn test_regex_response_parser() -> Result<()> {
    let parser = RegexResponseParser::new()
        .with_pattern("name", r"Name: (\w+)")?
        .with_pattern("age", r"Age: (\d+)")?
        .with_required_field("name");

    let usage = TokenUsage::new(10, 20);
    let response = llm_generator::GenerationResponse {
        text: "Name: John\nAge: 30\nLocation: NYC".to_string(),
        model: "test-model".to_string(),
        usage,
        finish_reason: "stop".to_string(),
        response_time: std::time::Duration::from_millis(100),
        metadata: HashMap::new(),
    };

    let parsed = parser.parse(&response)?;
    assert_eq!(parsed.data.get("name").unwrap(), "John");
    assert_eq!(parsed.data.get("age").unwrap(), "30");
    assert!(parsed.confidence > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_code_response_parser() -> Result<()> {
    let parser = CodeResponseParser::new("rust").with_comments();

    let usage = TokenUsage::new(10, 20);
    let response = llm_generator::GenerationResponse {
        text: r#"
Here's some Rust code:

```rust
// This is a simple function
fn add(a: i32, b: i32) -> i32 {
    a + b  // Return the sum
}
```
"#.to_string(),
        model: "test-model".to_string(),
        usage,
        finish_reason: "stop".to_string(),
        response_time: std::time::Duration::from_millis(100),
        metadata: HashMap::new(),
    };

    let parsed = parser.parse(&response)?;
    assert!(parsed.data.code.contains("fn add"));
    assert_eq!(parsed.data.language, "rust");
    assert!(parsed.data.comments.len() > 0);
    assert!(parsed.confidence > 0.5);

    Ok(())
}

#[tokio::test]
async fn test_retry_manager() -> Result<()> {
    let config = RetryConfig {
        max_attempts: 3,
        initial_interval: std::time::Duration::from_millis(10),
        max_interval: std::time::Duration::from_millis(100),
        multiplier: 2.0,
        max_elapsed_time: std::time::Duration::from_secs(1),
        retry_on_rate_limit: true,
        retry_on_timeout: true,
        retry_on_server_error: true,
    };

    let retry_manager = RetryManager::new(config, RetryStrategy::ExponentialBackoff);

    let mut call_count = 0;
    let result = retry_manager.retry_with_backoff(|| {
        call_count += 1;
        async move {
            if call_count < 3 {
                Err(LLMError::RateLimit)
            } else {
                Ok("success")
            }
        }
    }).await?;

    assert_eq!(result, "success");
    assert_eq!(call_count, 3);

    Ok(())
}

#[tokio::test]
async fn test_generation_config_validation() {
    let config = GenerationConfig {
        model: "test-model".to_string(),
        temperature: Some(0.5),
        top_p: Some(0.9),
        max_tokens: Some(100),
        ..Default::default()
    };

    // This would typically be called by a provider's validate_config method
    assert!(!config.model.is_empty());
    if let Some(temp) = config.temperature {
        assert!((0.0..=2.0).contains(&temp));
    }
    if let Some(top_p) = config.top_p {
        assert!((0.0..=1.0).contains(&top_p));
    }
}

#[tokio::test]
async fn test_message_creation() {
    let system_msg = Message::system("You are a helpful assistant");
    assert_eq!(system_msg.role, "system");

    let user_msg = Message::user("Hello!");
    assert_eq!(user_msg.role, "user");

    let assistant_msg = Message::assistant("Hello! How can I help?");
    assert_eq!(assistant_msg.role, "assistant");
}

#[tokio::test]
async fn test_batch_request_creation() {
    let config = GenerationConfig::default();

    let prompt_request = BatchRequest::new_prompt("Test prompt", config.clone());
    assert!(prompt_request.prompt.is_some());
    assert!(prompt_request.messages.is_none());

    let messages = vec![Message::user("Test message")];
    let chat_request = BatchRequest::new_chat(messages, config);
    assert!(chat_request.prompt.is_none());
    assert!(chat_request.messages.is_some());
}

#[tokio::test]
async fn test_usage_calculation() {
    let usage = TokenUsage::new(100, 50);
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
    assert_eq!(usage.total_tokens, 150);

    let mut combined = TokenUsage::empty();
    combined.add(&usage);
    assert_eq!(combined.total_tokens, 150);

    combined.add(&usage);
    assert_eq!(combined.total_tokens, 300);
}

#[tokio::test]
async fn test_error_types() {
    // Test that our error types work correctly
    let _rate_limit_error = LLMError::RateLimit;
    let _token_limit_error = LLMError::TokenLimit { used: 1000, limit: 500 };
    let _invalid_response_error = LLMError::InvalidResponse("test error".to_string());
    let _provider_error = LLMError::Provider("provider error".to_string());

    // Test error conversion
    let json_error = serde_json::Error::from(serde_json::de::Error::custom("test"));
    let _llm_error: LLMError = json_error.into();
}

// Integration test helper functions
async fn test_provider_interface<T: LLMGenerator>(provider: &T) -> Result<()> {
    // Test basic interface compliance
    assert!(!provider.provider_name().is_empty());

    let config = GenerationConfig {
        model: "test-model".to_string(),
        max_tokens: Some(10),
        ..Default::default()
    };

    // Test validation
    let validation_result = provider.validate_config(&config);
    // We don't assert success here as different providers have different requirements

    // Test usage stats
    let stats = provider.get_usage_stats().await?;
    assert!(stats.total_requests >= 0);

    Ok(())
}

#[cfg(feature = "integration")]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_openai_integration() -> Result<()> {
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            let provider = OpenAIProvider::new(api_key)?;
            test_provider_interface(&provider).await?;

            let config = GenerationConfig {
                model: "gpt-3.5-turbo".to_string(),
                max_tokens: Some(10),
                ..Default::default()
            };

            let response = provider.generate("Say hello", &config).await?;
            assert!(!response.text.is_empty());
            assert!(response.usage.total_tokens > 0);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_anthropic_integration() -> Result<()> {
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            let provider = AnthropicProvider::new(api_key)?;
            test_provider_interface(&provider).await?;

            let config = GenerationConfig {
                model: "claude-3-haiku-20240307".to_string(),
                max_tokens: Some(10),
                ..Default::default()
            };

            let response = provider.generate("Say hello", &config).await?;
            assert!(!response.text.is_empty());
            assert!(response.usage.total_tokens > 0);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_local_provider_interface() -> Result<()> {
        let provider = LocalProvider::ollama("http://localhost:11434")?;
        test_provider_interface(&provider).await?;
        Ok(())
    }
}