# LLM Generator

A comprehensive Rust library for interfacing with Large Language Models (LLMs) from various providers, featuring async operations, rate limiting, batch processing, and robust error handling.

## Features

### 🚀 **Multiple Providers**
- **OpenAI** (GPT-3.5, GPT-4, GPT-4 Turbo)
- **Anthropic** (Claude 3 Opus, Sonnet, Haiku)
- **Local Models** (Ollama, llama.cpp, Oobabooga)

### ⚡ **Advanced Capabilities**
- **Async/Await** - Non-blocking operations with Tokio
- **Streaming Support** - Real-time token streaming
- **Batch Processing** - Efficient handling of multiple requests
- **Rate Limiting** - Built-in rate limiting with burst support
- **Retry Logic** - Exponential backoff with configurable strategies
- **Token Counting** - Accurate token estimation and validation
- **Usage Tracking** - Comprehensive usage statistics and cost calculation

### 🛠 **Developer Experience**
- **Prompt Templates** - Reusable prompt templates with variable substitution
- **Response Parsing** - JSON, regex, and code extraction utilities
- **Error Handling** - Detailed error types with context
- **Configuration** - Flexible configuration with validation
- **Testing** - Comprehensive test suite with mocking support

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
llm_generator = "0.1.0"
tokio = { version = "1.0", features = ["full"] }
```

### Basic Usage

```rust
use llm_generator::{OpenAIProvider, LLMGenerator, GenerationConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize provider
    let provider = OpenAIProvider::new("your-api-key")?;

    // Configure generation
    let config = GenerationConfig {
        model: "gpt-3.5-turbo".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        ..Default::default()
    };

    // Generate response
    let response = provider.generate("Tell me a joke about Rust", &config).await?;
    println!("Response: {}", response.text);
    println!("Tokens used: {}", response.usage.total_tokens);

    Ok(())
}
```

### Chat Conversations

```rust
use llm_generator::Message;

let messages = vec![
    Message::system("You are a helpful programming assistant."),
    Message::user("How do I handle errors in Rust?"),
];

let response = provider.generate_chat(&messages, &config).await?;
println!("Assistant: {}", response.text);
```

### Streaming

```rust
let response = provider.generate_stream(
    "Count from 1 to 10",
    &config,
    |chunk| {
        print!("{}", chunk);
        Ok(())
    }
).await?;
```

## Provider Setup

### OpenAI

```rust
use llm_generator::OpenAIProvider;

let provider = OpenAIProvider::new("sk-...")?
    .with_organization("org-...") // Optional
    .with_timeout(Duration::from_secs(30));
```

### Anthropic

```rust
use llm_generator::AnthropicProvider;

let provider = AnthropicProvider::new("sk-ant-...")?
    .with_version("2023-06-01")
    .with_timeout(Duration::from_secs(60));
```

### Local Models (Ollama)

```rust
use llm_generator::{LocalProvider, LocalProviderType};

let provider = LocalProvider::ollama("http://localhost:11434")?
    .with_model("llama2");

// List available models
let models = provider.list_models().await?;
println!("Available models: {:?}", models);
```

## Advanced Features

### Batch Processing

```rust
use llm_generator::{BatchProcessor, BatchRequest, RateLimiter, RateConfig};

let batch_processor = BatchProcessor::new(provider, token_counter)
    .with_rate_limiter(rate_limiter)
    .with_max_concurrent(5);

let requests = vec![
    BatchRequest::new_prompt("Question 1", config.clone()),
    BatchRequest::new_prompt("Question 2", config.clone()),
    BatchRequest::new_prompt("Question 3", config.clone()),
];

let result = batch_processor.process_batch(requests).await?;
println!("Success rate: {:.1}%", result.success_rate() * 100.0);
println!("Throughput: {:.2} req/sec", result.throughput);
```

### Rate Limiting

```rust
use llm_generator::{RateLimiter, RateConfig};

// OpenAI free tier limits
let rate_config = RateConfig::openai_free();
let rate_limiter = RateLimiter::new(rate_config)?;

// Custom rate limits
let custom_config = RateConfig {
    requests_per_minute: 60,
    tokens_per_minute: 90000,
    concurrent_requests: 5,
    burst_allowance: 10,
};
```

### Retry Logic

```rust
use llm_generator::{RetryManager, RetryConfig, RetryStrategy};

let retry_config = RetryConfig {
    max_attempts: 3,
    initial_interval: Duration::from_millis(100),
    max_interval: Duration::from_secs(60),
    multiplier: 2.0,
    retry_on_rate_limit: true,
    retry_on_timeout: true,
    retry_on_server_error: true,
    ..Default::default()
};

let retry_manager = RetryManager::new(retry_config, RetryStrategy::ExponentialBackoff);
```

### Prompt Templates

```rust
use llm_generator::{PromptBuilder, PromptTemplate};

let mut builder = PromptBuilder::with_default_templates();

// Custom template
let template = PromptTemplate::new(
    "code_review",
    "Review this {{language}} code:\n\n{{code}}\n\nProvide feedback on:\n{{criteria}}"
);
builder.add_template(template);

// Render with variables
let mut variables = HashMap::new();
variables.insert("language".to_string(), "Rust".to_string());
variables.insert("code".to_string(), "fn main() { println!(\"Hello\"); }".to_string());
variables.insert("criteria".to_string(), "style, performance, safety".to_string());

let prompt = builder.render_template("code_review", &variables)?;
```

### Response Parsing

```rust
use llm_generator::{JsonResponseParser, RegexResponseParser, CodeResponseParser};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct ApiResponse {
    status: String,
    data: Vec<String>,
}

// JSON parsing
let json_parser = JsonResponseParser::<ApiResponse>::new();
let parsed = json_parser.parse(&response)?;

// Regex extraction
let regex_parser = RegexResponseParser::new()
    .with_pattern("name", r"Name: (\w+)")?
    .with_pattern("age", r"Age: (\d+)")?;

// Code extraction
let code_parser = CodeResponseParser::new("rust").with_comments();
```

### Token Counting

```rust
use llm_generator::TokenCounter;

let counter = TokenCounter::new("gpt-3.5-turbo")?;

// Count tokens in text
let tokens = counter.count_tokens("Hello, world!");

// Count tokens in messages
let message_tokens = counter.count_message_tokens(&messages);

// Validate token limits
counter.validate_token_limit(prompt_tokens, Some(max_tokens))?;
```

### Usage Tracking

```rust
// Get usage statistics
let stats = provider.get_usage_stats().await?;
println!("Total requests: {}", stats.total_requests);
println!("Total cost: ${:.4}", stats.total_cost);
println!("Average response time: {:?}", stats.average_response_time);
println!("Error rate: {:.2}%", stats.error_rate * 100.0);

// Reset statistics
provider.reset_usage_stats().await?;
```

## Configuration

### Generation Config

```rust
use llm_generator::GenerationConfig;

let config = GenerationConfig {
    model: "gpt-4".to_string(),
    max_tokens: Some(2000),
    temperature: Some(0.8),
    top_p: Some(0.9),
    frequency_penalty: Some(0.1),
    presence_penalty: Some(0.1),
    stop_sequences: Some(vec!["END".to_string()]),
    stream: false,
    timeout: Some(Duration::from_secs(60)),
    custom_params: HashMap::new(),
};
```

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_ORGANIZATION="org-..."  # Optional

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Local models
export OLLAMA_HOST="http://localhost:11434"
```

## Error Handling

```rust
use llm_generator::{LLMError, Result};

match provider.generate("test", &config).await {
    Ok(response) => println!("Success: {}", response.text),
    Err(LLMError::RateLimit) => println!("Rate limit exceeded"),
    Err(LLMError::TokenLimit { used, limit }) => {
        println!("Token limit exceeded: {}/{}", used, limit);
    }
    Err(LLMError::InvalidApiKey) => println!("Invalid API key"),
    Err(LLMError::Timeout) => println!("Request timed out"),
    Err(e) => println!("Error: {}", e),
}
```

## Examples

See the `examples/` directory for more comprehensive examples:

- `basic_usage.rs` - Basic provider usage and configuration
- `advanced_batch.rs` - Batch processing with rate limiting
- `streaming_chat.rs` - Streaming conversations
- `template_system.rs` - Advanced prompt templating
- `response_parsing.rs` - Response parsing and validation

## Testing

```bash
# Unit tests
cargo test

# Integration tests (requires API keys)
OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... cargo test --features integration

# Benchmarks
cargo bench
```

## Performance

- **Concurrent Processing**: Handle multiple requests simultaneously
- **Connection Pooling**: Efficient HTTP connection reuse
- **Memory Efficient**: Streaming responses to minimize memory usage
- **Rate Limited**: Respect API limits to avoid throttling
- **Caching**: Token counting and model information caching

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Application   │───▶│ LLMGenerator │───▶│    Providers    │
└─────────────────┘    │   (Trait)    │    │ OpenAI/Anthropic│
                       └──────────────┘    │     /Local      │
                              │             └─────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   Middleware     │
                    │ • Rate Limiting  │
                    │ • Retry Logic    │
                    │ • Usage Tracking │
                    │ • Token Counting │
                    └──────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Changelog

### v0.1.0
- Initial release
- Support for OpenAI, Anthropic, and local providers
- Batch processing with rate limiting
- Prompt templates and response parsing
- Comprehensive usage tracking
- Streaming support
- Retry logic with exponential backoff