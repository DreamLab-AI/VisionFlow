use llm_generator::{
    OpenAIProvider, AnthropicProvider, LocalProvider, LocalProviderType,
    LLMGenerator, GenerationConfig, Message,
    BatchProcessor, BatchRequest,
    RateLimiter, RateConfig,
    TokenCounter, PromptBuilder,
    RetryManager, RetryConfig, RetryStrategy,
};
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Example 1: Basic OpenAI usage
    println!("=== OpenAI Example ===");
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let openai = OpenAIProvider::new(api_key)?;

        let config = GenerationConfig {
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            ..Default::default()
        };

        let response = openai.generate("Tell me a joke about programming", &config).await?;
        println!("OpenAI Response: {}", response.text);
        println!("Tokens used: {}", response.usage.total_tokens);
    } else {
        println!("OpenAI API key not found, skipping OpenAI example");
    }

    // Example 2: Anthropic usage
    println!("\n=== Anthropic Example ===");
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        let anthropic = AnthropicProvider::new(api_key)?;

        let config = GenerationConfig {
            model: "claude-3-haiku-20240307".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            ..Default::default()
        };

        let response = anthropic.generate("Explain async/await in Rust", &config).await?;
        println!("Anthropic Response: {}", response.text);
        println!("Tokens used: {}", response.usage.total_tokens);
    } else {
        println!("Anthropic API key not found, skipping Anthropic example");
    }

    // Example 3: Local Ollama usage
    println!("\n=== Local Ollama Example ===");
    let ollama = LocalProvider::ollama("http://localhost:11434")?
        .with_model("llama2");

    // Test if Ollama is available
    match ollama.list_models().await {
        Ok(models) => {
            println!("Available Ollama models: {:?}", models);

            let config = GenerationConfig {
                model: "llama2".to_string(),
                max_tokens: Some(50),
                temperature: Some(0.7),
                ..Default::default()
            };

            if let Ok(response) = ollama.generate("What is Rust?", &config).await {
                println!("Ollama Response: {}", response.text);
            }
        }
        Err(_) => {
            println!("Ollama not available, skipping local example");
        }
    }

    // Example 4: Chat conversation
    println!("\n=== Chat Conversation Example ===");
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let openai = OpenAIProvider::new(api_key)?;

        let messages = vec![
            Message::system("You are a helpful programming assistant."),
            Message::user("How do I handle errors in Rust?"),
        ];

        let config = GenerationConfig {
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(150),
            ..Default::default()
        };

        let response = openai.generate_chat(&messages, &config).await?;
        println!("Chat Response: {}", response.text);
    }

    // Example 5: Streaming
    println!("\n=== Streaming Example ===");
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let openai = OpenAIProvider::new(api_key)?;

        let config = GenerationConfig {
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(100),
            stream: true,
            ..Default::default()
        };

        print!("Streaming: ");
        let response = openai.generate_stream(
            "Count from 1 to 10",
            &config,
            |chunk| {
                print!("{}", chunk);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                Ok(())
            }
        ).await?;
        println!("\nFinal response length: {}", response.text.len());
    }

    // Example 6: Batch processing
    println!("\n=== Batch Processing Example ===");
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let openai = Arc::new(OpenAIProvider::new(api_key)?);
        let token_counter = Arc::new(TokenCounter::new("gpt-3.5-turbo")?);

        let rate_limiter = Arc::new(RateLimiter::new(RateConfig::openai_free())?);
        let retry_manager = Arc::new(RetryManager::new(
            RetryConfig::default(),
            RetryStrategy::ExponentialBackoff,
        ));

        let batch_processor = BatchProcessor::new(openai, token_counter)
            .with_rate_limiter(rate_limiter)
            .with_retry_manager(retry_manager)
            .with_max_concurrent(3);

        let requests = vec![
            BatchRequest::new_prompt(
                "What is machine learning?",
                GenerationConfig {
                    model: "gpt-3.5-turbo".to_string(),
                    max_tokens: Some(50),
                    ..Default::default()
                }
            ),
            BatchRequest::new_prompt(
                "Explain quantum computing",
                GenerationConfig {
                    model: "gpt-3.5-turbo".to_string(),
                    max_tokens: Some(50),
                    ..Default::default()
                }
            ),
            BatchRequest::new_prompt(
                "What is blockchain?",
                GenerationConfig {
                    model: "gpt-3.5-turbo".to_string(),
                    max_tokens: Some(50),
                    ..Default::default()
                }
            ),
        ];

        let result = batch_processor.process_batch(requests).await?;
        println!("Batch processing completed:");
        println!("  Successful: {}", result.successful.len());
        println!("  Failed: {}", result.failed.len());
        println!("  Throughput: {:.2} requests/sec", result.throughput);
        println!("  Success rate: {:.2}%", result.success_rate() * 100.0);
    }

    // Example 7: Prompt templates
    println!("\n=== Prompt Template Example ===");
    let mut prompt_builder = PromptBuilder::with_default_templates();

    let mut variables = HashMap::new();
    variables.insert("context".to_string(), "Rust is a systems programming language".to_string());
    variables.insert("question".to_string(), "What makes Rust memory safe?".to_string());

    let rendered = prompt_builder.render_template("question_answer", &variables)?;
    println!("Rendered template:\n{}", rendered);

    // Example 8: Token counting
    println!("\n=== Token Counting Example ===");
    let token_counter = TokenCounter::new("gpt-3.5-turbo")?;

    let text = "This is a sample text for token counting in Rust.";
    let token_count = token_counter.count_tokens(text);
    println!("Text: '{}'", text);
    println!("Token count: {}", token_count);

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello, how are you?"),
    ];
    let message_tokens = token_counter.count_message_tokens(&messages);
    println!("Message token count: {}", message_tokens);

    println!("\n=== All examples completed! ===");
    Ok(())
}