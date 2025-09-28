# Contributing Guidelines

Welcome to AutoSchema KG Rust! We appreciate your interest in contributing to this project. This guide will help you understand our development process, coding standards, and how to submit your contributions.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submission Process](#submission-process)
- [Code Review Process](#code-review-process)
- [Release Process](#release-process)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Rust 1.70+** installed via [rustup](https://rustup.rs/)
- **Git** for version control
- **Neo4j** for local development (optional, can use Docker)
- **Docker** and **Docker Compose** for testing
- **VSCode** or another Rust-compatible editor

### Setting Up Development Environment

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/autoschema_kg_rust.git
   cd autoschema_kg_rust
   ```

2. **Install Development Dependencies**
   ```bash
   # Install Rust tools
   rustup component add rustfmt clippy
   cargo install cargo-watch cargo-tarpaulin cargo-audit

   # Install pre-commit hooks (optional but recommended)
   pip install pre-commit
   pre-commit install
   ```

3. **Start Development Services**
   ```bash
   # Start Neo4j with Docker Compose
   docker-compose -f docker-compose.dev.yml up -d neo4j

   # Set up environment variables
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Build and Test**
   ```bash
   # Build the project
   cargo build

   # Run tests
   cargo test

   # Run with watching (for development)
   cargo watch -x test
   ```

## Development Environment

### Recommended Tools

#### VS Code Extensions
```json
{
  "recommendations": [
    "rust-lang.rust-analyzer",
    "vadimcn.vscode-lldb",
    "serayuzgur.crates",
    "tamasfe.even-better-toml",
    "ms-vscode.test-adapter-converter",
    "hbenl.vscode-test-explorer"
  ]
}
```

#### VS Code Settings
```json
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.cargo.buildScripts.enable": true,
  "rust-analyzer.procMacro.enable": true,
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "files.watcherExclude": {
    "**/target/**": true
  }
}
```

### Environment Configuration

#### Development Environment (`.env.development`)
```bash
# Application
APP_ENV=development
RUST_LOG=debug
RUST_BACKTRACE=1

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=devpassword

# LLM APIs (use test keys)
OPENAI_API_KEY=sk-test-key
ANTHROPIC_API_KEY=test-key

# Development settings
ENABLE_METRICS=false
LOG_LEVEL=debug
```

### Git Configuration

#### Git Hooks Setup
```bash
#!/bin/sh
# .git/hooks/pre-commit

set -e

echo "Running pre-commit checks..."

# Format code
cargo fmt --all -- --check
if [ $? -ne 0 ]; then
    echo "Please run 'cargo fmt' to format your code"
    exit 1
fi

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings
if [ $? -ne 0 ]; then
    echo "Please fix clippy warnings"
    exit 1
fi

# Run tests
cargo test --all-features
if [ $? -ne 0 ]; then
    echo "Tests failed"
    exit 1
fi

echo "Pre-commit checks passed!"
```

#### Git Configuration
```bash
git config --local core.autocrlf false
git config --local core.eol lf
git config --local pull.rebase true
```

## Coding Standards

### Rust Style Guide

We follow the [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/) with these additions:

#### Code Formatting
```bash
# Format all code
cargo fmt

# Check formatting without making changes
cargo fmt -- --check
```

#### Linting with Clippy
```bash
# Run clippy with all features
cargo clippy --all-targets --all-features -- -D warnings

# Fix auto-fixable issues
cargo clippy --fix --all-targets --all-features
```

### Code Organization

#### Module Structure
```rust
// Standard library imports first
use std::collections::HashMap;
use std::sync::Arc;

// External crate imports second
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

// Internal imports last
use crate::error::{Result, KgError};
use crate::types::Entity;

// Re-exports at the top of lib.rs
pub use entity::Entity;
pub use graph::Graph;
pub use error::{KgError, Result};
```

#### Naming Conventions
```rust
// Types: PascalCase
struct KnowledgeGraph;
enum ProcessingState;

// Functions and variables: snake_case
fn process_entities() -> Result<()>;
let entity_count = 42;

// Constants: SCREAMING_SNAKE_CASE
const MAX_BATCH_SIZE: usize = 1000;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

// Modules: snake_case
mod knowledge_graph;
mod llm_generator;
```

#### Error Handling
```rust
// Use Result types consistently
pub type Result<T> = std::result::Result<T, KgError>;

// Provide context for errors
fn process_file(path: &Path) -> Result<Vec<Entity>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| KgError::FileRead {
            path: path.to_path_buf(),
            source: e,
        })?;

    parse_entities(&content)
        .map_err(|e| KgError::ParseError {
            context: format!("Failed to parse file: {}", path.display()),
            source: e,
        })
}

// Use thiserror for error definitions
#[derive(Error, Debug)]
pub enum KgError {
    #[error("Failed to read file {path}")]
    FileRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Parse error: {context}")]
    ParseError {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
```

#### Documentation Standards
```rust
//! Module-level documentation
//!
//! This module provides functionality for processing knowledge graphs.
//! It includes utilities for entity extraction, relationship mapping,
//! and graph construction.

/// Processes entities from various data sources.
///
/// This function takes a data source and extracts entities according to
/// the provided configuration. It supports parallel processing for
/// improved performance on large datasets.
///
/// # Arguments
///
/// * `source` - The data source to process
/// * `config` - Processing configuration options
///
/// # Returns
///
/// Returns a `Result` containing the processed entities or an error.
///
/// # Errors
///
/// This function will return an error if:
/// - The data source cannot be read
/// - The data format is invalid
/// - Processing fails due to resource constraints
///
/// # Examples
///
/// ```
/// use autoschema_kg_rust::{DataSource, ProcessingConfig};
///
/// let source = DataSource::from_file("data.csv")?;
/// let config = ProcessingConfig::default();
/// let entities = process_entities(source, config).await?;
/// ```
pub async fn process_entities(
    source: DataSource,
    config: ProcessingConfig,
) -> Result<Vec<Entity>> {
    // Implementation
}
```

### Performance Guidelines

#### Memory Management
```rust
// Use appropriate data structures
use std::collections::HashMap; // For key-value lookups
use indexmap::IndexMap; // When insertion order matters
use dashmap::DashMap; // For concurrent access

// Pre-allocate when size is known
let mut entities = Vec::with_capacity(expected_size);
let mut map = HashMap::with_capacity(expected_size);

// Use references to avoid unnecessary clones
fn process_entity(entity: &Entity) -> Result<ProcessedEntity> {
    // Process without taking ownership
}

// Use Cow for optional cloning
use std::borrow::Cow;

fn process_text(text: Cow<str>) -> String {
    match text {
        Cow::Borrowed(s) => s.to_uppercase(),
        Cow::Owned(s) => s.to_uppercase(),
    }
}
```

#### Async Programming
```rust
// Use appropriate async patterns
use tokio::task;
use futures::future::try_join_all;

// Parallel processing
async fn process_batch(items: Vec<Item>) -> Result<Vec<Processed>> {
    let tasks: Vec<_> = items.into_iter()
        .map(|item| task::spawn(async move { process_item(item).await }))
        .collect();

    let results = try_join_all(tasks).await?;
    Ok(results.into_iter().collect::<Result<Vec<_>>>()?)
}

// Use streams for large datasets
use tokio_stream::{Stream, StreamExt};

async fn process_stream<S>(stream: S) -> Result<Vec<Processed>>
where
    S: Stream<Item = Item>,
{
    let processed: Vec<Processed> = stream
        .map(|item| process_item(item))
        .buffer_unordered(10) // Process up to 10 items concurrently
        .try_collect()
        .await?;

    Ok(processed)
}
```

## Testing Guidelines

### Test Organization

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new("test_id", "Person");
        assert_eq!(entity.id(), "test_id");
        assert_eq!(entity.entity_type(), "Person");
    }

    #[rstest]
    #[case("valid_id", "Person", true)]
    #[case("", "Person", false)]
    #[case("valid_id", "", false)]
    fn test_entity_validation(
        #[case] id: &str,
        #[case] entity_type: &str,
        #[case] expected_valid: bool,
    ) {
        let result = Entity::new(id, entity_type).validate();
        assert_eq!(result.is_ok(), expected_valid);
    }

    proptest! {
        #[test]
        fn test_entity_roundtrip_serialization(
            id in "[a-zA-Z0-9_]{1,100}",
            entity_type in "[a-zA-Z]{1,50}",
        ) {
            let original = Entity::new(&id, &entity_type);
            let serialized = serde_json::to_string(&original).unwrap();
            let deserialized: Entity = serde_json::from_str(&serialized).unwrap();
            assert_eq!(original, deserialized);
        }
    }
}
```

#### Integration Tests
```rust
// tests/integration_test.rs
use autoschema_kg_rust::prelude::*;
use serial_test::serial;
use testcontainers::*;

#[tokio::test]
#[serial]
async fn test_full_processing_pipeline() {
    // Set up test environment
    let _neo4j_container = start_neo4j_container().await;

    // Initialize components
    let config = TestConfig::default();
    let kg_builder = KnowledgeGraphBuilder::new(config.graph_config).await?;

    // Test data
    let test_data = create_test_dataset();

    // Process data
    let stats = kg_builder.process_dataset(&test_data, config.processing_config).await?;

    // Verify results
    assert!(stats.entities_processed > 0);
    assert!(stats.relationships_created > 0);
    assert_eq!(stats.errors.len(), 0);
}

async fn start_neo4j_container() -> Container<'static, Neo4j> {
    let docker = clients::Cli::default();
    let neo4j_image = images::generic::GenericImage::new("neo4j", "5.11")
        .with_env_var("NEO4J_AUTH", "neo4j/testpassword");

    docker.run(neo4j_image)
}
```

#### Benchmark Tests
```rust
// benches/processing_benchmarks.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use autoschema_kg_rust::*;

fn benchmark_entity_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_processing");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            size,
            |b, &size| {
                let entities = generate_test_entities(size);
                b.iter(|| process_entities_parallel(&entities))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_entity_processing);
criterion_main!(benches);
```

### Test Data Management

#### Test Utilities
```rust
// tests/common/mod.rs
use fake::{Fake, Faker};
use fake::faker::name::en::*;

pub struct TestDataBuilder {
    entity_count: usize,
    relationship_count: usize,
}

impl TestDataBuilder {
    pub fn new() -> Self {
        Self {
            entity_count: 100,
            relationship_count: 200,
        }
    }

    pub fn with_entities(mut self, count: usize) -> Self {
        self.entity_count = count;
        self
    }

    pub fn build(self) -> TestDataset {
        let entities: Vec<Entity> = (0..self.entity_count)
            .map(|i| {
                Entity::builder()
                    .id(format!("entity_{}", i))
                    .entity_type("Person")
                    .property("name", Name().fake::<String>())
                    .build()
            })
            .collect();

        TestDataset { entities }
    }
}

pub struct TestConfig {
    pub graph_config: GraphConfig,
    pub processing_config: ProcessingConfig,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            graph_config: GraphConfig::builder()
                .neo4j_uri("bolt://localhost:7688") // Test port
                .username("neo4j")
                .password("testpassword")
                .build(),
            processing_config: ProcessingConfig::builder()
                .batch_size(100)
                .parallel_workers(2)
                .build(),
        }
    }
}
```

### Mock and Test Doubles

```rust
use mockall::{automock, predicate::*};

#[automock]
pub trait LLMProvider {
    async fn generate(&self, prompt: &str) -> Result<String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generation_with_mock() {
        let mut mock_provider = MockLLMProvider::new();

        mock_provider
            .expect_generate()
            .with(eq("test prompt"))
            .times(1)
            .returning(|_| Ok("test response".to_string()));

        let generator = LLMGenerator::new(Box::new(mock_provider));
        let result = generator.generate("test prompt").await.unwrap();

        assert_eq!(result, "test response");
    }
}
```

## Documentation

### API Documentation

#### Rustdoc Guidelines
```rust
/// High-level module documentation explaining the purpose and usage.
///
/// # Examples
///
/// Basic usage:
/// ```
/// use autoschema_kg_rust::Entity;
///
/// let entity = Entity::new("id", "Person");
/// ```

/// Represents an entity in the knowledge graph.
///
/// An entity is a fundamental unit of information that can have
/// properties and relationships with other entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier for the entity
    pub id: String,
    /// Type classification of the entity
    pub entity_type: String,
}

impl Entity {
    /// Creates a new entity with the given ID and type.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the entity
    /// * `entity_type` - Type classification
    ///
    /// # Examples
    ///
    /// ```
    /// use autoschema_kg_rust::Entity;
    ///
    /// let entity = Entity::new("person_123", "Person");
    /// assert_eq!(entity.id(), "person_123");
    /// ```
    pub fn new(id: impl Into<String>, entity_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            entity_type: entity_type.into(),
        }
    }
}
```

#### README Updates
When adding new features, update the README with:
- Feature descriptions
- Usage examples
- Performance impact
- Breaking changes

### Code Examples

#### Example Structure
```
examples/
├── basic_usage.rs           # Simple getting started example
├── advanced_processing.rs   # Complex processing pipeline
├── custom_providers.rs      # Custom LLM provider implementation
├── performance_tuning.rs    # Performance optimization examples
└── data/                   # Example datasets
    ├── sample.csv
    ├── sample.json
    └── sample.graphml
```

#### Example Template
```rust
//! Basic usage example for AutoSchema KG Rust
//!
//! This example demonstrates how to:
//! - Set up a knowledge graph builder
//! - Process a CSV dataset
//! - Perform basic retrieval operations

use autoschema_kg_rust::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Step 1: Configure the knowledge graph builder
    let config = GraphConfig::builder()
        .neo4j_uri("bolt://localhost:7687")
        .username("neo4j")
        .password("password")
        .build();

    let builder = KnowledgeGraphBuilder::new(config).await?;

    // Step 2: Process a dataset
    let processing_config = ProcessingConfig::builder()
        .batch_size(1000)
        .parallel_workers(4)
        .build();

    let stats = builder
        .process_dataset("examples/data/sample.csv", processing_config)
        .await?;

    println!("Processed {} entities and {} relationships",
             stats.entities_processed,
             stats.relationships_created);

    Ok(())
}
```

## Submission Process

### Pull Request Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/add-new-functionality
   ```

2. **Make Changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation

3. **Test Locally**
   ```bash
   # Run all tests
   cargo test --all-features

   # Check formatting and linting
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features -- -D warnings

   # Run benchmarks if performance-related
   cargo bench

   # Test with different feature combinations
   cargo test --no-default-features
   cargo test --all-features
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new entity processing functionality

   - Implement parallel entity processing
   - Add support for custom validation rules
   - Include comprehensive tests and benchmarks

   Closes #123"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/add-new-functionality
   ```

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

#### Examples
```bash
feat(kg_construction): add batch processing support

Implement batch processing for large datasets to improve
memory usage and processing speed.

- Add BatchProcessor struct
- Implement configurable batch sizes
- Add progress reporting
- Include comprehensive tests

Closes #456

fix(retriever): resolve infinite loop in graph traversal

The graph traversal algorithm could enter an infinite loop
when circular references were present in the knowledge graph.

Breaking change: The traversal API now requires a max_depth parameter.

BREAKING CHANGE: traversal_method now requires max_depth parameter
```

### Pull Request Template

```markdown
## Description
Brief description of the changes and why they're needed.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Benchmarks run (if performance-related)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] Tests added for new functionality

## Breaking Changes
List any breaking changes and migration guide.

## Performance Impact
Describe any performance implications.
```

## Code Review Process

### Review Criteria

#### Functionality
- [ ] Code solves the stated problem
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Performance considerations are addressed

#### Code Quality
- [ ] Code is readable and well-structured
- [ ] Naming conventions are followed
- [ ] Comments explain the "why" not the "what"
- [ ] No unnecessary complexity

#### Testing
- [ ] Adequate test coverage
- [ ] Tests are meaningful and test the right things
- [ ] Integration tests for user-facing features
- [ ] Benchmarks for performance-critical code

#### Documentation
- [ ] Public APIs are documented
- [ ] Examples are provided where helpful
- [ ] Breaking changes are clearly marked

### Review Process

1. **Automated Checks**
   - CI pipeline runs all tests
   - Code formatting and linting checks
   - Security vulnerability scans
   - Performance regression tests

2. **Human Review**
   - At least one maintainer review required
   - Domain expert review for specialized areas
   - Focus on design, correctness, and maintainability

3. **Approval and Merge**
   - All checks must pass
   - Required reviews completed
   - No unresolved conversations
   - Squash and merge preferred

### Reviewer Guidelines

#### Providing Feedback
```markdown
# Good feedback examples

## Constructive
"This function could be more efficient using a HashMap instead of linear search.
Here's an example: [code snippet]"

## Specific
"Line 42: This error message could be more helpful by including the entity ID"

## Educational
"Consider using `?` operator here for cleaner error propagation"

# Avoid

## Non-constructive
"This is wrong"

## Vague
"This doesn't look right"

## Nitpicky
"Use single quotes instead of double quotes" (unless it's a style guide violation)
```

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version when you make incompatible API changes
- **MINOR** version when you add functionality in a backwards compatible manner
- **PATCH** version when you make backwards compatible bug fixes

### Release Checklist

#### Pre-release
- [ ] Update CHANGELOG.md
- [ ] Update version in Cargo.toml files
- [ ] Update documentation
- [ ] Run full test suite
- [ ] Performance benchmarks
- [ ] Security audit

#### Release
- [ ] Tag release in git
- [ ] Publish to crates.io
- [ ] Update documentation site
- [ ] Create GitHub release with notes
- [ ] Update Docker images

#### Post-release
- [ ] Monitor for issues
- [ ] Update dependent projects
- [ ] Communicate changes to users

### Changelog Format

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- New feature descriptions

### Changed
- Changed functionality descriptions

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security improvements

## [1.0.0] - 2023-12-01

### Added
- Initial release
- Knowledge graph construction
- Multi-hop retrieval
- LLM integration
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, announcements
- **Discord**: Real-time chat and support
- **Email**: Security issues and private matters

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Annual recognition posts

### Getting Help

- Check existing issues and discussions
- Read the documentation
- Ask questions in discussions
- Join our Discord community

## Advanced Topics

### Custom Providers

```rust
// Implementing a custom LLM provider
use async_trait::async_trait;

pub struct CustomProvider {
    endpoint: String,
    api_key: String,
}

#[async_trait]
impl LLMProvider for CustomProvider {
    async fn generate(&self, prompt: &str) -> Result<GenerationResponse> {
        // Implementation
    }
}
```

### Performance Optimization

```rust
// Example of performance-critical code
pub fn optimize_vector_similarity(
    query: &[f32],
    vectors: &[Vec<f32>],
) -> Vec<(usize, f32)> {
    // Use SIMD operations when available
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { similarity_avx2(query, vectors) };
        }
    }

    // Fallback implementation
    similarity_fallback(query, vectors)
}
```

### Security Considerations

- Validate all inputs
- Sanitize data before database operations
- Use secure defaults
- Follow OWASP guidelines
- Regular security audits

Thank you for contributing to AutoSchema KG Rust! Your contributions help make this project better for everyone.