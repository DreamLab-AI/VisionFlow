# Ontology Downloader Service

Complete production-ready service for downloading and processing ontology data from GitHub repositories.

## Overview

The Ontology Downloader system provides:

- **GitHub Integration**: Downloads markdown files containing OWL ontology definitions
- **Intelligent Filtering**: Processes only files with `OntologyBlock` marker and `public:: true` gate
- **Parsing**: Extracts structured ontology data (classes, properties, relationships)
- **Storage**: SQLite database with comprehensive schema and search capabilities
- **Retry Logic**: Exponential backoff with configurable retry parameters
- **Progress Tracking**: Real-time progress monitoring during downloads
- **Rate Limiting**: Respects GitHub API rate limits automatically

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OntologySync                              │
│  High-level orchestration of download and storage            │
└─────────────────────────────────────────────────────────────┘
                    │                    │
        ┌───────────┴────────┐  ┌───────┴──────────┐
        │                    │  │                   │
┌───────▼────────┐  ┌────────▼──▼─────┐  ┌─────────▼─────────┐
│ OntologyDownloader│  │ OntologyStorage │  │  Progress Tracking │
│                    │  │                 │  │                   │
│ - GitHub API       │  │ - SQLite DB     │  │ - Real-time stats │
│ - File scanning    │  │ - CRUD ops      │  │ - Error tracking  │
│ - Parsing          │  │ - Search        │  │ - Completion %    │
└────────────────────┘  └─────────────────┘  └───────────────────┘
```

## Components

### 1. OntologyDownloader

Downloads and parses ontology files from GitHub.

**Features:**
- Repository file scanning (recursive)
- Framework file identification (ETSI, OntologyDefinition, PropertySchema)
- Public gate filtering (`public:: true`)
- OWL block extraction from markdown
- Class, property, and relationship extraction
- Exponential backoff retry logic
- Rate limit handling

**Configuration:**
```rust
OntologyDownloaderConfig {
    github_token: String,           // GitHub personal access token
    repo_owner: String,             // Repository owner (default: "jjohare")
    repo_name: String,              // Repository name (default: "logseq")
    base_path: String,              // Base path (default: "mainKnowledgeGraph/pages")
    max_retries: u32,               // Maximum retry attempts (default: 3)
    initial_retry_delay_ms: u64,    // Initial delay (default: 1000ms)
    max_retry_delay_ms: u64,        // Maximum delay (default: 30000ms)
    request_timeout_secs: u64,      // Request timeout (default: 30s)
    respect_rate_limits: bool,      // Honor GitHub rate limits (default: true)
}
```

### 2. OntologyStorage

SQLite-based persistent storage for ontology data.

**Schema:**
- `ontology_blocks`: Main block metadata
- `ontology_properties`: Logseq properties (key-value pairs)
- `ontology_owl_content`: OWL Functional Syntax blocks
- `ontology_classes`: Extracted OWL classes
- `ontology_owl_properties`: OWL properties (ObjectProperty, DataProperty)
- `ontology_relationships`: Semantic relationships (SubClassOf, Domain, Range, etc.)
- `sync_metadata`: Synchronization tracking

**Operations:**
- `save_block()`: Store a single ontology block
- `save_blocks()`: Batch save multiple blocks
- `get_block()`: Retrieve block by ID
- `list_all_blocks()`: Get all blocks
- `search_by_class()`: Find blocks containing specific classes
- `search_by_property()`: Find blocks with specific properties
- `delete_block()`: Remove a block
- `clear_all()`: Wipe all data
- `get_statistics()`: Database metrics

### 3. OntologySync

High-level orchestration combining download and storage.

**Features:**
- Coordinated download and storage
- Progress monitoring
- Error collection
- Statistics reporting
- Sync metadata tracking

## Usage Examples

### Basic Download

```rust
use webxr::services::ontology_downloader::{OntologyDownloader, OntologyDownloaderConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let config = OntologyDownloaderConfig::from_env()?;
    let downloader = OntologyDownloader::new(config)?;

    let blocks = downloader.download_all().await?;
    println!("Downloaded {} ontology blocks", blocks.len());

    Ok(())
}
```

### Download with Storage

```rust
use webxr::services::ontology_downloader::{OntologyDownloader, OntologyDownloaderConfig};
use webxr::services::ontology_storage::OntologyStorage;

#[tokio::main]
async fn main() -> Result<()> {
    let config = OntologyDownloaderConfig::from_env()?;
    let downloader = OntologyDownloader::new(config)?;
    let storage = OntologyStorage::new("ontology.db")?;

    let blocks = downloader.download_all().await?;
    let saved = storage.save_blocks(&blocks)?;

    println!("Saved {} blocks to database", saved);

    Ok(())
}
```

### Full Sync with Progress

```rust
use webxr::services::ontology_downloader::OntologyDownloaderConfig;
use webxr::services::ontology_storage::OntologyStorage;
use webxr::services::ontology_sync::OntologySync;

#[tokio::main]
async fn main() -> Result<()> {
    let config = OntologyDownloaderConfig::from_env()?;
    let storage = OntologyStorage::new("ontology.db")?;
    let sync = OntologySync::new(config, storage)?;

    // Start sync in background
    let sync_handle = tokio::spawn({
        let sync = sync.clone();
        async move { sync.sync().await }
    });

    // Monitor progress
    loop {
        let progress = sync.get_progress().await;
        println!("Progress: {:.1}%", progress.percentage());

        if progress.completed_at.is_some() {
            break;
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    let result = sync_handle.await??;
    println!("Sync complete: {} blocks saved", result.blocks_saved);

    Ok(())
}
```

### Search and Query

```rust
use webxr::services::ontology_storage::OntologyStorage;

fn search_examples() -> Result<()> {
    let storage = OntologyStorage::new("ontology.db")?;

    // Search by class name
    let avatar_blocks = storage.search_by_class("Avatar")?;
    println!("Found {} blocks with Avatar class", avatar_blocks.len());

    // Search by property
    let mature_blocks = storage.search_by_property("maturity")?;
    println!("Found {} blocks with maturity property", mature_blocks.len());

    // Get specific block
    if let Some(block) = storage.get_block("some:block:id")? {
        println!("Block title: {}", block.title);
        println!("Classes: {:?}", block.classes);
        println!("Relationships: {}", block.relationships.len());
    }

    // Statistics
    let stats = storage.get_statistics()?;
    println!("Total blocks: {}", stats.total_blocks);
    println!("Total classes: {}", stats.total_classes);
    println!("Total relationships: {}", stats.total_relationships);

    Ok(())
}
```

## Configuration

### Environment Variables

```bash
# Required
export GITHUB_TOKEN="ghp_your_token_here"

# Optional (with defaults)
export GITHUB_OWNER="jjohare"
export GITHUB_REPO="logseq"
export GITHUB_BASE_PATH="mainKnowledgeGraph/pages"
```

### Database Location

```rust
// File-based
let storage = OntologyStorage::new("path/to/ontology.db")?;

// In-memory (for testing)
let storage = OntologyStorage::in_memory()?;
```

## Data Model

### OntologyBlock

```rust
pub struct OntologyBlock {
    pub id: String,                          // Unique identifier
    pub source_file: String,                 // GitHub file path
    pub title: String,                       // Block title
    pub properties: HashMap<String, Vec<String>>,  // Logseq properties
    pub owl_content: Vec<String>,            // Raw OWL blocks
    pub classes: Vec<String>,                // Extracted classes
    pub properties_list: Vec<String>,        // OWL properties
    pub relationships: Vec<OntologyRelationship>,  // Semantic relationships
    pub downloaded_at: DateTime<Utc>,        // Download timestamp
    pub content_hash: String,                // Content hash for deduplication
}
```

### OntologyRelationship

```rust
pub struct OntologyRelationship {
    pub subject: String,                     // Source entity
    pub predicate: String,                   // Relationship predicate
    pub object: String,                      // Target entity
    pub relationship_type: RelationshipType, // Typed relationship
}

pub enum RelationshipType {
    SubClassOf,      // rdfs:subClassOf
    ObjectProperty,  // owl:ObjectProperty
    DataProperty,    // owl:DataProperty
    DisjointWith,    // owl:disjointWith
    EquivalentTo,    // owl:equivalentClass
    InverseOf,       // owl:inverseOf
    Domain,          // rdfs:domain
    Range,           // rdfs:range
    Other(String),   // Custom relationship
}
```

## Error Handling

The service includes comprehensive error handling:

```rust
pub enum DownloaderError {
    GitHubApi(String),        // GitHub API errors
    Network(reqwest::Error),  // Network failures
    Parse(String),            // Parsing errors
    Database(String),         // Database errors
    RateLimit(Duration),      // Rate limit exceeded (with retry info)
    Auth(String),             // Authentication failures
    Config(String),           // Configuration errors
}
```

## Rate Limiting

The downloader automatically handles GitHub rate limits:

1. Checks `X-RateLimit-Remaining` header
2. Respects `Retry-After` header
3. Calculates wait time from `X-RateLimit-Reset`
4. Pauses execution and retries automatically
5. Configurable via `respect_rate_limits` flag

## Performance

- **Parallel Processing**: Uses async/await for concurrent downloads
- **Connection Pooling**: Reuses HTTP connections
- **Batch Operations**: Database transactions for multiple blocks
- **Indexed Queries**: Optimized search with database indexes
- **Memory Efficiency**: Streaming downloads, no large buffers

## Testing

```bash
# Run unit tests
cargo test --lib ontology

# Run integration tests
cargo test --test ontology_downloader_integration_tests --features ontology

# Run example
cargo run --example ontology_downloader_example --features ontology
```

## Production Deployment

### Requirements

1. GitHub personal access token with repo read permissions
2. SQLite database storage location
3. Network access to api.github.com

### Recommended Settings

```rust
OntologyDownloaderConfig {
    max_retries: 5,
    initial_retry_delay_ms: 2000,
    max_retry_delay_ms: 60000,
    request_timeout_secs: 60,
    respect_rate_limits: true,
}
```

### Monitoring

Monitor these metrics:

- `progress.percentage()`: Download completion
- `progress.errors`: Error tracking
- `statistics.total_blocks`: Data volume
- `statistics.last_sync_time`: Freshness

### Backup

```bash
# Backup SQLite database
cp ontology.db ontology.db.backup

# Vacuum to optimize
sqlite3 ontology.db "VACUUM;"
```

## Security

- **Token Storage**: Never commit tokens to source control
- **Environment Variables**: Use `.env` files or secure secret management
- **Rate Limiting**: Prevents API abuse
- **Input Validation**: All GitHub responses validated
- **SQL Injection**: Parameterized queries prevent injection

## Troubleshooting

### Authentication Errors

```
Error: Authentication error: Invalid GitHub token
```

**Solution**: Verify token is valid and has repo read permissions

### Rate Limit Exceeded

```
Error: Rate limit exceeded, retry after: 3600s
```

**Solution**: Wait for rate limit reset or use authenticated requests (higher limits)

### Parse Errors

```
Error: Parse error: Failed to extract OWL blocks
```

**Solution**: Check markdown format matches expected structure

### Database Locked

```
Error: Database error: database is locked
```

**Solution**: Ensure only one process accesses database, or use connection pooling

## Future Enhancements

- [ ] Incremental sync (only download changed files)
- [ ] Parallel file downloads
- [ ] Compression for storage efficiency
- [ ] GraphQL API for advanced queries
- [ ] Real-time sync via GitHub webhooks
- [ ] Multi-repository support
- [ ] Export to RDF/Turtle formats
- [ ] Validation against OWL schemas

## License

MIT
