# Ontology Downloader - Quick Start Guide

## Installation

The service is already integrated into the project with the `ontology` feature flag.

## Setup

### 1. Set GitHub Token

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxx"
```

Or create a `.env` file:
```
GITHUB_TOKEN=xxxxxxxxxxxxxxxxxxxxx
```

### 2. Build with Ontology Feature

```bash
cargo build --features ontology
```

## Quick Examples

### Download Only

```rust
use webxr::services::ontology_downloader::{OntologyDownloader, OntologyDownloaderConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = OntologyDownloaderConfig::with_token(
        "xxxxxxxxxxxxxxxxxxxxx".to_string()
    );

    let downloader = OntologyDownloader::new(config)?;
    let blocks = downloader.download_all().await?;

    println!("Downloaded {} blocks", blocks.len());
    Ok(())
}
```

### Download + Store

```rust
use webxr::services::ontology_downloader::{OntologyDownloader, OntologyDownloaderConfig};
use webxr::services::ontology_storage::OntologyStorage;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = OntologyDownloaderConfig::with_token(
        std::env::var("GITHUB_TOKEN")?
    );

    let downloader = OntologyDownloader::new(config)?;
    let storage = OntologyStorage::new("ontology.db")?;

    let blocks = downloader.download_all().await?;
    storage.save_blocks(&blocks)?;

    println!("Saved {} blocks", blocks.len());
    Ok(())
}
```

### Full Sync

```rust
use webxr::services::ontology_downloader::OntologyDownloaderConfig;
use webxr::services::ontology_storage::OntologyStorage;
use webxr::services::ontology_sync::OntologySync;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = OntologyDownloaderConfig::from_env()?;
    let storage = OntologyStorage::new("ontology.db")?;
    let sync = OntologySync::new(config, storage)?;

    let result = sync.sync().await?;

    println!("Synced {} blocks", result.blocks_saved);
    println!("Found {} classes", result.statistics.total_classes);
    Ok(())
}
```

## Run the Example

```bash
cargo run --example ontology_downloader_example --features ontology
```

## Query the Database

```rust
use webxr::services::ontology_storage::OntologyStorage;

fn main() -> anyhow::Result<()> {
    let storage = OntologyStorage::new("ontology.db")?;

    // Search by class
    let blocks = storage.search_by_class("Avatar")?;
    println!("Found {} blocks with Avatar class", blocks.len());

    // Get statistics
    let stats = storage.get_statistics()?;
    println!("Total blocks: {}", stats.total_blocks);
    println!("Total classes: {}", stats.total_classes);

    Ok(())
}
```

## File Locations

- **Downloader**: `/home/devuser/workspace/project/src/services/ontology_downloader.rs`
- **Storage**: `/home/devuser/workspace/project/src/services/ontology_storage.rs`
- **Sync**: `/home/devuser/workspace/project/src/services/ontology_sync.rs`
- **Example**: `/home/devuser/workspace/project/examples/ontology_downloader_example.rs`
- **Tests**: `/home/devuser/workspace/project/tests/ontology_downloader_integration_tests.rs`
- **Docs**: `/home/devuser/workspace/project/docs/ontology-downloader.md`

## Key Features

✅ GitHub repository scanning
✅ Public gate filtering (`public:: true`)
✅ OWL block parsing
✅ SQLite database storage
✅ Progress tracking
✅ Retry logic with exponential backoff
✅ Rate limit handling

## Configuration Options

```rust
OntologyDownloaderConfig {
    github_token: String,           // Required
    repo_owner: String,             // Default: "jjohare"
    repo_name: String,              // Default: "logseq"
    base_path: String,              // Default: "mainKnowledgeGraph/pages"
    max_retries: u32,               // Default: 3
    initial_retry_delay_ms: u64,    // Default: 1000
    max_retry_delay_ms: u64,        // Default: 30000
    request_timeout_secs: u64,      // Default: 30
    respect_rate_limits: bool,      // Default: true
}
```

## Common Operations

### Clear Database
```rust
storage.clear_all()?;
```

### Get Block by ID
```rust
if let Some(block) = storage.get_block("block:id")? {
    println!("Title: {}", block.title);
}
```

### List All Blocks
```rust
let blocks = storage.list_all_blocks()?;
```

### Search by Property
```rust
let blocks = storage.search_by_property("maturity")?;
```

### Delete Block
```rust
storage.delete_block("block:id")?;
```

## Troubleshooting

### Authentication Error
- Verify `GITHUB_TOKEN` is set correctly
- Check token has repo read permissions

### Rate Limited
- Service automatically retries after rate limit reset
- Set `respect_rate_limits: true` in config

### Database Locked
- Ensure only one process accesses database
- Close database connections when done

### Parse Errors
- Check markdown format matches expected structure
- Ensure files have `public:: true` and OntologyBlock markers

## Need Help?

See full documentation: `/home/devuser/workspace/project/docs/ontology-downloader.md`
