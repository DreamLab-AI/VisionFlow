use anyhow::Result;
use std::env;

#[cfg(feature = "ontology")]
use webxr::services::ontology_downloader::{OntologyDownloader, OntologyDownloaderConfig};
#[cfg(feature = "ontology")]
use webxr::services::ontology_storage::OntologyStorage;
#[cfg(feature = "ontology")]
use webxr::services::ontology_sync::OntologySync;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    #[cfg(not(feature = "ontology"))]
    {
        println!("This example requires the 'ontology' feature to be enabled.");
        println!("Run with: cargo run --example ontology_downloader_example --features ontology");
        return Ok(());
    }

    #[cfg(feature = "ontology")]
    {
        println!("=== Ontology Downloader Example ===\n");

        let github_token = env::var("GITHUB_TOKEN")
            .or_else(|_| env::var("GH_TOKEN"))
            .unwrap_or_else(|_| {
                println!("Warning: No GitHub token found. Using provided token.");
                "xxxxxxxxxxxxxxxxxxxxx".to_string()
            });

        let config = OntologyDownloaderConfig {
            github_token,
            repo_owner: "jjohare".to_string(),
            repo_name: "logseq".to_string(),
            base_path: "mainKnowledgeGraph/pages".to_string(),
            max_retries: 3,
            initial_retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            request_timeout_secs: 30,
            respect_rate_limits: true,
        };

        println!("Configuration:");
        println!("  Repository: {}/{}", config.repo_owner, config.repo_name);
        println!("  Base Path: {}", config.base_path);
        println!("  Max Retries: {}", config.max_retries);
        println!();

        println!("--- Option 1: Direct Download ---\n");
        example_direct_download(config.clone()).await?;

        println!("\n--- Option 2: Download with Storage ---\n");
        example_download_with_storage(config.clone()).await?;

        println!("\n--- Option 3: Full Sync with Progress ---\n");
        example_full_sync(config.clone()).await?;
    }

    Ok(())
}

#[cfg(feature = "ontology")]
async fn example_direct_download(config: OntologyDownloaderConfig) -> Result<()> {
    println!("Creating downloader...");
    let downloader = OntologyDownloader::new(config)?;

    println!("Starting download...");
    let blocks = downloader.download_all().await?;

    println!("\nDownload Results:");
    println!("  Total blocks found: {}", blocks.len());

    if !blocks.is_empty() {
        println!("\nFirst block details:");
        let first_block = &blocks[0];
        println!("  ID: {}", first_block.id);
        println!("  Title: {}", first_block.title);
        println!("  Source: {}", first_block.source_file);
        println!("  Classes: {}", first_block.classes.len());
        println!("  Properties: {}", first_block.properties.len());
        println!("  Relationships: {}", first_block.relationships.len());

        if !first_block.classes.is_empty() {
            println!("  Sample classes:");
            for class in first_block.classes.iter().take(3) {
                println!("    - {}", class);
            }
        }

        if !first_block.relationships.is_empty() {
            println!("  Sample relationships:");
            for rel in first_block.relationships.iter().take(3) {
                println!("    - {} -> {} -> {}", rel.subject, rel.predicate, rel.object);
            }
        }
    }

    let progress = downloader.get_progress().await;
    println!("\nProgress Summary:");
    println!("  Files processed: {}/{}", progress.processed_files, progress.total_files);
    println!("  Completion: {:.1}%", progress.percentage());
    println!("  Errors: {}", progress.errors.len());

    if !progress.errors.is_empty() {
        println!("\nErrors encountered:");
        for (idx, error) in progress.errors.iter().take(5).enumerate() {
            println!("  {}. {}", idx + 1, error);
        }
    }

    Ok(())
}

#[cfg(feature = "ontology")]
async fn example_download_with_storage(config: OntologyDownloaderConfig) -> Result<()> {
    println!("Creating downloader and storage...");
    let downloader = OntologyDownloader::new(config)?;
    let storage = OntologyStorage::new("ontology.db")?;

    println!("Downloading ontology data...");
    let blocks = downloader.download_all().await?;

    println!("Saving to database...");
    let saved_count = storage.save_blocks(&blocks)?;

    println!("\nStorage Results:");
    println!("  Blocks saved: {}", saved_count);

    let stats = storage.get_statistics()?;
    println!("\nDatabase Statistics:");
    println!("  Total blocks: {}", stats.total_blocks);
    println!("  Total classes: {}", stats.total_classes);
    println!("  Total properties: {}", stats.total_properties);
    println!("  Total relationships: {}", stats.total_relationships);

    println!("\nSearching for blocks with 'Avatar' class...");
    let avatar_blocks = storage.search_by_class("Avatar")?;
    println!("  Found {} blocks", avatar_blocks.len());

    if !avatar_blocks.is_empty() {
        println!("\nFirst matching block:");
        let block = &avatar_blocks[0];
        println!("  Title: {}", block.title);
        println!("  Classes: {:?}", block.classes);
    }

    Ok(())
}

#[cfg(feature = "ontology")]
async fn example_full_sync(config: OntologyDownloaderConfig) -> Result<()> {
    println!("Creating sync system...");
    let storage = OntologyStorage::new("ontology_sync.db")?;
    let sync = OntologySync::new(config, storage)?;

    println!("Starting synchronization...");

    let sync_handle = tokio::spawn({
        let sync = sync.clone();
        async move { sync.sync().await }
    });

    let progress_handle = tokio::spawn({
        let sync = sync.clone();
        async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                let progress = sync.get_progress().await;

                if progress.total_files > 0 {
                    println!(
                        "  Progress: {}/{} files ({:.1}%), {} blocks found",
                        progress.processed_files,
                        progress.total_files,
                        progress.percentage(),
                        progress.ontology_blocks_found
                    );

                    if let Some(ref current_file) = progress.current_file {
                        println!("  Current: {}", current_file);
                    }
                }

                if progress.completed_at.is_some() {
                    break;
                }
            }
        }
    });

    let result = sync_handle.await??;
    let _ = progress_handle.await;

    println!("\nSync Results:");
    println!("  Blocks downloaded: {}", result.blocks_downloaded);
    println!("  Blocks saved: {}", result.blocks_saved);
    println!("  Duration: {} seconds", result.duration_seconds);
    println!("  Errors: {}", result.errors.len());

    println!("\nFinal Statistics:");
    println!("  Total blocks: {}", result.statistics.total_blocks);
    println!("  Total classes: {}", result.statistics.total_classes);
    println!("  Total properties: {}", result.statistics.total_properties);
    println!("  Total relationships: {}", result.statistics.total_relationships);

    if let Some(last_sync) = result.statistics.last_sync_time {
        println!("  Last sync: {}", last_sync);
    }

    if !result.errors.is_empty() {
        println!("\nErrors during sync:");
        for (idx, error) in result.errors.iter().take(5).enumerate() {
            println!("  {}. {}", idx + 1, error);
        }
        if result.errors.len() > 5 {
            println!("  ... and {} more", result.errors.len() - 5);
        }
    }

    Ok(())
}
