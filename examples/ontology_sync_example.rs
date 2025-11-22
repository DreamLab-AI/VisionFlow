// examples/ontology_sync_example.rs
//! Example: Enhanced Ontology Sync Service Usage
//!
//! Demonstrates the new features:
//! - Selective filtering by priority
//! - Metadata extraction (domains, topics)
//! - LRU caching
//! - GitHub commit date enrichment
//! - Batch statistics

use std::sync::Arc;
use visionflow::adapters::neo4j_ontology_repository::Neo4jOntologyRepository;
use visionflow::services::github::api::GitHubClient;
use visionflow::services::github::content_enhanced::EnhancedContentAPI;
use visionflow::services::github::types::OntologyPriority;
use visionflow::services::local_file_sync_service::LocalFileSyncService;
use visionflow::services::ontology_enrichment_service::OntologyEnrichmentService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🚀 Enhanced Ontology Sync Service Example");
    println!("==========================================\n");

    // Setup services (example - adjust to your configuration)
    let github_client = Arc::new(GitHubClient::new(
        std::env::var("GITHUB_TOKEN")?,
        std::env::var("GITHUB_OWNER").unwrap_or_else(|_| "jjohare".to_string()),
        std::env::var("GITHUB_REPO").unwrap_or_else(|_| "logseq".to_string()),
        std::env::var("GITHUB_BRANCH").unwrap_or_else(|_| "main".to_string()),
        std::env::var("GITHUB_BASE_PATH")
            .unwrap_or_else(|_| "mainKnowledgeGraph/pages".to_string()),
    ));

    let content_api = Arc::new(EnhancedContentAPI::new(github_client));

    // Note: You'll need to provide your actual repository implementations
    // This is just a placeholder
    let kg_repo: Arc<dyn visionflow::ports::knowledge_graph_repository::KnowledgeGraphRepository> =
        unimplemented!("Provide your KG repository");
    let onto_repo = Arc::new(Neo4jOntologyRepository::new("neo4j://localhost:7687", "neo4j", "password"));
    let enrichment_service = Arc::new(OntologyEnrichmentService::new(/* ... */));

    let sync_service = LocalFileSyncService::new(
        content_api.clone(),
        kg_repo,
        onto_repo,
        enrichment_service,
    );

    // ==================
    // 1. Run Initial Sync
    // ==================
    println!("📥 Step 1: Running initial sync with ontology enhancements...\n");

    let stats = sync_service.sync_with_github_delta().await?;

    println!("\n✅ Sync Complete!\n");
    println!("Basic Statistics:");
    println!("  Total files scanned: {}", stats.total_files);
    println!("  Files synced from local: {}", stats.files_synced_from_local);
    println!("  Files updated from GitHub: {}", stats.files_updated_from_github);
    println!("  Knowledge graph files: {}", stats.kg_files_processed);
    println!("  Ontology files: {}", stats.ontology_files_processed);
    println!("  Skipped files: {}", stats.skipped_files);
    println!("  Duration: {:?}", stats.duration);

    println!("\n📊 Ontology-Specific Statistics:");
    println!(
        "  Priority 1 (public + ontology): {}",
        stats.priority1_files
    );
    println!("  Priority 2 (ontology only): {}", stats.priority2_files);
    println!("  Priority 3 (public only): {}", stats.priority3_files);
    println!("  Total classes extracted: {}", stats.total_classes);
    println!("  Total properties extracted: {}", stats.total_properties);
    println!("  Total relationships: {}", stats.total_relationships);

    let cache_hit_rate = if stats.cache_hits + stats.cache_misses > 0 {
        (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0
    } else {
        0.0
    };

    println!("\n💾 Cache Performance:");
    println!("  Cache hits: {}", stats.cache_hits);
    println!("  Cache misses: {}", stats.cache_misses);
    println!("  Hit rate: {:.2}%", cache_hit_rate);

    // ==================
    // 2. Query by Priority
    // ==================
    println!("\n\n📑 Step 2: Querying files by priority...\n");

    // Get Priority 1 files (highest value - both public and ontology)
    let priority1_files = sync_service
        .get_ontology_files_by_priority(OntologyPriority::Priority1)
        .await;

    println!(
        "Found {} Priority 1 files (public + ontology):",
        priority1_files.len()
    );

    for (i, (path, cached_file)) in priority1_files.iter().enumerate().take(5) {
        println!("\n  {}. {}", i + 1, path);
        println!(
            "     Domain: {}",
            cached_file
                .metadata
                .source_domain
                .as_ref()
                .unwrap_or(&"Unknown".to_string())
        );
        println!("     Classes: {}", cached_file.metadata.class_count);
        println!("     Properties: {}", cached_file.metadata.property_count);
        println!(
            "     Relationships: {}",
            cached_file.metadata.relationship_count
        );
        println!("     Topics: {:?}", cached_file.metadata.topics);

        if let Some(commit_date) = cached_file.metadata.git_commit_date {
            println!("     Last commit: {}", commit_date);
        }
    }

    if priority1_files.len() > 5 {
        println!("\n  ... and {} more", priority1_files.len() - 5);
    }

    // Get Priority 2 files (ontology only)
    let priority2_files = sync_service
        .get_ontology_files_by_priority(OntologyPriority::Priority2)
        .await;

    println!(
        "\n\nFound {} Priority 2 files (ontology only)",
        priority2_files.len()
    );

    // ==================
    // 3. Enrich with Git Commit Dates
    // ==================
    println!("\n\n🕐 Step 3: Enriching with GitHub commit dates...\n");
    println!("  This will fetch git history for Priority 1 & 2 files");
    println!("  (May take a few minutes depending on file count)\n");

    let enriched_count = sync_service.enrich_with_commit_dates().await?;

    println!("✅ Enriched {} files with commit dates", enriched_count);

    // Show enriched files
    let enriched_priority1 = sync_service
        .get_ontology_files_by_priority(OntologyPriority::Priority1)
        .await;

    println!("\nSample enriched Priority 1 files:");
    for (i, (path, cached_file)) in enriched_priority1.iter().enumerate().take(3) {
        if let Some(commit_date) = cached_file.metadata.git_commit_date {
            println!("  {}. {} - Last commit: {}", i + 1, path, commit_date);
        }
    }

    // ==================
    // 4. Cache Statistics
    // ==================
    println!("\n\n💾 Step 4: Cache statistics...\n");

    let cache_stats = sync_service.get_cache_statistics().await;

    println!("Cache Status:");
    println!(
        "  Size: {}/{}",
        cache_stats.current_size, cache_stats.max_size
    );
    println!("  Hit rate: {:.2}%", cache_stats.hit_rate() * 100.0);
    println!("  Total hits: {}", cache_stats.hits);
    println!("  Total misses: {}", cache_stats.misses);
    println!("  Invalidations: {}", cache_stats.invalidations);
    println!("  Evictions: {}", cache_stats.evictions);

    // ==================
    // 5. Domain Analysis
    // ==================
    println!("\n\n🔍 Step 5: Analyzing source domains...\n");

    // Collect all cached files
    let all_cached = sync_service
        .get_ontology_files_by_priority(OntologyPriority::Priority1)
        .await
        .into_iter()
        .chain(
            sync_service
                .get_ontology_files_by_priority(OntologyPriority::Priority2)
                .await,
        )
        .collect::<Vec<_>>();

    // Count files by domain
    let mut domain_counts = std::collections::HashMap::new();
    for (_, cached_file) in &all_cached {
        if let Some(domain) = &cached_file.metadata.source_domain {
            *domain_counts.entry(domain.clone()).or_insert(0) += 1;
        }
    }

    println!("Source Domain Distribution:");
    let mut domain_vec: Vec<_> = domain_counts.iter().collect();
    domain_vec.sort_by(|a, b| b.1.cmp(a.1));

    for (domain, count) in domain_vec.iter().take(10) {
        println!("  {}: {} files", domain, count);
    }

    // ==================
    // 6. Topic Analysis
    // ==================
    println!("\n\n🏷️  Step 6: Analyzing topics...\n");

    // Collect all topics
    let mut topic_counts = std::collections::HashMap::new();
    for (_, cached_file) in &all_cached {
        for topic in &cached_file.metadata.topics {
            *topic_counts.entry(topic.clone()).or_insert(0) += 1;
        }
    }

    println!("Most Common Topics:");
    let mut topic_vec: Vec<_> = topic_counts.iter().collect();
    topic_vec.sort_by(|a, b| b.1.cmp(a.1));

    for (topic, count) in topic_vec.iter().take(15) {
        println!("  {}: {} files", topic, count);
    }

    // ==================
    // Summary
    // ==================
    println!("\n\n📋 Summary");
    println!("==========");
    println!("✅ Synced {} files", stats.total_files);
    println!(
        "✅ Identified {} high-priority ontology files",
        stats.priority1_files + stats.priority2_files
    );
    println!("✅ Extracted {} classes", stats.total_classes);
    println!("✅ Extracted {} properties", stats.total_properties);
    println!("✅ Mapped {} relationships", stats.total_relationships);
    println!("✅ Identified {} source domains", domain_counts.len());
    println!("✅ Tagged {} unique topics", topic_counts.len());
    println!(
        "✅ Cache performance: {:.2}% hit rate",
        cache_stats.hit_rate() * 100.0
    );

    if enriched_count > 0 {
        println!("✅ Enriched {} files with git history", enriched_count);
    }

    println!("\n🎉 Example complete!");

    Ok(())
}
