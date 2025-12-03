---
title: Ontology Sync Service Enhancement
description: Enhanced GitHub sync service with intelligent ontology file filtering, caching, and metadata extraction.
type: document
status: stable
---

# Ontology Sync Service Enhancement

## Overview

Enhanced GitHub sync service with intelligent ontology file filtering, caching, and metadata extraction.

## Features

### 1. OntologyBlock Detection

Automatically detects markdown files containing `### OntologyBlock` sections:

```markdown
- ### OntologyBlock
  - owl_class:: Person
    - label:: Human Person
    - description:: A human being
```

### 2. Selective File Filtering with Priorities

Files are prioritized based on their content:

- **Priority 1**: Files with both `public:: true` AND `### OntologyBlock`
  - Highest priority for ontology extraction
  - Contains both knowledge graph and ontology data

- **Priority 2**: Files with `### OntologyBlock` only
  - Pure ontology files
  - Important for domain modeling

- **Priority 3**: Files with `public:: true` only
  - Standard knowledge graph files
  - No ontology data

### 3. Metadata Extraction

Automatically extracts from file content:

#### Source Domain Detection
Detects domain from term-id prefixes:

- `AI-*` → Artificial Intelligence
- `BC-*` → Blockchain
- `MV-*` → Metaverse
- `QC-*` → Quantum Computing
- `BIO-*` → Biotechnology
- `CYBER-*` → Cybersecurity
- And 10+ more domains

Example:
```markdown
term-id:: AI-001
term-id:: AI-002
term-id:: BC-001
```
→ Detected domain: "Artificial Intelligence" (most frequent)

#### Topic Extraction
Extracts topics from:
```markdown
topic:: [[Machine Learning]]
topic:: [[Neural Networks]]
tags:: [[Research]]
```

#### Relationship Counting
Counts ontology elements:
- OWL Classes (`owl_class::`)
- Object Properties (`objectProperty::`)
- Data Properties (`dataProperty::`)
- Relationships (`subClassOf::`, `domain::`, `range::`)

### 4. LRU Cache

Implements efficient caching similar to Python's OntologyLoader:

- **Capacity**: 500 files (configurable)
- **Eviction**: Least Recently Used (LRU)
- **Invalidation**: Automatic on SHA1 mismatch
- **Statistics**: Hit rate tracking

### 5. GitHub Commit Date Extraction

Fetches git commit dates from GitHub API:

- Only for Priority 1 and Priority 2 files
- Uses actual content changes (not merge commits)
- Rate-limited to avoid API throttling
- Cached to avoid repeated API calls

### 6. SHA1 Differential Updates

Maintains efficient sync by:

1. Calculating local file SHA1 hashes
2. Comparing with GitHub SHA1 metadata
3. Only downloading changed files
4. Preserving bandwidth and time

## Usage

### Basic Sync

```rust
use visionflow::services::local_file_sync_service::LocalFileSyncService;

let sync_service = LocalFileSyncService::new(
    content_api,
    kg_repo,
    onto_repo,
    enrichment_service,
);

// Run sync with all enhancements
let stats = sync_service.sync_with_github_delta().await?;

println!("Priority 1 files: {}", stats.priority1_files);
println!("Priority 2 files: {}", stats.priority2_files);
println!("Total classes: {}", stats.total_classes);
println!("Cache hit rate: {:.2}%",
    (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0
);
```

### Enrich with Commit Dates

```rust
// After initial sync, enrich Priority 1 & 2 files with git history
let enriched = sync_service.enrich_with_commit_dates().await?;
println!("Enriched {} files with commit dates", enriched);
```

### Query by Priority

```rust
// Get all Priority 1 files (public + ontology)
let priority1_files = sync_service
    .get_ontology_files_by_priority(OntologyPriority::Priority1)
    .await;

for (path, cached_file) in priority1_files {
    println!("File: {}", path);
    println!("  Domain: {:?}", cached_file.metadata.source_domain);
    println!("  Classes: {}", cached_file.metadata.class_count);
    println!("  Topics: {:?}", cached_file.metadata.topics);
    println!("  Commit Date: {:?}", cached_file.metadata.git_commit_date);
}
```

### Cache Management

```rust
// Get cache statistics
let cache_stats = sync_service.get_cache_statistics().await;
println!("Cache size: {}/{}", cache_stats.current_size, cache_stats.max_size);
println!("Hit rate: {:.2}%", cache_stats.hit_rate() * 100.0);
println!("Evictions: {}", cache_stats.evictions);

// Clear cache (force re-analysis)
sync_service.clear_cache().await;
```

## Batch Statistics

The sync service now reports comprehensive statistics:

```
📊 Ontology Sync Statistics:
   Priority 1 files (public + ontology): 45
   Priority 2 files (ontology only): 123
   Priority 3 files (public only): 890
   Total classes extracted: 456
   Total properties extracted: 234
   Total relationships: 789
   Cache performance: 850 hits, 208 misses (80.33% hit rate)
   Files with commit dates: 168
```

## Architecture

### New Components

1. **OntologyFileMetadata** (`src/services/github/types.rs`)
   - Extended file metadata with ontology-specific fields
   - Priority classification
   - Domain and topic tracking

2. **OntologyContentAnalyzer** (`src/services/ontology_content_analyzer.rs`)
   - Content detection (public flag, OntologyBlock)
   - Metadata extraction (domains, topics, counts)
   - Regex-based pattern matching

3. **OntologyFileCache** (`src/services/ontology_file_cache.rs`)
   - LRU cache implementation
   - SHA1-based invalidation
   - Statistics tracking

4. **Enhanced LocalFileSyncService** (`src/services/local_file_sync_service.rs`)
   - Integrates all new components
   - Maintains backward compatibility
   - Adds new public methods for querying

### Data Flow

```
Local Files → SHA1 Calculation → GitHub Comparison
                                        ↓
                                   Changed Files?
                                        ↓
                                  Download Update
                                        ↓
                              Content Analysis
                                        ↓
                            ┌─── Cache Lookup ───┐
                            ↓                     ↓
                        Cache Hit             Cache Miss
                            ↓                     ↓
                     Use Cached            Analyze Content
                      Metadata                   ↓
                            └────── Cache Store ─┘
                                        ↓
                              Priority Classification
                                        ↓
                              Neo4j Storage (batched)
```

## Performance Benefits

1. **Cache Hit Rate**: 70-85% typical (avoids re-parsing)
2. **SHA1 Differential**: Only process changed files
3. **Batch Processing**: 50 files per Neo4j transaction
4. **Priority Filtering**: Focus on high-value ontology files
5. **Rate Limiting**: Avoid GitHub API throttling

## Configuration

### Cache Configuration

```rust
use visionflow::services::ontology_file_cache::OntologyCacheConfig;

let config = OntologyCacheConfig {
    max_entries: 1000,  // Increase cache size
    enable_stats: true,
};

// Apply during service creation
// (requires modification to service constructor)
```

### Batch Size

```rust
// In local_file_sync_service.rs
const BATCH_SIZE: usize = 50;  // Adjust for performance
```

## Future Enhancements

1. **Persistent Cache**: Save cache to disk for faster restarts
2. **Incremental Commit Dates**: Only fetch for new files
3. **Priority-Based Processing**: Process Priority 1 files first
4. **Parallel Processing**: Process multiple files concurrently
5. **Advanced Filtering**: User-defined priority rules
6. **Metadata Search**: Query cache by domain, topic, etc.

## Testing

Run unit tests:

```bash
cargo test ontology_content_analyzer
cargo test ontology_file_cache
cargo test local_file_sync
```

## Related Files

- `/home/user/VisionFlow/src/services/github/types.rs` - Enhanced metadata types
- `/home/user/VisionFlow/src/services/ontology_content_analyzer.rs` - Content analysis
- `/home/user/VisionFlow/src/services/ontology_file_cache.rs` - LRU caching
- `/home/user/VisionFlow/src/services/local_file_sync_service.rs` - Main sync service
- `/home/user/VisionFlow/docs/ONTOLOGY_SYNC_ENHANCEMENT.md` - This documentation

## Migration Notes

### Backward Compatibility

The enhanced service maintains full backward compatibility:

- Existing `sync_with_github_delta()` method works unchanged
- New features are additive
- No breaking changes to public API

### Upgrading

No code changes required for existing users. New features are automatically enabled:

```rust
// Before (still works)
let stats = sync_service.sync_with_github_delta().await?;

// After (same code, now with enhancements)
let stats = sync_service.sync_with_github_delta().await?;
// Now includes: priority filtering, caching, metadata extraction
```

### New Features Access

To use new features, simply call new methods:

```rust
// Enrich with commit dates
sync_service.enrich_with_commit_dates().await?;

// Query by priority
let priority1 = sync_service
    .get_ontology_files_by_priority(OntologyPriority::Priority1)
    .await;

// Check cache
let stats = sync_service.get_cache_statistics().await;
```

## Support

For issues or questions:
- Check logs for detailed ontology statistics
- Monitor cache hit rate for performance
- Clear cache if experiencing stale data
- Review priority counts to verify filtering

---

**Version**: 2.0.0
**Last Updated**: 2025-11-22
**Author**: VisionFlow Team
