# Debug Logging Implementation for VisionFlow Streaming GitHub Sync

## Overview

Comprehensive debug logging has been added to the VisionFlow streaming GitHub sync service to enable detailed diagnostic analysis of file processing, database operations, and worker coordination.

## Log Level Configuration

The system uses `env_logger` which respects the `RUST_LOG` environment variable from `.env`:

```bash
RUST_LOG=debug,webxr::config=debug,webxr::models::user_settings=debug,...
```

Logs are gated using proper macros:
- `debug!()` - Detailed diagnostic information (file parsing, DB operations, worker coordination)
- `info!()` - Progress milestones and important state changes
- `warn!()` - Recoverable errors and retries
- `error!()` - Unrecoverable errors

## Files Modified

### 1. `/home/devuser/workspace/project/src/services/streaming_sync_service.rs`

**Worker Initialization & Coordination:**
- `[StreamingSync][Worker-{id}]` logs when workers start/complete
- Logs file processing progress: `{current}/{total}` files per worker
- Logs result channel operations (send success/failure)

**File Fetching:**
- `[StreamingSync][Fetch]` logs URL fetch attempts with retry count
- Logs response size in bytes and fetch duration
- Logs retry delays and final failure after max attempts

**File Type Detection:**
- `[StreamingSync][Worker-{id}]` logs detected file type (KG/Ontology/Skip)
- Logs file size and content inspection results

**Knowledge Graph Processing:**
- Logs parse start/duration with node/edge counts
- Logs semaphore wait time for database access
- Logs individual node/edge save operations with success/failure counts
- Tracks total save duration per file

**Ontology Processing:**
- Logs parse start/duration with class/property/axiom counts
- Logs semaphore wait time for database access
- Logs individual class/property/axiom save operations with success/failure counts
- Tracks total save duration per file

**Example Debug Output:**
```
[StreamingSync][Worker-0] Starting to process file 1/25: AI-Concepts.md
[StreamingSync][Worker-0] Fetching content from: https://raw.githubusercontent.com/...
[StreamingSync][Fetch] Attempt 1/3 for URL: https://...
[StreamingSync][Fetch] Fetched 12543 bytes in 234ms for AI-Concepts.md
[StreamingSync][Worker-0] Detected file type for AI-Concepts.md: KnowledgeGraph
[StreamingSync][Worker-0] Parsing KG file: AI-Concepts.md
[StreamingSync][Worker-0] Parsed AI-Concepts.md in 45ms: 23 nodes, 18 edges
[StreamingSync][Worker-0] Waiting for DB semaphore to save 23 nodes and 18 edges
[StreamingSync][Worker-0] Acquired DB semaphore after 5ms, saving to database
[StreamingSync][Worker-0] Saved 23 nodes (0 failed) and 18 edges (0 failed) in 123ms
[StreamingSync][Worker-0] Released DB semaphore
```

### 2. `/home/devuser/workspace/project/src/adapters/sqlite_ontology_repository.rs`

**Batch Save Operations (`save_ontology`):**
- `[OntologyRepo]` logs mutex acquisition with timing
- Logs transaction BEGIN/COMMIT with total duration
- Logs PRAGMA foreign_keys ON/OFF operations
- Logs DELETE operations for clearing existing data
- Logs INSERT statement preparation with item counts
- Progress logs every 100 items: `Inserted {count}/{total}`
- Logs hierarchy relationship insertions with count
- Tracks operation durations for performance analysis

**Incremental Save Operations (`add_owl_class`):**
- Logs mutex acquisition per class with timing
- Logs INSERT execution with IRI
- Logs parent relationship insertions
- Tracks total save duration per class

**Example Debug Output:**
```
[OntologyRepo] Acquiring database mutex for save_ontology (batch: 150 classes, 45 properties, 89 axioms)
[OntologyRepo] Acquired mutex in 2ms
[OntologyRepo] Disabling foreign keys for bulk insert
[OntologyRepo] Beginning transaction
[OntologyRepo] Clearing existing ontology data
[OntologyRepo] Cleared existing data in 34ms
[OntologyRepo] Preparing INSERT statement for 150 classes
[OntologyRepo] Inserted 100/150 classes
[OntologyRepo] Inserted 150 classes in 456ms
[OntologyRepo] Inserting class hierarchies
[OntologyRepo] Inserted 234 hierarchy relationships in 123ms
[OntologyRepo] Preparing INSERT statement for 45 properties
[OntologyRepo] Inserted 45 properties in 89ms
[OntologyRepo] Preparing INSERT statement for 89 axioms
[OntologyRepo] Inserted 89 axioms in 156ms
[OntologyRepo] Committing transaction
[OntologyRepo] Transaction committed successfully in 1234ms
[OntologyRepo] Re-enabling foreign keys
```

### 3. `/home/devuser/workspace/project/src/services/github_sync_service.rs`

**File List Fetching:**
- `[GitHubSync]` logs GitHub API request start
- Logs file count and fetch duration
- Logs API errors with full context

**File Type Detection:**
- `[GitHubSync][FileType]` logs total line count
- Logs detection results (KG at line X, Ontology contains marker, Skip)
- Structured format for easy parsing

**Content Fetching with Retry:**
- `[GitHubSync][Fetch]` logs attempt number and URL
- Logs success with byte count
- Logs retry delays with exponential backoff
- Logs final failure after all retries

**Node/Edge Filtering:**
- `[GitHubSync][Filter]` logs filtering start with counts
- Logs individual filter decisions (keep/remove with reason)
- Logs filtering results with counts and duration
- Tracks source/target existence for edge filtering

**Example Debug Output:**
```
[GitHubSync] Fetching markdown files from GitHub repository
[GitHubSync] Fetched 125 files in 567ms
[GitHubSync][Fetch] Starting fetch for URL: https://raw.githubusercontent.com/...
[GitHubSync][Fetch] Attempt 1/3 for https://...
[GitHubSync][Fetch] Successfully fetched 8765 bytes
[GitHubSync][FileType] Analyzing file with 234 total lines (examining first 20)
[GitHubSync][FileType] Knowledge Graph detected at line 2
[GitHubSync][Filter] Starting node filtering with 523 accumulated nodes
[GitHubSync][Filter] Keeping page node: AI-Concepts
[GitHubSync][Filter] Filtered out linked_page 'Private-Notes' - not in public pages
[GitHubSync][Filter] Filtered 45 linked_page nodes in 12ms (kept 478 of 523 total nodes)
[GitHubSync][Filter] Starting edge filtering with 892 accumulated edges
[GitHubSync][Filter] Filtered out edge 123 -> 456: source_exists=true, target_exists=false
[GitHubSync][Filter] Filtered 67 orphan edges in 8ms (kept 825 of 892 total edges)
```

## Log Format Standards

All debug logs follow these conventions:

1. **Structured Format:** `[Module][Context] Message`
   - Module: `StreamingSync`, `OntologyRepo`, `GitHubSync`
   - Context: `Worker-{id}`, `Fetch`, `Filter`, `FileType`

2. **Context Identifiers:**
   - Worker ID for parallel operations
   - File names for file-specific operations
   - Operation type for categorization

3. **Performance Metrics:**
   - Duration tracking using `Instant::now()` and `elapsed()`
   - Formatted as `{:?}` for human-readable output (e.g., "234ms")

4. **Counts and Statistics:**
   - Before/after counts for filtering operations
   - Success/failure counts for batch operations
   - Progress indicators (e.g., "100/150 classes")

5. **Error Context:**
   - Full error messages with file names
   - Operation context (what was being attempted)
   - Retry information when applicable

## Performance Analysis Usage

With debug logging enabled, you can analyze:

1. **Worker Distribution:**
   ```bash
   grep "\[Worker-" /var/log/app.log | cut -d']' -f2 | sort | uniq -c
   ```

2. **Database Contention:**
   ```bash
   grep "Waiting for DB semaphore" /var/log/app.log
   grep "Acquired DB semaphore after" /var/log/app.log | awk '{print $7}'
   ```

3. **Parse Performance:**
   ```bash
   grep "Parsed.*in" /var/log/app.log | grep -oP "in \K[0-9.]+ms"
   ```

4. **Save Performance:**
   ```bash
   grep "Saved.*in" /var/log/app.log | grep -oP "in \K[0-9.]+ms"
   ```

5. **Filtering Statistics:**
   ```bash
   grep "\[Filter\]" /var/log/app.log | grep "Filtered"
   ```

## Testing

To verify logs appear correctly:

```bash
# Enable debug logging
export RUST_LOG=debug

# Run the sync service
cargo run --bin webxr

# Monitor logs in real-time
tail -f /var/log/app.log | grep -E "\[(StreamingSync|OntologyRepo|GitHubSync)\]"
```

## Benefits

1. **Diagnostic Capability:** Track exact flow of file through system
2. **Performance Profiling:** Identify bottlenecks with timing data
3. **Error Analysis:** Full context for debugging failures
4. **Worker Coordination:** Monitor parallel processing behavior
5. **Database Operations:** Track transaction timing and query execution
6. **Filtering Validation:** Verify node/edge filtering logic

## Notes

- Debug logs are properly gated and have minimal performance impact when disabled
- No sensitive data (API tokens, credentials) is logged
- Logs are concise (1 line per event) for easy parsing
- Structured format enables automated log analysis and monitoring
