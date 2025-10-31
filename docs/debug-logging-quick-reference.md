# Debug Logging Quick Reference

## Enable Debug Logging

Already configured in `.env`:
```bash
RUST_LOG=debug,webxr::config=debug,webxr::models::user_settings=debug,...
```

## Log Prefixes

| Prefix | Module | Purpose |
|--------|--------|---------|
| `[StreamingSync][Worker-{id}]` | Streaming sync worker | File processing by individual workers |
| `[StreamingSync][Fetch]` | Content fetching | HTTP requests and retries |
| `[OntologyRepo]` | Ontology database | SQLite operations, transactions, inserts |
| `[GitHubSync]` | GitHub sync service | File listing, type detection, filtering |
| `[GitHubSync][Fetch]` | Content fetching | GitHub API file downloads |
| `[GitHubSync][Filter]` | Node/edge filtering | Post-processing filter operations |
| `[GitHubSync][FileType]` | File type detection | Marker detection (public::, OntologyBlock) |

## Key Debug Patterns

### Worker Activity
```bash
# Watch worker processing
tail -f /var/log/app.log | grep '\[Worker-'

# Count files processed per worker
grep '\[Worker-' /var/log/app.log | cut -d']' -f2 | cut -d' ' -f1 | sort | uniq -c
```

### Database Performance
```bash
# Monitor semaphore wait times
grep 'Acquired DB semaphore after' /var/log/app.log

# Transaction timings
grep 'Transaction committed successfully in' /var/log/app.log

# Insert progress
grep 'Inserted.*/' /var/log/app.log
```

### File Processing Pipeline
```bash
# Track a specific file through the pipeline
grep 'AI-Concepts.md' /var/log/app.log

# File type detection results
grep '\[FileType\]' /var/log/app.log | grep 'detected'

# Parse performance
grep 'Parsed.*in' /var/log/app.log
```

### Filtering Analysis
```bash
# Node filtering decisions
grep '\[Filter\] Keeping\|\[Filter\] Filtered out' /var/log/app.log

# Filtering summary
grep '\[Filter\] Filtered.*nodes\|Filtered.*edges' /var/log/app.log
```

### Error Tracking
```bash
# All errors and warnings
grep -E 'ERROR|WARN' /var/log/app.log

# Retry attempts
grep 'Retry' /var/log/app.log

# Failed operations
grep 'Failed to' /var/log/app.log
```

## Performance Analysis

### Worker Load Balance
```bash
# Files per worker
grep 'Worker.*starting with' /var/log/app.log

# Worker completion times
grep 'Worker.*completed' /var/log/app.log
```

### Database Bottlenecks
```bash
# Average semaphore wait time
grep 'Acquired DB semaphore after' /var/log/app.log | \
  awk '{print $8}' | sed 's/ms,//' | \
  awk '{sum+=$1; count++} END {print sum/count "ms average"}'

# Transaction durations
grep 'Transaction committed successfully in' /var/log/app.log | \
  grep -oP 'in \K[0-9.]+ms'
```

### Fetch Performance
```bash
# Average fetch size and time
grep 'Fetched.*bytes in' /var/log/app.log

# Retry rate
grep 'Attempt 2\|Attempt 3' /var/log/app.log | wc -l
```

## Troubleshooting Scenarios

### Slow Sync
1. Check worker distribution: `grep 'Worker.*starting with' /var/log/app.log`
2. Check semaphore waits: `grep 'Waiting for DB semaphore' /var/log/app.log`
3. Check parse times: `grep 'Parsed.*in' /var/log/app.log | sort -t: -k2 -n`

### Missing Data
1. Check file type detection: `grep '\[FileType\]' /var/log/app.log`
2. Check filtering: `grep '\[Filter\] Filtered out' /var/log/app.log`
3. Check parse errors: `grep 'Parse error' /var/log/app.log`

### Database Errors
1. Check foreign key operations: `grep 'foreign keys' /var/log/app.log`
2. Check transaction state: `grep 'BEGIN\|COMMIT\|ROLLBACK' /var/log/app.log`
3. Check INSERT errors: `grep 'Failed to insert' /var/log/app.log`

### Worker Coordination Issues
1. Check worker spawn: `grep 'Spawning.*workers' /var/log/app.log`
2. Check worker completion: `grep 'Worker.*completed' /var/log/app.log`
3. Check result channel: `grep 'Failed to send result' /var/log/app.log`

## Example Debugging Session

```bash
# 1. Enable debug logging (already in .env)
export RUST_LOG=debug

# 2. Start the service and capture logs
cargo run --bin webxr 2>&1 | tee sync.log

# 3. In another terminal, monitor specific aspects
tail -f sync.log | grep '\[Worker-0\]'  # Watch worker 0
tail -f sync.log | grep 'Acquired DB semaphore after'  # Watch DB contention

# 4. After completion, analyze
grep 'ERROR\|WARN' sync.log > errors.log
grep '\[Filter\]' sync.log > filtering.log
grep 'Parsed.*in' sync.log | sort -t: -k2 -n > parse_times.log

# 5. Generate statistics
echo "Total files processed:"
grep 'Successfully sent result' sync.log | wc -l

echo "Average parse time:"
grep 'Parsed.*in' sync.log | grep -oP 'in \K[0-9.]+ms' | \
  awk '{sum+=$1; count++} END {print sum/count "ms"}'

echo "Average DB wait time:"
grep 'Acquired DB semaphore after' sync.log | grep -oP 'after \K[0-9.]+ms' | \
  awk '{sum+=$1; count++} END {print sum/count "ms"}'
```

## Log Volume Considerations

Debug logging is verbose. For production:

1. **Disable debug logs:**
   ```bash
   export RUST_LOG=info
   ```

2. **Enable only for specific modules:**
   ```bash
   export RUST_LOG=info,webxr::services::streaming_sync_service=debug
   ```

3. **Use log rotation:**
   ```bash
   # In supervisord.conf
   stdout_logfile_maxbytes=50MB
   stdout_logfile_backups=10
   ```

## Common Log Patterns

| Pattern | Meaning |
|---------|---------|
| `Fetched X bytes in Yms` | Successful file download |
| `Parsed X in Yms: N nodes, M edges` | Successful KG parse |
| `Parsed X in Yms: N classes, M properties, P axioms` | Successful ontology parse |
| `Acquired DB semaphore after Xms` | Database write started (X indicates contention) |
| `Saved N nodes (M failed)` | Completed node insertion |
| `Filtered N linked_page nodes` | Post-processing filter result |
| `Transaction committed successfully in Xms` | Batch insert completed |
| `Retry N/M for URL` | Retry attempt due to transient error |
