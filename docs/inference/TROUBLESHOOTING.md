# Inference Troubleshooting Guide

## Common Issues

### 1. "Ontology Not Loaded" Error

**Error:**
```
InferenceEngineError::OntologyNotLoaded
```

**Cause:** Attempting inference before loading ontology.

**Solution:**
```rust
// Always load before inferring
engine.load_ontology(classes, axioms).await?;
engine.infer().await?; // Now works
```

### 2. "Inconsistent Ontology" Error

**Error:**
```
InferenceEngineError::InconsistentOntology("Class X is unsatisfiable")
```

**Cause:** Ontology contains logical contradictions.

**Solution:**
```rust
// Check which classes are unsatisfiable
let validation = service.validate_ontology(ontology_id).await?;

for unsat in validation.unsatisfiable {
    println!("Unsatisfiable: {} - {}", unsat.class_iri, unsat.reason);
}
```

**Common Contradictions:**
- Class is subclass of both A and B where A and B are disjoint
- Cardinality constraints conflict
- Property restrictions conflict

### 3. Inference Returns Empty Results

**Symptoms:**
- `inferred_axioms` is empty
- No subsumptions found
- Classification produces no hierarchy

**Possible Causes:**
1. **No axioms to infer from:**
   ```rust
   // Check loaded axioms
   let stats = engine.get_statistics().await?;
   println!("Loaded axioms: {}", stats.loaded_axioms);
   ```

2. **Already classified:**
   ```rust
   // Results may be cached
   service.invalidate_cache(ontology_id).await;
   service.run_inference(ontology_id).await?; // Fresh
   ```

3. **EL limitations:**
   ```
   whelk-rs supports EL profile only.
   Complex axioms may not produce inferences.
   ```

### 4. Slow Performance

**Symptoms:**
- Inference takes >5 seconds
- High CPU usage
- Memory growth

**Solutions:**

1. **Enable caching:**
   ```rust
   let config = InferenceServiceConfig {
       enable_cache: true,
       ..Default::default()
   };
   ```

2. **Check ontology size:**
   ```rust
   let stats = engine.get_statistics().await?;
   if stats.loaded_classes > 10000 {
       // Use batch processing
       service.batch_inference(ontology_ids).await?;
   }
   ```

3. **Profile performance:**
   ```rust
   let start = std::time::Instant::now();
   engine.infer().await?;
   println!("Inference took: {:?}", start.elapsed());
   ```

### 5. Cache Not Working

**Symptoms:**
- Same ontology infers multiple times
- No speedup on repeated calls
- Cache statistics show 0% hit rate

**Diagnostics:**
```rust
let stats = cache.get_statistics().await;
println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
```

**Solutions:**

1. **Verify cache is enabled:**
   ```rust
   let config = InferenceServiceConfig {
       enable_cache: true, // Must be true
       ..Default::default()
   };
   ```

2. **Check checksum:**
   ```rust
   // Ontology changes invalidate cache
   // Same ontology should have same checksum
   ```

3. **Verify TTL:**
   ```rust
   let cache_config = CacheConfig {
       ttl_seconds: 3600, // Not expired
       ..Default::default()
   };
   ```

### 6. Parse Errors

**Error:**
```
ParseError::ParseError("OWL/XML parse error: ...")
```

**Solutions:**

1. **Verify format:**
   ```rust
   let format = OWLParser::detect_format(content);
   println!("Detected: {:?}", format);
   ```

2. **Try explicit format:**
   ```rust
   OWLParser::parse_with_format(content, OWLFormat::RdfXml)?;
   ```

3. **Check syntax:**
   - Valid XML
   - Correct namespaces
   - Well-formed structure

### 7. Memory Leaks

**Symptoms:**
- Memory grows over time
- Cache size increases unbounded
- Out of memory errors

**Solutions:**

1. **Limit cache size:**
   ```rust
   let config = CacheConfig {
       max_entries: 100, // Hard limit
       ..Default::default()
   };
   ```

2. **Periodic cleanup:**
   ```rust
   // In background task
   loop {
       tokio::time::sleep(Duration::from_secs(3600)).await;
       cache.cleanup_expired().await;
   }
   ```

3. **Clear engine:**
   ```rust
   // After inference
   engine.clear().await?;
   ```

### 8. Event Triggers Not Firing

**Symptoms:**
- Auto-inference not running
- Events not published

**Solutions:**

1. **Verify config:**
   ```rust
   let config = AutoInferenceConfig {
       auto_on_import: true, // Enable
       ..Default::default()
   };
   ```

2. **Check event bus:**
   ```rust
   // Ensure event bus is registered
   register_inference_triggers(event_bus, service, config).await;
   ```

3. **Verify rate limiting:**
   ```rust
   let config = AutoInferenceConfig {
       min_delay_ms: 1000, // Not too restrictive
       ..Default::default()
   };
   ```

### 9. API Endpoints Return 500

**Error:**
```
HTTP 500 Internal Server Error
```

**Diagnostics:**
```rust
// Check logs
RUST_LOG=debug cargo run
```

**Common Causes:**
1. Service not initialized
2. Repository connection failed
3. Inference engine error

**Solutions:**
```rust
// Proper error handling in handlers
match service.run_inference(ontology_id).await {
    Ok(results) => HttpResponse::Ok().json(results),
    Err(e) => {
        warn!("Inference failed: {:?}", e);
        HttpResponse::InternalServerError().json(error_response)
    }
}
```

### 10. Compilation Errors

**Error:**
```
error: cannot find module `inference`
```

**Solution:**
```rust
// In src/lib.rs
pub mod inference;

// In src/application/mod.rs
pub mod inference_service;

// In src/handlers/mod.rs
pub mod inference_handler;
```

**Error:**
```
feature `ontology` is required
```

**Solution:**
```toml
# In Cargo.toml
[features]
default = ["ontology"]
ontology = ["whelk", "horned-owl"]
```

## Debug Checklist

- [ ] Check logs (`RUST_LOG=debug`)
- [ ] Verify ontology loaded
- [ ] Check inference statistics
- [ ] Verify cache enabled
- [ ] Check cache hit rate
- [ ] Profile performance
- [ ] Test with small ontology
- [ ] Verify API endpoint registration
- [ ] Check event bus registration
- [ ] Review error messages

## Getting Help

1. **Check logs:** Enable debug logging
2. **Review docs:** Integration guide and API reference
3. **Test minimal example:** Isolate the issue
4. **Check GitHub issues:** Known problems
5. **Profile performance:** Identify bottlenecks

## Useful Commands

```bash
# Enable debug logging
RUST_LOG=debug cargo run

# Run specific test
cargo test --test inference --features ontology

# Check compilation
cargo check --all-features

# Run benchmarks
cargo test --release performance_tests

# Profile memory
valgrind --tool=massif ./target/release/server
```
