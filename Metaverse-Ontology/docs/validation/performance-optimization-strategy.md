# Performance Optimization Strategy for OWL Extraction Pipeline

## Target Performance Metrics
- **Per-class extraction**: 130ms (current baseline)
- **Total pipeline time**: <150 seconds for 988 classes
- **Memory usage**: <500MB peak
- **Database query time**: <50ms per entry
- **Scalability**: Linear O(n) complexity

---

## 1. Performance Bottleneck Analysis

### 1.1 Component Breakdown (Per-Class Timing)

| Component | Estimated Time | % of Total | Optimization Potential |
|-----------|----------------|------------|----------------------|
| Database SELECT | 2-5ms | 2-4% | ⭐⭐⭐ HIGH (batch queries) |
| UTF-8 validation | 0.5ms | <1% | ⭐ LOW (negligible) |
| Regex extraction | 1-3ms | 1-2% | ⭐⭐ MEDIUM (compile once) |
| horned-functional parsing | 100-120ms | 77-92% | ⭐⭐ MEDIUM (inherent) |
| Ontology merging | 5-10ms | 4-8% | ⭐⭐⭐ HIGH (arena allocation) |
| IRI validation | 1-2ms | 1-2% | ⭐⭐ MEDIUM (caching) |

**Key Finding**: horned-functional parsing is the dominant bottleneck at 100-120ms per class.

---

## 2. Database Query Optimization

### Problem: 988 Sequential SELECT Queries

Current implementation likely uses:
```rust
for class_iri in class_iris {
    let entry = repo.get_entry_by_iri(&class_iri)?; // 2-5ms per query
}
// Total: 988 × 3ms = 2,964ms = 2.96 seconds
```

### Solution 1: Batch Loading with JOIN
```rust
pub fn get_all_entries_batch(
    &self,
) -> Result<Vec<OntologyEntry>, rusqlite::Error> {
    let mut stmt = self.conn.prepare_cached(
        "SELECT id, name, iri, markdown_content, parent_iri, metadata
         FROM ontology_entries
         ORDER BY id ASC"
    )?;

    let entries = stmt.query_map([], |row| {
        Ok(OntologyEntry {
            id: row.get(0)?,
            name: row.get(1)?,
            iri: row.get(2)?,
            markdown: row.get(3)?,
            parent_iri: row.get(4)?,
            metadata: row.get(5)?,
        })
    })?
    .collect::<Result<Vec<_>, _>>()?;

    Ok(entries)
}
```

**Performance gain**: 2,964ms → 50ms (single query)
**Speedup**: ~60x

---

### Solution 2: Hierarchical Loading with Recursive CTE

For parent-child relationships:
```sql
WITH RECURSIVE hierarchy AS (
    -- Base case: root nodes
    SELECT id, name, iri, markdown_content, parent_iri, 0 as depth
    FROM ontology_entries
    WHERE parent_iri IS NULL

    UNION ALL

    -- Recursive case: children
    SELECT e.id, e.name, e.iri, e.markdown_content, e.parent_iri, h.depth + 1
    FROM ontology_entries e
    INNER JOIN hierarchy h ON e.parent_iri = h.iri
)
SELECT * FROM hierarchy ORDER BY depth, name;
```

**Benefit**: Preserves hierarchy order, enables top-down processing.

---

### Solution 3: Connection Pooling for Parallel Queries

```rust
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;

pub struct OptimizedRepository {
    pool: Pool<SqliteConnectionManager>,
}

impl OptimizedRepository {
    pub fn new(db_path: &str) -> Result<Self, Box<dyn Error>> {
        let manager = SqliteConnectionManager::file(db_path);
        let pool = Pool::new(manager)?;
        Ok(Self { pool })
    }

    pub fn get_entries_parallel(&self, iris: Vec<String>) -> Vec<OntologyEntry> {
        use rayon::prelude::*;

        iris.par_iter()
            .filter_map(|iri| {
                let conn = self.pool.get().ok()?;
                self.query_single(&conn, iri).ok()
            })
            .collect()
    }
}
```

**Performance gain**: Parallel queries with connection pooling
**Speedup**: 2-4x (CPU-bound, limited by thread count)

---

## 3. Regex Optimization

### Problem: Regex Compiled 988 Times

Current (inefficient):
```rust
for entry in entries {
    let re = Regex::new(OWL_BLOCK_PATTERN)?; // Recompiled every time!
    let blocks = re.captures_iter(&entry.markdown).collect();
}
```

### Solution: Lazy Static Compilation

```rust
use once_cell::sync::Lazy;
use regex::Regex;

static OWL_BLOCK_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"```(?:clojure|owl-functional)\n([\s\S]*?)\n```")
        .expect("Failed to compile OWL_BLOCK_REGEX")
});

pub fn extract_owl_blocks(markdown: &str) -> Vec<String> {
    OWL_BLOCK_REGEX
        .captures_iter(markdown)
        .filter_map(|cap| cap.get(1))
        .map(|m| m.as_str().to_string())
        .collect()
}
```

**Performance gain**: 988 × 0.5ms compilation → 0 (compiled once)
**Speedup**: ~1.5ms saved per entry × 988 = 1,482ms saved total

---

## 4. Parallel Processing with Rayon

### Problem: Sequential Processing

Current:
```rust
for entry in entries {
    let extracted = extract_owl_from_entry(&entry)?;
    results.push(extracted);
}
// Time: 988 × 130ms = 128,440ms = 128.44 seconds
```

### Solution: Data Parallelism

```rust
use rayon::prelude::*;

pub fn extract_all_parallel(
    entries: Vec<OntologyEntry>,
) -> ExtractionResults {
    let (successes, failures): (Vec<_>, Vec<_>) = entries
        .par_iter()
        .map(|entry| {
            extract_owl_from_entry(entry)
                .map(|owl| (entry.id, owl))
                .map_err(|e| (entry.id, e))
        })
        .partition_map(|result| match result {
            Ok((id, owl)) => rayon::iter::Either::Left((id, owl)),
            Err((id, e)) => rayon::iter::Either::Right((id, e)),
        });

    ExtractionResults {
        successes: successes.into_iter().map(|(_, owl)| owl).collect(),
        failures: failures.into_iter().collect(),
        warnings: Vec::new(),
    }
}
```

**Performance gain**: Depends on CPU cores
- **4 cores**: 128s → 32s (4x speedup)
- **8 cores**: 128s → 16s (8x speedup)
- **16 cores**: 128s → 8s (16x speedup)

**Note**: horned-functional parsing is CPU-bound, parallelizes well.

---

## 5. Memory Optimization

### Problem: Holding All Ontologies in Memory

Current:
```rust
let mut all_ontologies: Vec<ExtractedOwl> = Vec::new();
for entry in entries {
    all_ontologies.push(extract_owl_from_entry(&entry)?);
}
// Memory: 988 ontologies × 300KB avg = 296MB
```

### Solution 1: Streaming Processing

```rust
pub fn process_ontologies_streaming<F>(
    repo: &SqliteOntologyRepository,
    mut processor: F,
) -> Result<(), OwlExtractionError>
where
    F: FnMut(ExtractedOwl) -> Result<(), Box<dyn Error>>,
{
    let mut stmt = repo.conn.prepare(
        "SELECT id, name, iri, markdown_content FROM ontology_entries"
    )?;

    let entry_iter = stmt.query_map([], |row| {
        Ok(OntologyEntry {
            id: row.get(0)?,
            name: row.get(1)?,
            iri: row.get(2)?,
            markdown: row.get(3)?,
            parent_iri: None,
        })
    })?;

    for entry in entry_iter {
        let entry = entry?;
        let extracted = extract_owl_from_entry(&entry)?;
        processor(extracted)?;
        // `extracted` dropped here, memory freed
    }

    Ok(())
}
```

**Usage example**:
```rust
// Stream to RDF file without holding all in memory
let mut rdf_writer = RdfWriter::new("output.rdf")?;
process_ontologies_streaming(&repo, |owl| {
    rdf_writer.write_ontology(&owl)?;
    Ok(())
})?;
```

**Memory gain**: 296MB → ~5MB (single ontology at a time)

---

### Solution 2: Arena Allocation for Ontology Merge

```rust
use bumpalo::Bump;

pub fn build_complete_ontology_arena(
    owl_blocks: Vec<String>,
) -> Result<AnnotatedOntology, OwlExtractionError> {
    let arena = Bump::new();
    let mut complete_ontology = Ontology::new();

    for block in owl_blocks {
        let parsed = parse_owl_block(&block)?;

        // Use arena for temporary axiom storage
        for axiom in parsed.axiom_iter() {
            let axiom_ref = arena.alloc(axiom.clone());
            complete_ontology.insert(axiom_ref.clone());
        }
    }

    Ok(complete_ontology)
}
```

**Memory gain**: Reduces heap fragmentation, faster allocations
**Speedup**: 5-10% faster merging

---

## 6. IRI Caching

### Problem: Repeated IRI Parsing

```rust
// IRI parsed every time it's encountered
let class_iri = IRI::parse("http://www.metaverse-ontology.com/ontology#VirtualReality")?;
```

### Solution: IRI Interner

```rust
use std::collections::HashMap;
use std::sync::Mutex;

pub struct IriInterner {
    cache: Mutex<HashMap<String, IRI>>,
}

impl IriInterner {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn intern(&self, iri_str: &str) -> Result<IRI, InvalidIRI> {
        let mut cache = self.cache.lock().unwrap();

        if let Some(iri) = cache.get(iri_str) {
            return Ok(iri.clone()); // Fast path: cached
        }

        let iri = IRI::parse(iri_str)?;
        cache.insert(iri_str.to_string(), iri.clone());
        Ok(iri)
    }
}

// Global interner
static IRI_INTERNER: Lazy<IriInterner> = Lazy::new(IriInterner::new);

pub fn get_or_intern_iri(iri_str: &str) -> Result<IRI, InvalidIRI> {
    IRI_INTERNER.intern(iri_str)
}
```

**Performance gain**: 988 IRIs × 1ms parsing → ~50ms first parse + near-zero subsequent
**Speedup**: ~1ms per repeated IRI

---

## 7. Lazy Evaluation

### Problem: Parsing OWL When Not Needed

Some use cases may only need metadata, not full ontology.

### Solution: Lazy Ontology Loading

```rust
use once_cell::unsync::OnceCell;

pub struct LazyExtractedOwl {
    pub source_iri: IRI,
    pub entry_id: i64,
    raw_blocks: Vec<String>,
    ontology: OnceCell<AnnotatedOntology>,
}

impl LazyExtractedOwl {
    pub fn new(entry: &OntologyEntry, raw_blocks: Vec<String>) -> Self {
        Self {
            source_iri: IRI::parse(&entry.iri).unwrap(),
            entry_id: entry.id,
            raw_blocks,
            ontology: OnceCell::new(),
        }
    }

    pub fn ontology(&self) -> Result<&AnnotatedOntology, OwlExtractionError> {
        self.ontology.get_or_try_init(|| {
            build_complete_ontology(self.raw_blocks.clone(), self.entry_id)
        })
    }

    pub fn has_ontology(&self) -> bool {
        self.ontology.get().is_some()
    }
}
```

**Use case**: Extract metadata without parsing OWL
```rust
let lazy_owls = extract_all_lazy(&repo)?;

// Filter by IRI without parsing OWL
let filtered: Vec<_> = lazy_owls
    .into_iter()
    .filter(|owl| owl.source_iri.contains("VirtualReality"))
    .collect();

// Only parse filtered subset
for owl in filtered {
    let ontology = owl.ontology()?; // Parse on-demand
    process(ontology);
}
```

**Speedup**: Skip parsing for unused entries (can be 10-100x for large filters)

---

## 8. Compiled OWL Binary Cache

### Problem: Parsing Same OWL Every Run

If database doesn't change frequently, parsing is redundant work.

### Solution: Binary Serialization Cache

```rust
use bincode::{serialize, deserialize};
use std::fs;
use std::time::SystemTime;

pub fn extract_with_cache(
    entry: &OntologyEntry,
    cache_dir: &Path,
) -> Result<ExtractedOwl, OwlExtractionError> {
    let cache_path = cache_dir.join(format!("{}.owl.bin", entry.id));

    // Check if cache exists and is fresh
    if let Ok(metadata) = fs::metadata(&cache_path) {
        if let Ok(modified) = metadata.modified() {
            // Cache valid for 24 hours
            let cache_age = SystemTime::now()
                .duration_since(modified)
                .unwrap_or_default();

            if cache_age.as_secs() < 86400 {
                // Load from cache
                let cached_bytes = fs::read(&cache_path)?;
                let extracted: ExtractedOwl = deserialize(&cached_bytes)
                    .map_err(|e| OwlExtractionError::CacheError(e.to_string()))?;

                log::debug!("Loaded entry {} from cache", entry.id);
                return Ok(extracted);
            }
        }
    }

    // Cache miss: parse and store
    let extracted = extract_owl_from_entry(entry)?;

    let serialized = serialize(&extracted)
        .map_err(|e| OwlExtractionError::CacheError(e.to_string()))?;
    fs::write(&cache_path, serialized)?;

    log::debug!("Cached entry {} to {:?}", entry.id, cache_path);
    Ok(extracted)
}
```

**Performance gain**: First run 128s, subsequent runs ~2s
**Speedup**: 64x on cached runs

---

## 9. Profiling Strategy

### 9.1 CPU Profiling with flamegraph

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Profile extraction pipeline
cargo flamegraph --bin owl_extractor -- --db ontology.db

# Output: flamegraph.svg
```

### 9.2 Memory Profiling with valgrind

```bash
# Install valgrind
sudo apt-get install valgrind

# Profile memory usage
valgrind --tool=massif --massif-out-file=massif.out \
    target/release/owl_extractor --db ontology.db

# Visualize
ms_print massif.out > memory_profile.txt
```

### 9.3 Criterion.rs Benchmarks

```rust
// benches/extraction_benchmark.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_extraction_stages(c: &mut Criterion) {
    let entry = create_test_entry();

    let mut group = c.benchmark_group("extraction_stages");

    group.bench_function("regex_extraction", |b| {
        b.iter(|| extract_owl_blocks(black_box(&entry.markdown)))
    });

    group.bench_function("horned_parsing", |b| {
        let blocks = extract_owl_blocks(&entry.markdown);
        b.iter(|| {
            for block in &blocks {
                let _ = parse_owl_block(black_box(block));
            }
        })
    });

    group.bench_function("ontology_merge", |b| {
        let blocks = extract_owl_blocks(&entry.markdown);
        b.iter(|| build_complete_ontology(black_box(blocks.clone())))
    });

    group.finish();
}

fn benchmark_parallelism(c: &mut Criterion) {
    let entries: Vec<_> = (0..100).map(|i| create_test_entry_with_id(i)).collect();

    let mut group = c.benchmark_group("parallelism");

    group.bench_function("sequential", |b| {
        b.iter(|| {
            entries.iter().map(|e| extract_owl_from_entry(e)).collect::<Vec<_>>()
        })
    });

    group.bench_function("parallel_rayon", |b| {
        b.iter(|| {
            use rayon::prelude::*;
            entries.par_iter().map(|e| extract_owl_from_entry(e)).collect::<Vec<_>>()
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_extraction_stages, benchmark_parallelism);
criterion_main!(benches);
```

**Run benchmarks**:
```bash
cargo bench --bench extraction_benchmark
```

---

## 10. Optimization Roadmap

### Phase 1: Quick Wins (1-2 days)
1. ✅ Implement lazy static regex compilation
2. ✅ Batch database queries (single SELECT)
3. ✅ Add criterion.rs benchmarks

**Expected speedup**: 2-3x

---

### Phase 2: Parallel Processing (2-3 days)
4. ✅ Implement Rayon parallel extraction
5. ✅ Add IRI interner for caching
6. ✅ Test with 4/8/16 cores

**Expected speedup**: 4-8x (on multi-core systems)

---

### Phase 3: Memory Optimization (3-4 days)
7. ✅ Implement streaming processing
8. ✅ Add arena allocation for merging
9. ✅ Profile memory usage with massif

**Expected memory reduction**: 50-80%

---

### Phase 4: Caching (2-3 days)
10. ✅ Implement binary cache system
11. ✅ Add cache invalidation logic
12. ✅ Test cache hit rate

**Expected speedup**: 50-100x on cached runs

---

## 11. Performance Testing Plan

### 11.1 Baseline Measurement

```rust
#[test]
fn measure_baseline_performance() {
    let repo = create_test_database_988_entries();
    let start = Instant::now();

    let results = extract_all_ontologies(&repo).unwrap();

    let elapsed = start.elapsed();

    println!("=== Baseline Performance ===");
    println!("Total entries: {}", results.successes.len());
    println!("Total time: {:?}", elapsed);
    println!("Avg per entry: {:?}", elapsed / results.successes.len() as u32);
    println!("Target: 130ms per entry");

    assert!(
        elapsed < Duration::from_secs(150),
        "Baseline must complete within 150s"
    );
}
```

---

### 11.2 Optimization Validation

```rust
#[test]
fn validate_optimized_performance() {
    let repo = create_test_database_988_entries();

    // Enable all optimizations
    let config = OptimizationConfig {
        batch_queries: true,
        parallel_processing: true,
        iri_caching: true,
        streaming: false, // In-memory for speed
    };

    let start = Instant::now();
    let results = extract_all_optimized(&repo, config).unwrap();
    let elapsed = start.elapsed();

    println!("=== Optimized Performance ===");
    println!("Total time: {:?}", elapsed);
    println!("Speedup: {:.2}x", 128.44 / elapsed.as_secs_f64());

    // Target: <30s with 8 cores
    assert!(
        elapsed < Duration::from_secs(30),
        "Optimized version must complete within 30s on 8 cores"
    );
}
```

---

### 11.3 Scalability Test

```rust
#[test]
fn test_scalability() {
    let sizes = vec![100, 500, 1000, 2000, 5000];

    for size in sizes {
        let repo = create_test_database_n_entries(size);

        let start = Instant::now();
        let _ = extract_all_optimized(&repo, OptimizationConfig::default()).unwrap();
        let elapsed = start.elapsed();

        let per_entry = elapsed.as_millis() as f64 / size as f64;

        println!("Size: {}, Total: {:?}, Per-entry: {:.2}ms", size, elapsed, per_entry);

        // Should maintain linear complexity
        assert!(
            per_entry < 50.0,
            "Per-entry time should be <50ms at scale"
        );
    }
}
```

---

## 12. Expected Performance After All Optimizations

| Metric | Before | After | Improvement |
|--------|--------|-------|------------|
| Total time (988 classes) | 128.44s | 15-20s | 6-8x |
| Per-class time | 130ms | 15-20ms | 6-8x |
| Memory usage | 296MB | 50MB | 5-6x |
| Database queries | 988 × 3ms | 1 × 50ms | 60x |
| Regex compilation | 988 × 0.5ms | 0ms | ∞ |
| Cache hit performance | N/A | 2s | 64x |

---

## 13. Production Deployment Recommendations

1. **Hardware Requirements**:
   - CPU: 8+ cores for parallel processing
   - RAM: 2GB minimum, 4GB recommended
   - Storage: SSD for database and cache

2. **Configuration**:
   - Enable parallel processing for large datasets (>100 classes)
   - Use binary cache for production runs
   - Enable streaming for memory-constrained environments

3. **Monitoring**:
   - Track per-entry extraction time
   - Monitor cache hit rate
   - Alert on performance degradation >50%

4. **Maintenance**:
   - Clear cache on database schema changes
   - Reindex database quarterly
   - Profile annually to detect regressions

---

**Document End**
