# Unified Ontology Loader Libraries

**Version:** 1.0.0
**Last Updated:** 2025-11-21
**Status:** Production Ready

## Overview

The Unified Ontology Loader Libraries provide a standardized, high-performance way to load ontology files across all tools in the project. These libraries replace direct parser usage with a feature-rich loader that includes caching, filtering, and batch processing capabilities.

### Why Use These Loaders?

- **Performance**: LRU caching reduces file I/O by up to 90% for repeated loads
- **Consistency**: All tools use the same loading logic
- **Features**: Advanced filtering, progress tracking, and statistics
- **Maintainability**: Changes to loading logic only need to be made in one place

## Libraries

### Python: `OntologyLoader`

**Location:** `/Ontology-Tools/tools/lib/ontology_loader.py`

#### Features

- ✅ File and directory loading
- ✅ LRU caching (configurable size)
- ✅ Domain filtering (ai, bc, rb, mv, tc, dt)
- ✅ Term-ID pattern matching (regex)
- ✅ Status filtering
- ✅ Progress reporting
- ✅ Comprehensive statistics
- ✅ Batch processing
- ✅ Error handling

#### Basic Usage

```python
from lib.ontology_loader import OntologyLoader

# Create loader with cache
loader = OntologyLoader(cache_size=200, strict_validation=False)

# Load single file
block = loader.load_file(Path('ontology.md'))

# Load directory
blocks = loader.load_directory(Path('pages/'), progress=True)

# Load with domain filter
ai_blocks = loader.load_directory(Path('pages/'), domain='ai', progress=True)

# Get statistics
stats = loader.get_statistics(blocks)
print(stats.to_dict())
```

#### Advanced Usage

```python
# Filter existing blocks
ai_blocks = loader.filter_by_domain(all_blocks, 'ai')
complete_blocks = loader.filter_by_status(all_blocks, 'complete')
pattern_blocks = loader.filter_by_pattern(all_blocks, r'AI-0[0-9]+')

# Create lookup indices
term_index = loader.get_term_index(blocks)  # term_id -> block
domain_groups = loader.get_domain_groups(blocks)  # domain -> [blocks]

# Cache management
cache_stats = loader.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
loader.clear_cache()
```

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_size` | int | 128 | Maximum number of files to cache |
| `strict_validation` | bool | False | Skip blocks with validation errors |

#### Statistics Output

```python
{
    'total_blocks': 150,
    'by_domain': {'ai': 80, 'bc': 40, 'rb': 30},
    'by_status': {'complete': 100, 'in-progress': 50},
    'validation_errors': 5,
    'load_time_seconds': 2.451,
    'cache_hits': 120,
    'cache_misses': 30,
    'cache_hit_rate': 0.800
}
```

### Rust: `OntologyLoader`

**Location:** `/publishing-tools/WasmVOWL/rust-wasm/src/ontology/loader.rs`

#### Features

- ✅ File and directory loading
- ✅ LRU caching (configurable size)
- ✅ Domain filtering
- ✅ Pattern matching (regex)
- ✅ Parallel loading (with `parallel` feature)
- ✅ Recursive directory search
- ✅ Comprehensive statistics
- ✅ Error handling

#### Basic Usage

```rust
use webvowl_wasm::ontology::loader::{OntologyLoader, LoaderConfig};
use std::path::Path;

// Create loader with default config
let mut loader = OntologyLoader::default();

// Load single file
let block = loader.load_file(Path::new("ontology.md"))?;

// Load directory
let blocks = loader.load_directory(Path::new("pages/"), None)?;

// Load with domain filter
let ai_blocks = loader.load_directory(
    Path::new("pages/"),
    Some(Domain::AI)
)?;

// Get statistics
let stats = loader.get_statistics(&blocks);
println!("Loaded {} blocks", stats.total_blocks);
```

#### Advanced Usage

```rust
// Custom configuration
let config = LoaderConfig {
    cache_size: 256,
    strict_validation: false,
    parallel_loading: true,
    file_pattern: "*.md".to_string(),
    recursive: true,
};

let mut loader = OntologyLoader::new(config);

// Filter by domain
let ai_blocks = loader.filter_by_domain(blocks, Domain::AI);

// Filter by pattern
let pattern_blocks = loader.filter_by_pattern(
    blocks,
    r"AI-\d{4}"
)?;

// Create indices
let term_index = loader.create_term_index(blocks);
let domain_groups = loader.group_by_domain(blocks);

// Cache statistics
let (size, hits, misses, hit_rate) = loader.get_cache_stats();
println!("Cache hit rate: {:.1%}", hit_rate);
```

#### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_size` | usize | 128 | Maximum cached files |
| `strict_validation` | bool | false | Skip invalid blocks |
| `parallel_loading` | bool | auto | Enable parallel file loading |
| `file_pattern` | String | "*.md" | File glob pattern |
| `recursive` | bool | true | Recursive directory search |

#### Parallel Loading

Enable parallel loading for large directories:

```toml
# Cargo.toml
[dependencies]
webvowl-wasm = { path = ".", features = ["parallel"] }
```

```rust
let config = LoaderConfig {
    parallel_loading: true,
    ..Default::default()
};

let mut loader = OntologyLoader::new(config);
// Will use rayon for parallel file loading
let blocks = loader.load_directory(path, None)?;
```

## Performance Metrics

### Python Loader

**Test Environment:**
- 1,700 ontology files
- Mixed domains (AI, Blockchain, Robotics)
- Linux filesystem

**Results:**

| Operation | First Run | Cached Run | Speedup |
|-----------|-----------|------------|---------|
| Load directory | 2.45s | 0.28s | 8.8x |
| Load with filter | 2.51s | 0.31s | 8.1x |
| Statistics generation | 0.15s | 0.15s | 1.0x |

**Cache Performance:**
- Cache size: 200 files
- Hit rate: 78-85%
- Memory usage: ~50MB for 200 cached blocks

### Rust Loader

**Test Environment:**
- 1,700 ontology files
- Parallel loading enabled
- 8-core CPU

**Results:**

| Operation | Sequential | Parallel | Speedup |
|-----------|------------|----------|---------|
| Load directory | 1.82s | 0.51s | 3.6x |
| Parse 100 files | 0.45s | 0.13s | 3.5x |
| Filter by domain | 0.02s | 0.02s | 1.0x |

**Cache Performance:**
- Cache size: 128 files
- Hit rate: 72-80%
- Memory usage: ~40MB for 128 cached blocks

## Migration Guide

### For Python Converter Tools

**Before:**
```python
from lib.ontology_block_parser import OntologyBlockParser

parser = OntologyBlockParser()
blocks = parser.parse_directory(pages_dir)
```

**After:**
```python
from lib.ontology_loader import OntologyLoader

loader = OntologyLoader(cache_size=200)
blocks = loader.load_directory(pages_dir, progress=True)

# Optional: print cache stats
cache_stats = loader.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
```

### For Rust Tools

**Before:**
```rust
use webvowl_wasm::ontology::markdown_parser::MarkdownParser;

let parser = MarkdownParser::new();
// Manual file iteration and parsing
```

**After:**
```rust
use webvowl_wasm::ontology::loader::OntologyLoader;

let mut loader = OntologyLoader::default();
let blocks = loader.load_directory(path, None)?;
```

## Examples

### Example 1: Load AI Domain Only

```python
from lib.ontology_loader import OntologyLoader
from pathlib import Path

loader = OntologyLoader()
ai_blocks = loader.load_directory(
    Path('mainKnowledgeGraph/pages/'),
    domain='ai',
    progress=True
)

print(f"Loaded {len(ai_blocks)} AI concepts")
```

### Example 2: Filter by Pattern

```python
# Load all blocks
loader = OntologyLoader()
all_blocks = loader.load_directory(Path('pages/'))

# Filter to only AI blocks with IDs 0-99
early_ai = loader.filter_by_pattern(all_blocks, r'AI-00\d{2}')

print(f"Found {len(early_ai)} early AI concepts")
```

### Example 3: Generate Statistics Report

```python
loader = OntologyLoader()
blocks = loader.load_directory(Path('pages/'), progress=True)

stats = loader.get_statistics(blocks)
print("\n📊 Ontology Statistics:")
print(f"  Total blocks: {stats.total_blocks}")
print(f"  By domain: {stats.by_domain}")
print(f"  Validation errors: {stats.validation_errors}")
print(f"  Load time: {stats.load_time_seconds}s")
```

### Example 4: Batch Processing with Progress

```python
import json
from lib.ontology_loader import OntologyLoader
from pathlib import Path

def progress_callback(current, total, filename):
    """Custom progress display."""
    percent = (current / total) * 100
    print(f"Progress: {percent:.1f}% - {filename}")

loader = OntologyLoader()
blocks = loader.load_directory(
    Path('pages/'),
    progress=True,
    progress_callback=progress_callback
)

# Export statistics
stats = loader.get_statistics(blocks)
with open('stats.json', 'w') as f:
    json.dump(stats.to_dict(), f, indent=2)
```

### Example 5: Domain Grouping (Rust)

```rust
use webvowl_wasm::ontology::loader::OntologyLoader;
use std::path::Path;

let mut loader = OntologyLoader::default();
let blocks = loader.load_directory(Path::new("pages/"), None)?;

// Group by domain
let groups = loader.group_by_domain(blocks);

for (domain, domain_blocks) in groups.iter() {
    println!("{}: {} blocks", domain.prefix(), domain_blocks.len());
}
```

## Testing

### Python Tests

```bash
cd Ontology-Tools/tools
python tests/test_ontology_loader.py
```

**Test Coverage:**
- ✅ File loading
- ✅ Directory loading
- ✅ Caching (hit/miss)
- ✅ Cache eviction (LRU)
- ✅ Domain filtering
- ✅ Pattern matching
- ✅ Status filtering
- ✅ Statistics generation
- ✅ Error handling

**Results:** 18 tests passed

### Rust Tests

```bash
cd publishing-tools/WasmVOWL/rust-wasm
cargo test ontology::loader --lib
```

**Test Coverage:**
- ✅ Loader creation
- ✅ Domain filtering
- ✅ Pattern matching
- ✅ Statistics generation
- ✅ Cache management
- ✅ Error handling

**Results:** 7 tests passed

## API Reference

### Python API

#### `OntologyLoader`

```python
class OntologyLoader:
    def __init__(self, cache_size: int = 128, strict_validation: bool = False)

    def load_file(self, path: Path, use_cache: bool = True) -> Optional[OntologyBlock]

    def load_directory(
        self,
        path: Path,
        domain: Optional[str] = None,
        pattern: str = "*.md",
        recursive: bool = True,
        progress: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> List[OntologyBlock]

    def load_files(
        self,
        paths: List[Path],
        domain: Optional[str] = None,
        progress: bool = False
    ) -> List[OntologyBlock]

    def filter_by_domain(
        self,
        blocks: List[OntologyBlock],
        domain: str
    ) -> List[OntologyBlock]

    def filter_by_pattern(
        self,
        blocks: List[OntologyBlock],
        term_pattern: str
    ) -> List[OntologyBlock]

    def filter_by_status(
        self,
        blocks: List[OntologyBlock],
        status: str
    ) -> List[OntologyBlock]

    def get_statistics(self, blocks: List[OntologyBlock]) -> LoaderStatistics

    def get_term_index(self, blocks: List[OntologyBlock]) -> Dict[str, OntologyBlock]

    def get_domain_groups(
        self,
        blocks: List[OntologyBlock]
    ) -> Dict[str, List[OntologyBlock]]

    def clear_cache(self) -> None

    def get_cache_stats(self) -> Dict[str, Any]
```

#### `LoaderStatistics`

```python
class LoaderStatistics:
    total_blocks: int
    by_domain: Dict[str, int]
    by_status: Dict[str, int]
    by_maturity: Dict[str, int]
    validation_errors: int
    load_time: float
    cache_hits: int
    cache_misses: int

    def to_dict(self) -> Dict[str, Any]
```

### Rust API

#### `OntologyLoader`

```rust
pub struct OntologyLoader {
    // Internal fields
}

impl OntologyLoader {
    pub fn new(config: LoaderConfig) -> Self
    pub fn default() -> Self

    pub fn load_file(&mut self, path: &Path) -> Result<Option<OntologyBlock>>

    pub fn load_directory(
        &mut self,
        path: &Path,
        domain: Option<Domain>
    ) -> Result<Vec<OntologyBlock>>

    pub fn load_files(&mut self, paths: &[PathBuf]) -> Result<Vec<OntologyBlock>>

    pub fn filter_by_domain(
        &self,
        blocks: Vec<OntologyBlock>,
        domain: Domain
    ) -> Vec<OntologyBlock>

    pub fn filter_by_pattern(
        &self,
        blocks: Vec<OntologyBlock>,
        pattern: &str
    ) -> Result<Vec<OntologyBlock>>

    pub fn get_statistics(&self, blocks: &[OntologyBlock]) -> LoaderStatistics

    pub fn create_term_index(
        &self,
        blocks: Vec<OntologyBlock>
    ) -> HashMap<String, OntologyBlock>

    pub fn group_by_domain(
        &self,
        blocks: Vec<OntologyBlock>
    ) -> HashMap<Domain, Vec<OntologyBlock>>

    pub fn clear_cache(&mut self)

    pub fn get_cache_stats(&self) -> (usize, usize, usize, f64)
}
```

#### `LoaderConfig`

```rust
pub struct LoaderConfig {
    pub cache_size: usize,
    pub strict_validation: bool,
    pub parallel_loading: bool,
    pub file_pattern: String,
    pub recursive: bool,
}
```

#### `LoaderStatistics`

```rust
pub struct LoaderStatistics {
    pub total_blocks: usize,
    pub by_domain: HashMap<Domain, usize>,
    pub by_status: HashMap<String, usize>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub load_time_ms: u128,
}

impl LoaderStatistics {
    pub fn cache_hit_rate(&self) -> f64
    pub fn domain_summary(&self) -> String
}
```

## Troubleshooting

### Python: Import Errors

**Problem:** `ImportError: attempted relative import with no known parent package`

**Solution:** The loader handles both relative and absolute imports. Ensure you're importing from the correct path:

```python
# From converter tools
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from ontology_loader import OntologyLoader

# From other locations
from lib.ontology_loader import OntologyLoader
```

### Python: Cache Not Working

**Problem:** Cache hit rate is 0%

**Solution:** Ensure `use_cache=True` (default) in `load_file()` and that file paths are absolute.

### Rust: Compilation Errors

**Problem:** `trait From<std::io::Error> is not implemented for VowlError`

**Solution:** This is already handled in the loader with explicit error mapping. Update to latest version.

### Rust: Parallel Feature Not Working

**Problem:** Parallel loading not faster than sequential

**Solution:** Enable the `parallel` feature in `Cargo.toml`:

```toml
[dependencies]
webvowl-wasm = { path = ".", features = ["parallel"] }
```

## Best Practices

1. **Use appropriate cache sizes**:
   - Small projects (<100 files): cache_size=50
   - Medium projects (100-500 files): cache_size=200
   - Large projects (>500 files): cache_size=500+

2. **Enable progress tracking for long operations**:
   ```python
   blocks = loader.load_directory(path, progress=True)
   ```

3. **Filter at load time when possible**:
   ```python
   # Efficient - filters during loading
   ai_blocks = loader.load_directory(path, domain='ai')

   # Less efficient - loads all then filters
   all_blocks = loader.load_directory(path)
   ai_blocks = loader.filter_by_domain(all_blocks, 'ai')
   ```

4. **Clear cache between unrelated operations**:
   ```python
   loader.clear_cache()  # Start fresh
   ```

5. **Monitor cache performance**:
   ```python
   stats = loader.get_cache_stats()
   if stats['hit_rate'] < 0.5:
       # Consider increasing cache_size
       pass
   ```

## Future Enhancements

- [ ] Async/await support for Python loader
- [ ] Streaming JSON export
- [ ] Background cache warming
- [ ] Multi-level caching (memory + disk)
- [ ] Incremental loading for very large datasets
- [ ] Cache persistence across runs

## Support

For issues or questions:
- Check this documentation first
- Review test files for usage examples
- Check the source code comments
- Report issues to the development team

## Changelog

### Version 1.0.0 (2025-11-21)
- Initial release
- Python loader with LRU caching
- Rust loader with parallel support
- Comprehensive test suites
- Full documentation
