#!/usr/bin/env python3
"""
Ontology Loader Performance Benchmark
======================================

Measures performance metrics for the unified ontology loader.
"""

import time
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from ontology_loader import OntologyLoader
from ontology_block_parser import OntologyBlockParser


def benchmark_loader():
    """Benchmark the OntologyLoader."""

    # Find test directory (mainKnowledgeGraph/pages or sample data)
    test_dirs = [
        Path("/home/user/logseq/mainKnowledgeGraph/pages"),
        Path("/home/user/logseq/test-data"),
        Path("mainKnowledgeGraph/pages")
    ]

    test_dir = None
    for d in test_dirs:
        if d.exists():
            test_dir = d
            break

    if not test_dir:
        print("⚠️  No test directory found. Creating sample data...")
        test_dir = Path("/tmp/ontology_benchmark")
        test_dir.mkdir(exist_ok=True)

        # Create 100 sample files
        for i in range(100):
            domain = ['ai', 'bc', 'rb', 'mv'][i % 4]
            content = f"""
- ### OntologyBlock
  - **Identification**
    - term-id:: {domain.upper()}-{i:04d}
    - preferred-term:: Test Term {i}
    - public-access:: true
    - ontology:: true
    - source-domain:: {domain}
    - status:: complete
    - last-updated:: 2025-01-01

  - **Definition**
    - definition:: A test ontology block {i}.

  - **Semantic Classification**
    - owl:class:: {domain}:TestClass{i}
    - owl:physicality:: VirtualEntity
    - owl:role:: Object

  - #### Relationships
    - is-subclass-of:: [[{domain}:ParentClass]]
"""
            (test_dir / f"test_{i}.md").write_text(content)

    print("=" * 80)
    print("Ontology Loader Performance Benchmark")
    print("=" * 80)
    print(f"Test directory: {test_dir}")
    print(f"Files: {len(list(test_dir.glob('*.md')))}")
    print()

    # Benchmark 1: First load (cold cache)
    print("📊 Benchmark 1: First Load (Cold Cache)")
    loader = OntologyLoader(cache_size=200)

    start = time.time()
    blocks = loader.load_directory(test_dir, progress=False)
    cold_time = time.time() - start

    print(f"   Time: {cold_time:.3f}s")
    print(f"   Blocks loaded: {len(blocks)}")
    print(f"   Throughput: {len(blocks)/cold_time:.1f} blocks/sec")

    cache_stats = loader.get_cache_stats()
    print(f"   Cache: {cache_stats['cache_hits']} hits, {cache_stats['cache_misses']} misses")
    print()

    # Benchmark 2: Second load (warm cache)
    print("📊 Benchmark 2: Second Load (Warm Cache)")
    start = time.time()
    blocks2 = loader.load_directory(test_dir, progress=False)
    warm_time = time.time() - start

    print(f"   Time: {warm_time:.3f}s")
    print(f"   Speedup: {cold_time/warm_time:.1f}x")

    cache_stats = loader.get_cache_stats()
    print(f"   Cache: {cache_stats['cache_hits']} hits, {cache_stats['cache_misses']} misses")
    print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
    print()

    # Benchmark 3: Domain filtering
    print("📊 Benchmark 3: Domain Filtering")
    start = time.time()
    ai_blocks = loader.load_directory(test_dir, domain='ai', progress=False)
    filter_time = time.time() - start

    print(f"   Time: {filter_time:.3f}s")
    print(f"   Blocks loaded: {len(ai_blocks)}")
    print()

    # Benchmark 4: Statistics generation
    print("📊 Benchmark 4: Statistics Generation")
    start = time.time()
    stats = loader.get_statistics(blocks)
    stats_time = time.time() - start

    print(f"   Time: {stats_time:.3f}s")
    print(f"   Total blocks: {stats.total_blocks}")
    print(f"   Domains: {len(stats.by_domain)}")
    print()

    # Benchmark 5: Pattern filtering
    print("📊 Benchmark 5: Pattern Filtering")
    start = time.time()
    pattern_blocks = loader.filter_by_pattern(blocks, r'.*-00\d{2}')
    pattern_time = time.time() - start

    print(f"   Time: {pattern_time:.3f}s")
    print(f"   Matched blocks: {len(pattern_blocks)}")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Cold load time:      {cold_time:.3f}s")
    print(f"Warm load time:      {warm_time:.3f}s")
    print(f"Cache speedup:       {cold_time/warm_time:.1f}x")
    print(f"Domain filter time:  {filter_time:.3f}s")
    print(f"Statistics time:     {stats_time:.3f}s")
    print(f"Pattern filter time: {pattern_time:.3f}s")
    print(f"Cache hit rate:      {cache_stats['hit_rate']:.1%}")
    print()

    return {
        'cold_load_time': cold_time,
        'warm_load_time': warm_time,
        'speedup': cold_time / warm_time,
        'domain_filter_time': filter_time,
        'stats_time': stats_time,
        'pattern_filter_time': pattern_time,
        'cache_hit_rate': cache_stats['hit_rate'],
        'blocks_loaded': len(blocks)
    }


if __name__ == '__main__':
    benchmark_loader()
