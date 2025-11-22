#!/usr/bin/env python3
"""
Ontology Loader Cache Benchmark
================================

Demonstrates cache effectiveness for repeated file access.
"""

import time
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from ontology_loader import OntologyLoader


def benchmark_cache():
    """Benchmark cache effectiveness."""

    # Find a few test files
    test_dir = Path("/home/user/logseq/mainKnowledgeGraph/pages")
    if not test_dir.exists():
        print("⚠️  Test directory not found")
        return

    all_files = sorted(list(test_dir.glob("*.md")))[:50]  # Take first 50 files

    print("=" * 80)
    print("Ontology Loader Cache Effectiveness Benchmark")
    print("=" * 80)
    print(f"Test files: {len(all_files)}")
    print()

    # Benchmark: Load same files multiple times
    print("📊 Loading each file 3 times (simulating repeated access)...")

    loader = OntologyLoader(cache_size=100)

    # First pass - cold cache
    print("\nPass 1 (Cold Cache):")
    start = time.time()
    for f in all_files:
        loader.load_file(f)
    pass1_time = time.time() - start

    stats1 = loader.get_cache_stats()
    print(f"   Time: {pass1_time:.3f}s")
    print(f"   Cache: {stats1['cache_hits']} hits, {stats1['cache_misses']} misses")
    print(f"   Hit rate: {stats1['hit_rate']:.1%}")

    # Second pass - warm cache
    print("\nPass 2 (Warm Cache):")
    start = time.time()
    for f in all_files:
        loader.load_file(f)
    pass2_time = time.time() - start

    stats2 = loader.get_cache_stats()
    print(f"   Time: {pass2_time:.3f}s")
    print(f"   Cache: {stats2['cache_hits']} hits, {stats2['cache_misses']} misses")
    print(f"   Hit rate: {stats2['hit_rate']:.1%}")
    print(f"   Speedup: {pass1_time/pass2_time:.1f}x")

    # Third pass - fully cached
    print("\nPass 3 (Fully Cached):")
    start = time.time()
    for f in all_files:
        loader.load_file(f)
    pass3_time = time.time() - start

    stats3 = loader.get_cache_stats()
    print(f"   Time: {pass3_time:.3f}s")
    print(f"   Cache: {stats3['cache_hits']} hits, {stats3['cache_misses']} misses")
    print(f"   Hit rate: {stats3['hit_rate']:.1%}")
    print(f"   Speedup: {pass1_time/pass3_time:.1f}x")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Pass 1 (cold):  {pass1_time:.3f}s - {stats1['hit_rate']:.1%} hit rate")
    print(f"Pass 2 (warm):  {pass2_time:.3f}s - {stats2['hit_rate']:.1%} hit rate - {pass1_time/pass2_time:.1f}x faster")
    print(f"Pass 3 (hot):   {pass3_time:.3f}s - {stats3['hit_rate']:.1%} hit rate - {pass1_time/pass3_time:.1f}x faster")
    print()
    print(f"Cache size: {stats3['cache_size']} files cached")
    print(f"Total hits: {stats3['cache_hits']}")
    print(f"Total misses: {stats3['cache_misses']}")
    print(f"Overall hit rate: {stats3['hit_rate']:.1%}")


if __name__ == '__main__':
    benchmark_cache()
