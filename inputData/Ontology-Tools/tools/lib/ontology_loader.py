#!/usr/bin/env python3
"""
Unified Ontology Loader Library
================================

High-performance loader for ontology files with caching, filtering, and batch processing.
All converter tools should use this library for consistent loading behavior.

Features:
- File and directory loading
- Domain filtering
- Term-ID pattern matching
- LRU caching for performance
- Progress reporting for large batches
- Comprehensive error handling
- Statistics generation

Usage:
    from ontology_loader import OntologyLoader

    loader = OntologyLoader(cache_size=100)

    # Load single file
    block = loader.load_file(Path('file.md'))

    # Load directory with filtering
    blocks = loader.load_directory(
        Path('mainKnowledgeGraph/pages/'),
        domain='ai',
        progress=True
    )

    # Get statistics
    stats = loader.get_statistics(blocks)
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable
from functools import lru_cache
from collections import defaultdict
import time

try:
    from .ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG
except ImportError:
    from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG


class LoaderStatistics:
    """Statistics for loaded ontology blocks."""

    def __init__(self):
        self.total_blocks = 0
        self.by_domain = defaultdict(int)
        self.by_status = defaultdict(int)
        self.by_maturity = defaultdict(int)
        self.validation_errors = 0
        self.load_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_blocks': self.total_blocks,
            'by_domain': dict(self.by_domain),
            'by_status': dict(self.by_status),
            'by_maturity': dict(self.by_maturity),
            'validation_errors': self.validation_errors,
            'load_time_seconds': round(self.load_time, 3),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': round(self.cache_hits / max(1, self.cache_hits + self.cache_misses), 3)
        }


class OntologyLoader:
    """
    Unified loader for ontology files with advanced features.

    This is the recommended way to load ontology files across all tools.
    """

    def __init__(self, cache_size: int = 128, strict_validation: bool = False):
        """
        Initialize the loader.

        Args:
            cache_size: Maximum number of files to cache (LRU)
            strict_validation: If True, skip blocks with validation errors
        """
        self.parser = OntologyBlockParser()
        self.cache_size = cache_size
        self.strict_validation = strict_validation
        self._file_cache: Dict[str, Optional[OntologyBlock]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def load_file(self, path: Path, use_cache: bool = True) -> Optional[OntologyBlock]:
        """
        Load a single ontology file.

        Args:
            path: Path to markdown file
            use_cache: Whether to use file cache

        Returns:
            OntologyBlock if valid, None otherwise
        """
        path_str = str(path.resolve())

        # Check cache
        if use_cache and path_str in self._file_cache:
            self._cache_hits += 1
            return self._file_cache[path_str]

        self._cache_misses += 1

        # Parse file
        try:
            block = self.parser.parse_file(path)

            # Validate if strict mode
            if block and self.strict_validation:
                errors = block.validate()
                if errors:
                    block = None

            # Cache result
            if use_cache:
                self._file_cache[path_str] = block
                # Implement LRU eviction
                if len(self._file_cache) > self.cache_size:
                    # Remove oldest entry (simple FIFO, not true LRU)
                    oldest = next(iter(self._file_cache))
                    del self._file_cache[oldest]

            return block

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def load_directory(
        self,
        path: Path,
        domain: Optional[str] = None,
        pattern: str = "*.md",
        recursive: bool = True,
        progress: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[OntologyBlock]:
        """
        Load all ontology files from a directory.

        Args:
            path: Directory path
            domain: Filter by domain (e.g., 'ai', 'bc', 'rb')
            pattern: Glob pattern for files
            recursive: Whether to search recursively
            progress: Show progress output
            progress_callback: Custom progress function(current, total, filename)

        Returns:
            List of successfully loaded OntologyBlock objects
        """
        start_time = time.time()

        pages_path = Path(path)
        if not pages_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        # Find all matching files
        if recursive:
            md_files = sorted(pages_path.rglob(pattern))
        else:
            md_files = sorted(pages_path.glob(pattern))

        total_files = len(md_files)
        blocks = []

        for idx, md_file in enumerate(md_files, 1):
            # Progress reporting
            if progress:
                print(f"Loading {idx}/{total_files}: {md_file.name}", end='\r')

            if progress_callback:
                progress_callback(idx, total_files, md_file.name)

            # Load file
            block = self.load_file(md_file)

            # Apply domain filter
            if block and block.term_id:
                if domain and block.get_domain() != domain.lower():
                    continue
                blocks.append(block)

        if progress:
            print(f"\nLoaded {len(blocks)} ontology blocks in {time.time() - start_time:.2f}s")

        return blocks

    def load_files(
        self,
        paths: List[Path],
        domain: Optional[str] = None,
        progress: bool = False
    ) -> List[OntologyBlock]:
        """
        Load multiple specific files.

        Args:
            paths: List of file paths
            domain: Optional domain filter
            progress: Show progress output

        Returns:
            List of successfully loaded OntologyBlock objects
        """
        blocks = []
        total = len(paths)

        for idx, path in enumerate(paths, 1):
            if progress:
                print(f"Loading {idx}/{total}: {path.name}", end='\r')

            block = self.load_file(path)

            if block and block.term_id:
                if domain and block.get_domain() != domain.lower():
                    continue
                blocks.append(block)

        if progress:
            print(f"\nLoaded {len(blocks)} ontology blocks")

        return blocks

    def filter_by_domain(
        self,
        blocks: List[OntologyBlock],
        domain: str
    ) -> List[OntologyBlock]:
        """
        Filter blocks by domain.

        Args:
            blocks: List of ontology blocks
            domain: Domain to filter (e.g., 'ai', 'bc', 'rb')

        Returns:
            Filtered list of blocks
        """
        domain_lower = domain.lower()
        return [b for b in blocks if b.get_domain() == domain_lower]

    def filter_by_pattern(
        self,
        blocks: List[OntologyBlock],
        term_pattern: str
    ) -> List[OntologyBlock]:
        """
        Filter blocks by term-id regex pattern.

        Args:
            blocks: List of ontology blocks
            term_pattern: Regex pattern to match against term-id

        Returns:
            Filtered list of blocks
        """
        pattern = re.compile(term_pattern)
        return [b for b in blocks if b.term_id and pattern.search(b.term_id)]

    def filter_by_status(
        self,
        blocks: List[OntologyBlock],
        status: str
    ) -> List[OntologyBlock]:
        """
        Filter blocks by status.

        Args:
            blocks: List of ontology blocks
            status: Status to filter (e.g., 'complete', 'in-progress')

        Returns:
            Filtered list of blocks
        """
        return [b for b in blocks if b.status and b.status.lower() == status.lower()]

    def get_statistics(self, blocks: List[OntologyBlock]) -> LoaderStatistics:
        """
        Generate comprehensive statistics for loaded blocks.

        Args:
            blocks: List of ontology blocks

        Returns:
            LoaderStatistics object with detailed metrics
        """
        stats = LoaderStatistics()
        stats.total_blocks = len(blocks)
        stats.cache_hits = self._cache_hits
        stats.cache_misses = self._cache_misses

        for block in blocks:
            # Count by domain
            domain = block.get_domain()
            if domain:
                stats.by_domain[domain] += 1

            # Count by status
            if block.status:
                stats.by_status[block.status] += 1

            # Count by maturity
            if block.maturity:
                stats.by_maturity[block.maturity] += 1

            # Count validation errors
            errors = block.validate()
            if errors:
                stats.validation_errors += 1

        return stats

    def get_term_index(self, blocks: List[OntologyBlock]) -> Dict[str, OntologyBlock]:
        """
        Create a term-id -> block lookup index.

        Args:
            blocks: List of ontology blocks

        Returns:
            Dictionary mapping term-id to OntologyBlock
        """
        return {b.term_id: b for b in blocks if b.term_id}

    def get_domain_groups(
        self,
        blocks: List[OntologyBlock]
    ) -> Dict[str, List[OntologyBlock]]:
        """
        Group blocks by domain.

        Args:
            blocks: List of ontology blocks

        Returns:
            Dictionary mapping domain to list of blocks
        """
        groups = defaultdict(list)
        for block in blocks:
            domain = block.get_domain()
            if domain:
                groups[domain].append(block)
        return dict(groups)

    def clear_cache(self) -> None:
        """Clear the file cache."""
        self._file_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache metrics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(1, total_requests)

        return {
            'cache_size': len(self._file_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': round(hit_rate, 3),
            'total_requests': total_requests
        }


def main():
    """Example usage and CLI interface."""
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python ontology_loader.py <directory> [--domain DOMAIN] [--stats]")
        print("\nExample: python ontology_loader.py mainKnowledgeGraph/pages/ --domain ai --stats")
        sys.exit(1)

    path = Path(sys.argv[1])

    # Parse arguments
    domain = None
    show_stats = False

    if '--domain' in sys.argv:
        idx = sys.argv.index('--domain')
        if idx + 1 < len(sys.argv):
            domain = sys.argv[idx + 1]

    if '--stats' in sys.argv:
        show_stats = True

    # Load ontology blocks
    loader = OntologyLoader(cache_size=200)

    if path.is_file():
        print(f"Loading file: {path}")
        block = loader.load_file(path)

        if block:
            print(f"\n✅ Successfully loaded: {block.preferred_term}")
            print(f"   Term ID: {block.term_id}")
            print(f"   Domain: {block.get_domain()}")
            print(f"   IRI: {block.get_full_iri()}")
            print(f"   Status: {block.status}")

            # Validation
            errors = block.validate()
            if errors:
                print(f"\n⚠️  Validation errors:")
                for error in errors:
                    print(f"   - {error}")
            else:
                print(f"\n✅ Validation: PASSED")
        else:
            print("❌ Failed to load ontology block")

    elif path.is_dir():
        print(f"Loading directory: {path}")
        if domain:
            print(f"Filtering by domain: {domain}")

        blocks = loader.load_directory(path, domain=domain, progress=True)

        print(f"\n✅ Successfully loaded {len(blocks)} ontology blocks")

        if show_stats:
            stats = loader.get_statistics(blocks)
            print("\n📊 Statistics:")
            print(json.dumps(stats.to_dict(), indent=2))

            cache_stats = loader.get_cache_stats()
            print("\n💾 Cache Performance:")
            print(json.dumps(cache_stats, indent=2))

    else:
        print(f"Error: {path} is neither a file nor a directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
