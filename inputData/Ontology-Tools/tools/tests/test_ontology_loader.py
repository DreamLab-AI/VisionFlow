#!/usr/bin/env python3
"""
Tests for OntologyLoader
=========================

Comprehensive test suite for the unified ontology loader library.
"""

import unittest
import tempfile
from pathlib import Path
from typing import List

import sys
lib_path = str(Path(__file__).parent.parent / 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from ontology_loader import OntologyLoader, LoaderStatistics
from ontology_block_parser import OntologyBlock


class TestOntologyLoader(unittest.TestCase):
    """Test suite for OntologyLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = OntologyLoader(cache_size=10, strict_validation=False)
        self.temp_dir = tempfile.mkdtemp()

    def create_test_file(self, filename: str, domain: str, term_id: str, status: str = "complete") -> Path:
        """Create a test ontology file."""
        content = f"""
- ### OntologyBlock
  id:: test-ontology-{term_id}
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - term-id:: {term_id}
    - preferred-term:: Test {domain.upper()} Term
    - source-domain:: {domain}
    - status:: {status}
    - version:: 1.0
    - last-updated:: 2025-01-01

  - **Definition**
    - definition:: A test ontology block for {domain} domain.
    - maturity:: mature
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: {domain}:TestClass
    - owl:physicality:: VirtualEntity
    - owl:role:: Object

  - #### Relationships
    - is-subclass-of:: [[{domain}:ParentClass]]

## Additional Content
This is additional content below the ontology block.
"""
        file_path = Path(self.temp_dir) / filename
        file_path.write_text(content)
        return file_path

    def test_load_single_file(self):
        """Test loading a single ontology file."""
        file_path = self.create_test_file("test_ai.md", "ai", "AI-0001")

        block = self.loader.load_file(file_path)

        self.assertIsNotNone(block)
        self.assertEqual(block.term_id, "AI-0001")
        self.assertEqual(block.preferred_term, "Test AI Term")
        self.assertEqual(block.get_domain(), "ai")

    def test_cache_hit(self):
        """Test that caching works correctly."""
        file_path = self.create_test_file("test_cache.md", "ai", "AI-0002")

        # First load - cache miss
        block1 = self.loader.load_file(file_path)
        cache_stats_1 = self.loader.get_cache_stats()
        self.assertEqual(cache_stats_1['cache_misses'], 1)
        self.assertEqual(cache_stats_1['cache_hits'], 0)

        # Second load - cache hit
        block2 = self.loader.load_file(file_path)
        cache_stats_2 = self.loader.get_cache_stats()
        self.assertEqual(cache_stats_2['cache_hits'], 1)

        # Should be the same block
        self.assertEqual(block1.term_id, block2.term_id)

    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        # Create more files than cache size
        files = []
        for i in range(15):
            file_path = self.create_test_file(f"test_{i}.md", "ai", f"AI-{i:04d}")
            files.append(file_path)

        # Load all files
        for file_path in files:
            self.loader.load_file(file_path)

        # Cache should not exceed max size
        cache_stats = self.loader.get_cache_stats()
        self.assertLessEqual(cache_stats['cache_size'], 10)

    def test_load_directory(self):
        """Test loading all files from a directory."""
        # Create test files
        self.create_test_file("ai_1.md", "ai", "AI-0001")
        self.create_test_file("ai_2.md", "ai", "AI-0002")
        self.create_test_file("bc_1.md", "bc", "BC-0001")

        blocks = self.loader.load_directory(Path(self.temp_dir))

        self.assertEqual(len(blocks), 3)

    def test_domain_filtering(self):
        """Test filtering blocks by domain."""
        # Create test files from different domains
        self.create_test_file("ai_1.md", "ai", "AI-0001")
        self.create_test_file("ai_2.md", "ai", "AI-0002")
        self.create_test_file("bc_1.md", "bc", "BC-0001")

        # Load with domain filter
        blocks = self.loader.load_directory(Path(self.temp_dir), domain='ai')

        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(block.get_domain(), "ai")

    def test_filter_by_domain(self):
        """Test filtering existing blocks by domain."""
        # Load all blocks
        self.create_test_file("ai_1.md", "ai", "AI-0001")
        self.create_test_file("bc_1.md", "bc", "BC-0001")

        all_blocks = self.loader.load_directory(Path(self.temp_dir))

        # Filter by domain
        ai_blocks = self.loader.filter_by_domain(all_blocks, 'ai')

        self.assertEqual(len(ai_blocks), 1)
        self.assertEqual(ai_blocks[0].get_domain(), "ai")

    def test_filter_by_pattern(self):
        """Test filtering blocks by term-id pattern."""
        # Create blocks
        self.create_test_file("ai_1.md", "ai", "AI-0001")
        self.create_test_file("ai_2.md", "ai", "AI-0002")
        self.create_test_file("ai_10.md", "ai", "AI-0010")

        all_blocks = self.loader.load_directory(Path(self.temp_dir))

        # Filter by pattern (only single-digit IDs)
        filtered = self.loader.filter_by_pattern(all_blocks, r'AI-000[12]')

        self.assertEqual(len(filtered), 2)

    def test_filter_by_status(self):
        """Test filtering blocks by status."""
        self.create_test_file("complete.md", "ai", "AI-0001", status="complete")
        self.create_test_file("in_progress.md", "ai", "AI-0002", status="in-progress")

        all_blocks = self.loader.load_directory(Path(self.temp_dir))

        # Filter by status
        complete_blocks = self.loader.filter_by_status(all_blocks, "complete")

        self.assertEqual(len(complete_blocks), 1)
        self.assertEqual(complete_blocks[0].status, "complete")

    def test_get_statistics(self):
        """Test statistics generation."""
        # Create test files
        self.create_test_file("ai_1.md", "ai", "AI-0001", status="complete")
        self.create_test_file("ai_2.md", "ai", "AI-0002", status="complete")
        self.create_test_file("bc_1.md", "bc", "BC-0001", status="in-progress")

        blocks = self.loader.load_directory(Path(self.temp_dir))
        stats = self.loader.get_statistics(blocks)

        self.assertEqual(stats.total_blocks, 3)
        self.assertEqual(stats.by_domain['ai'], 2)
        self.assertEqual(stats.by_domain['bc'], 1)
        self.assertEqual(stats.by_status['complete'], 2)
        self.assertEqual(stats.by_status['in-progress'], 1)

    def test_get_term_index(self):
        """Test term-id index creation."""
        self.create_test_file("ai_1.md", "ai", "AI-0001")
        self.create_test_file("bc_1.md", "bc", "BC-0001")

        blocks = self.loader.load_directory(Path(self.temp_dir))
        index = self.loader.get_term_index(blocks)

        self.assertEqual(len(index), 2)
        self.assertIn("AI-0001", index)
        self.assertIn("BC-0001", index)
        self.assertEqual(index["AI-0001"].preferred_term, "Test AI Term")

    def test_get_domain_groups(self):
        """Test grouping blocks by domain."""
        self.create_test_file("ai_1.md", "ai", "AI-0001")
        self.create_test_file("ai_2.md", "ai", "AI-0002")
        self.create_test_file("bc_1.md", "bc", "BC-0001")

        blocks = self.loader.load_directory(Path(self.temp_dir))
        groups = self.loader.get_domain_groups(blocks)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups['ai']), 2)
        self.assertEqual(len(groups['bc']), 1)

    def test_clear_cache(self):
        """Test cache clearing."""
        file_path = self.create_test_file("test.md", "ai", "AI-0001")

        # Load file to populate cache
        self.loader.load_file(file_path)

        # Clear cache
        self.loader.clear_cache()

        cache_stats = self.loader.get_cache_stats()
        self.assertEqual(cache_stats['cache_size'], 0)
        self.assertEqual(cache_stats['cache_hits'], 0)
        self.assertEqual(cache_stats['cache_misses'], 0)

    def test_load_invalid_file(self):
        """Test loading a file without ontology block."""
        invalid_file = Path(self.temp_dir) / "invalid.md"
        invalid_file.write_text("# Just a regular markdown file\nNo ontology block here.")

        block = self.loader.load_file(invalid_file)

        self.assertIsNone(block)

    def test_load_nonexistent_directory(self):
        """Test loading from a non-existent directory."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_directory(Path("/nonexistent/path"))

    def test_statistics_to_dict(self):
        """Test statistics serialization to dictionary."""
        self.create_test_file("ai_1.md", "ai", "AI-0001")

        blocks = self.loader.load_directory(Path(self.temp_dir))
        stats = self.loader.get_statistics(blocks)
        stats_dict = stats.to_dict()

        self.assertIn('total_blocks', stats_dict)
        self.assertIn('by_domain', stats_dict)
        self.assertIn('cache_hit_rate', stats_dict)
        self.assertEqual(stats_dict['total_blocks'], 1)

    def test_load_files_list(self):
        """Test loading specific list of files."""
        file1 = self.create_test_file("ai_1.md", "ai", "AI-0001")
        file2 = self.create_test_file("bc_1.md", "bc", "BC-0001")
        self.create_test_file("ai_2.md", "ai", "AI-0002")  # Not in list

        blocks = self.loader.load_files([file1, file2])

        self.assertEqual(len(blocks), 2)
        term_ids = {b.term_id for b in blocks}
        self.assertIn("AI-0001", term_ids)
        self.assertIn("BC-0001", term_ids)
        self.assertNotIn("AI-0002", term_ids)


class TestLoaderStatistics(unittest.TestCase):
    """Test suite for LoaderStatistics."""

    def test_statistics_creation(self):
        """Test creating empty statistics."""
        stats = LoaderStatistics()

        self.assertEqual(stats.total_blocks, 0)
        self.assertEqual(len(stats.by_domain), 0)
        self.assertEqual(stats.validation_errors, 0)

    def test_statistics_to_dict(self):
        """Test converting statistics to dictionary."""
        stats = LoaderStatistics()
        stats.total_blocks = 5
        stats.by_domain['ai'] = 3
        stats.by_domain['bc'] = 2
        stats.cache_hits = 10
        stats.cache_misses = 5

        stats_dict = stats.to_dict()

        self.assertEqual(stats_dict['total_blocks'], 5)
        self.assertEqual(stats_dict['by_domain']['ai'], 3)
        self.assertEqual(stats_dict['cache_hit_rate'], 0.667)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
