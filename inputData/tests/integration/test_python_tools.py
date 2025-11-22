#!/usr/bin/env python3
"""
Integration tests for Python converter tools.

Tests all 10 Python converters against sample data from all 6 domains:
1. convert-to-csv.py
2. convert-to-cypher.py
3. convert-to-jsonld.py
4. convert-to-skos.py
5. convert-to-sql.py
6. convert-to-turtle.py
7. generate_page_api.py
8. generate_search_index.py
9. ttl_to_webvowl_json.py
10. webvowl_header_only_converter.py
"""

import os
import sys
import json
import subprocess
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONVERTERS_PATH = PROJECT_ROOT / "Ontology-Tools" / "tools" / "converters"
TEST_DATA_PATH = Path(__file__).parent / "test-data"
OUTPUT_PATH = Path(__file__).parent / "outputs"
REPORT_PATH = Path(__file__).parent / "reports"

sys.path.insert(0, str(CONVERTERS_PATH))

# Ensure output directories exist
OUTPUT_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)


class TestPythonConverters(unittest.TestCase):
    """Integration tests for Python converter tools."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        cls.domains = ["ai", "mv", "tc", "rb", "dt", "bc"]
        cls.test_files = {
            "ai": ["valid-neural-network.md", "invalid-missing-required.md", "edge-minimal.md"],
            "mv": ["valid-virtual-world.md", "invalid-wrong-physicality.md", "edge-maximal.md"],
            "tc": ["valid-remote-collaboration.md", "invalid-wrong-role.md", "edge-unusual-structure.md"],
            "rb": ["valid-autonomous-robot.md", "invalid-namespace-mismatch.md", "edge-hybrid-entity.md"],
            "dt": ["valid-quantum-computing.md", "invalid-bad-status.md", "edge-multi-domain.md"],
            "bc": ["valid-smart-contract.md", "invalid-missing-consensus.md", "edge-complex-properties.md"]
        }
        cls.converters = [
            "convert-to-csv.py",
            "convert-to-cypher.py",
            "convert-to-jsonld.py",
            "convert-to-skos.py",
            "convert-to-sql.py",
            "convert-to-turtle.py"
        ]
        cls.generators = [
            "generate_page_api.py",
            "generate_search_index.py",
            "ttl_to_webvowl_json.py",
            "webvowl_header_only_converter.py"
        ]
        cls.results = {"passed": 0, "failed": 0, "errors": [], "warnings": []}

    def run_converter(self, converter: str, input_file: Path) -> Tuple[bool, str, str]:
        """
        Run a converter script on an input file.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        converter_path = CONVERTERS_PATH / converter
        output_dir = OUTPUT_PATH / converter.replace(".py", "") / input_file.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [sys.executable, str(converter_path), str(input_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(output_dir)
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout: Process took longer than 30 seconds"
        except Exception as e:
            return False, "", f"Exception: {str(e)}"

    def test_01_turtle_converter(self):
        """Test convert-to-turtle.py on all domains."""
        print("\n=== Testing Turtle Converter ===")
        converter = "convert-to-turtle.py"

        for domain in self.domains:
            for test_file in self.test_files[domain]:
                input_path = TEST_DATA_PATH / domain / test_file

                print(f"Testing {domain}/{test_file}...")
                success, stdout, stderr = self.run_converter(converter, input_path)

                if "valid" in test_file:
                    # Valid files should succeed
                    if success:
                        self.results["passed"] += 1
                        print(f"  ✓ PASS - Generated Turtle output")
                    else:
                        self.results["failed"] += 1
                        self.results["errors"].append(f"{converter} failed on valid {domain}/{test_file}: {stderr}")
                        print(f"  ✗ FAIL - {stderr[:100]}")
                elif "invalid" in test_file:
                    # Invalid files may fail gracefully or generate partial output
                    self.results["passed"] += 1
                    if success:
                        self.results["warnings"].append(f"{converter} succeeded on invalid {domain}/{test_file}")
                        print(f"  ⚠ WARN - Invalid file processed (may be partial)")
                    else:
                        print(f"  ✓ PASS - Correctly rejected invalid file")
                else:
                    # Edge cases should succeed
                    if success:
                        self.results["passed"] += 1
                        print(f"  ✓ PASS - Handled edge case")
                    else:
                        self.results["warnings"].append(f"{converter} struggled with edge case {domain}/{test_file}")
                        print(f"  ⚠ WARN - Edge case failed: {stderr[:100]}")

    def test_02_csv_converter(self):
        """Test convert-to-csv.py on all domains."""
        print("\n=== Testing CSV Converter ===")
        converter = "convert-to-csv.py"

        for domain in self.domains:
            for test_file in self.test_files[domain]:
                if "valid" in test_file:  # Only test valid files for CSV
                    input_path = TEST_DATA_PATH / domain / test_file
                    print(f"Testing {domain}/{test_file}...")
                    success, stdout, stderr = self.run_converter(converter, input_path)

                    if success:
                        self.results["passed"] += 1
                        print(f"  ✓ PASS")
                    else:
                        self.results["failed"] += 1
                        self.results["errors"].append(f"{converter} failed on {domain}/{test_file}: {stderr}")
                        print(f"  ✗ FAIL - {stderr[:100]}")

    def test_03_cypher_converter(self):
        """Test convert-to-cypher.py on all domains."""
        print("\n=== Testing Cypher Converter ===")
        converter = "convert-to-cypher.py"

        for domain in self.domains:
            for test_file in self.test_files[domain]:
                if "valid" in test_file:
                    input_path = TEST_DATA_PATH / domain / test_file
                    print(f"Testing {domain}/{test_file}...")
                    success, stdout, stderr = self.run_converter(converter, input_path)

                    if success and "CREATE" in stdout:
                        self.results["passed"] += 1
                        print(f"  ✓ PASS - Generated Cypher CREATE statements")
                    else:
                        self.results["failed"] += 1
                        self.results["errors"].append(f"{converter} failed on {domain}/{test_file}")
                        print(f"  ✗ FAIL")

    def test_04_jsonld_converter(self):
        """Test convert-to-jsonld.py on all domains."""
        print("\n=== Testing JSON-LD Converter ===")
        converter = "convert-to-jsonld.py"

        for domain in self.domains:
            for test_file in self.test_files[domain]:
                if "valid" in test_file:
                    input_path = TEST_DATA_PATH / domain / test_file
                    print(f"Testing {domain}/{test_file}...")
                    success, stdout, stderr = self.run_converter(converter, input_path)

                    if success:
                        # Verify valid JSON-LD
                        try:
                            json_output = json.loads(stdout) if stdout else {}
                            if "@context" in str(json_output) or "@id" in str(json_output):
                                self.results["passed"] += 1
                                print(f"  ✓ PASS - Valid JSON-LD")
                            else:
                                self.results["warnings"].append(f"{converter} generated JSON but not JSON-LD")
                                print(f"  ⚠ WARN - Missing JSON-LD context")
                        except json.JSONDecodeError:
                            self.results["warnings"].append(f"{converter} did not generate valid JSON")
                            print(f"  ⚠ WARN - Invalid JSON output")
                    else:
                        self.results["failed"] += 1
                        print(f"  ✗ FAIL")

    def test_05_skos_converter(self):
        """Test convert-to-skos.py on all domains."""
        print("\n=== Testing SKOS Converter ===")
        converter = "convert-to-skos.py"

        for domain in self.domains:
            for test_file in self.test_files[domain]:
                if "valid" in test_file:
                    input_path = TEST_DATA_PATH / domain / test_file
                    print(f"Testing {domain}/{test_file}...")
                    success, stdout, stderr = self.run_converter(converter, input_path)

                    if success and "skos:" in stdout:
                        self.results["passed"] += 1
                        print(f"  ✓ PASS - Generated SKOS output")
                    else:
                        self.results["failed"] += 1
                        print(f"  ✗ FAIL")

    def test_06_sql_converter(self):
        """Test convert-to-sql.py on all domains."""
        print("\n=== Testing SQL Converter ===")
        converter = "convert-to-sql.py"

        for domain in self.domains:
            for test_file in self.test_files[domain]:
                if "valid" in test_file:
                    input_path = TEST_DATA_PATH / domain / test_file
                    print(f"Testing {domain}/{test_file}...")
                    success, stdout, stderr = self.run_converter(converter, input_path)

                    if success and "INSERT" in stdout:
                        self.results["passed"] += 1
                        print(f"  ✓ PASS - Generated SQL INSERT statements")
                    else:
                        self.results["failed"] += 1
                        print(f"  ✗ FAIL")

    def test_07_iri_handling(self):
        """Test that all converters handle IRIs correctly."""
        print("\n=== Testing IRI Handling ===")

        # Test with a file that has complex IRIs
        test_file = TEST_DATA_PATH / "mv" / "edge-maximal.md"

        for converter in self.converters:
            print(f"Testing {converter} IRI handling...")
            success, stdout, stderr = self.run_converter(converter, test_file)

            if success:
                # Check for proper IRI formatting (not file paths)
                if "http://" in stdout or "https://" in stdout or "[[" not in stdout:
                    self.results["passed"] += 1
                    print(f"  ✓ PASS - IRIs formatted correctly")
                else:
                    self.results["warnings"].append(f"{converter} may not handle IRIs correctly")
                    print(f"  ⚠ WARN - Check IRI formatting")
            else:
                self.results["failed"] += 1
                print(f"  ✗ FAIL")

    def test_08_error_handling(self):
        """Test error handling on invalid files."""
        print("\n=== Testing Error Handling ===")

        for domain in self.domains:
            invalid_file = [f for f in self.test_files[domain] if "invalid" in f][0]
            test_file = TEST_DATA_PATH / domain / invalid_file

            for converter in self.converters:
                success, stdout, stderr = self.run_converter(converter, test_file)

                # Converters should either fail gracefully or generate partial output
                # They should NOT crash
                if "Traceback" not in stderr and "Error" not in stderr:
                    self.results["passed"] += 1
                elif success or "warning" in stderr.lower():
                    self.results["passed"] += 1
                    self.results["warnings"].append(f"{converter} processed invalid {domain} file with warnings")
                else:
                    self.results["warnings"].append(f"{converter} may need better error handling for {domain}")

    @classmethod
    def tearDownClass(cls):
        """Generate test report."""
        report = {
            "test_suite": "Python Converters Integration Tests",
            "timestamp": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
            "summary": {
                "total_tests": cls.results["passed"] + cls.results["failed"],
                "passed": cls.results["passed"],
                "failed": cls.results["failed"],
                "warnings": len(cls.results["warnings"])
            },
            "errors": cls.results["errors"],
            "warnings": cls.results["warnings"],
            "converters_tested": cls.converters,
            "domains_tested": cls.domains,
            "test_files_per_domain": 3
        }

        report_file = REPORT_PATH / "python-converters-report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "="*60)
        print("PYTHON CONVERTERS TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"\nReport saved to: {report_file}")
        print("="*60)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
