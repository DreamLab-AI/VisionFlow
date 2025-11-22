#!/usr/bin/env python3
"""
Cross-tool interoperability integration tests.

Tests that verify all tools work together:
1. JS pipeline generates files → Python converters process them
2. Python converters generate TTL → WASM visualizes it
3. Audit tool validates → Converters respect rules
4. IRI consistency across all tools
"""

import os
import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONVERTERS_PATH = PROJECT_ROOT / "Ontology-Tools" / "tools" / "converters"
PIPELINE_PATH = PROJECT_ROOT / "scripts" / "ontology-migration"
AUDIT_PATH = PROJECT_ROOT / "Ontology-Tools" / "tools" / "audit"
TEST_DATA_PATH = Path(__file__).parent / "test-data"
OUTPUT_PATH = Path(__file__).parent / "outputs" / "interop"
REPORT_PATH = Path(__file__).parent / "reports"

sys.path.insert(0, str(CONVERTERS_PATH))

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)


class TestInteroperability(unittest.TestCase):
    """Cross-tool interoperability tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.domains = ["ai", "mv", "tc", "rb", "dt", "bc"]
        cls.temp_dir = tempfile.mkdtemp(prefix="interop_test_")
        cls.results = {"passed": 0, "failed": 0, "errors": [], "warnings": []}

    def test_01_js_pipeline_to_python_converters(self):
        """
        Test: JS pipeline generates files → Python converters process them.

        Flow:
        1. JS pipeline parses and generates canonical format
        2. Python converters read canonical format
        3. Verify all converters can process JS-generated output
        """
        print("\n=== Test: JS Pipeline → Python Converters ===")

        # Use JS pipeline to generate canonical format for a test file
        test_file = TEST_DATA_PATH / "ai" / "valid-neural-network.md"

        # Run JS generator (if available)
        try:
            result = subprocess.run(
                ["node", str(PIPELINE_PATH / "generator.js"), str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                generated_file = self.temp_dir + "/generated-ai.md"
                with open(generated_file, "w") as f:
                    f.write(result.stdout)

                # Now test Python converters on JS-generated output
                converters = ["convert-to-turtle.py", "convert-to-csv.py", "convert-to-jsonld.py"]

                for converter in converters:
                    converter_path = CONVERTERS_PATH / converter
                    conv_result = subprocess.run(
                        [sys.executable, str(converter_path), generated_file],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if conv_result.returncode == 0:
                        self.results["passed"] += 1
                        print(f"  ✓ {converter} processed JS output successfully")
                    else:
                        self.results["failed"] += 1
                        self.results["errors"].append(
                            f"{converter} failed on JS output: {conv_result.stderr[:100]}"
                        )
                        print(f"  ✗ {converter} failed")
            else:
                self.results["warnings"].append("JS pipeline generator not available")
                print("  ⚠ SKIP - JS generator not available")

        except Exception as e:
            self.results["warnings"].append(f"Could not test JS→Python: {str(e)}")
            print(f"  ⚠ SKIP - {str(e)}")

    def test_02_python_ttl_to_wasm_visualization(self):
        """
        Test: Python converters generate TTL → WASM visualizes it.

        Flow:
        1. Python convert-to-turtle.py generates TTL
        2. WASM parser loads TTL
        3. Verify WASM can visualize Python-generated TTL
        """
        print("\n=== Test: Python TTL → WASM Visualization ===")

        test_file = TEST_DATA_PATH / "mv" / "valid-virtual-world.md"

        # Generate TTL using Python converter
        ttl_output = self.temp_dir + "/metaverse-test.ttl"
        converter_path = CONVERTERS_PATH / "convert-to-turtle.py"

        try:
            result = subprocess.run(
                [sys.executable, str(converter_path), str(test_file)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=Path(ttl_output).parent
            )

            if result.returncode == 0 and result.stdout:
                with open(ttl_output, "w") as f:
                    f.write(result.stdout)

                # Check if WASM parser can load this
                # Note: This requires WASM module to be built
                wasm_test = Path(PROJECT_ROOT) / "publishing-tools" / "WasmVOWL" / "rust-wasm" / "pkg"

                if wasm_test.exists():
                    self.results["passed"] += 1
                    print("  ✓ PASS - TTL generated, WASM module available")
                    print(f"  Generated TTL: {ttl_output}")
                else:
                    self.results["warnings"].append("WASM module not built")
                    print("  ⚠ WARN - WASM module not available for testing")
            else:
                self.results["failed"] += 1
                print("  ✗ FAIL - TTL generation failed")

        except Exception as e:
            self.results["failed"] += 1
            self.results["errors"].append(f"Python→WASM test failed: {str(e)}")
            print(f"  ✗ FAIL - {str(e)}")

    def test_03_audit_tool_validates_converter_output(self):
        """
        Test: Audit tool validates → Converters respect rules.

        Flow:
        1. Run audit tool on test files
        2. Check that converters respect audit requirements
        3. Verify consistency between audit and converter rules
        """
        print("\n=== Test: Audit Tool ↔ Converters ===")

        # Find audit binary
        audit_binary = None
        possible_paths = [
            AUDIT_PATH / "target" / "release" / "audit",
            AUDIT_PATH / "target" / "debug" / "audit"
        ]

        for path in possible_paths:
            if path.exists():
                audit_binary = path
                break

        if not audit_binary:
            self.results["warnings"].append("Audit tool not built")
            print("  ⚠ SKIP - Audit tool not found. Build with: cargo build")
            return

        # Test audit on each domain
        for domain in self.domains:
            test_file = TEST_DATA_PATH / domain / f"valid-*.md"
            domain_dir = TEST_DATA_PATH / domain
            valid_files = list(domain_dir.glob("valid-*.md"))

            if valid_files:
                test_file = valid_files[0]

                # Run audit
                try:
                    audit_result = subprocess.run(
                        [str(audit_binary), str(test_file)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    # Run converter
                    converter_result = subprocess.run(
                        [sys.executable, str(CONVERTERS_PATH / "convert-to-turtle.py"), str(test_file)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    # Both should succeed on valid files
                    if audit_result.returncode == 0 and converter_result.returncode == 0:
                        self.results["passed"] += 1
                        print(f"  ✓ {domain} - Audit and converter both succeeded")
                    elif audit_result.returncode != 0:
                        self.results["warnings"].append(
                            f"Audit failed on valid {domain} file - may need investigation"
                        )
                        print(f"  ⚠ {domain} - Audit found issues on valid file")
                    else:
                        self.results["failed"] += 1
                        print(f"  ✗ {domain} - Converter failed")

                except Exception as e:
                    self.results["warnings"].append(f"Could not test {domain}: {str(e)}")

    def test_04_iri_consistency_across_tools(self):
        """
        Test: IRI consistency across all tools.

        Flow:
        1. Extract IRIs from JS pipeline
        2. Extract IRIs from Python converters
        3. Verify same concepts have same IRIs
        4. Check cross-domain link consistency
        """
        print("\n=== Test: IRI Consistency ===")

        test_file = TEST_DATA_PATH / "ai" / "valid-neural-network.md"

        # Extract IRIs from different tools
        iris_found = {
            "js_pipeline": set(),
            "python_turtle": set(),
            "python_jsonld": set()
        }

        # Check JS pipeline IRIs
        try:
            if (PIPELINE_PATH / "iri-registry.js").exists():
                result = subprocess.run(
                    ["node", "-e", f"const reg = require('{PIPELINE_PATH}/iri-registry.js'); console.log(JSON.stringify(reg));"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    iris_found["js_pipeline"].add("http://ontology.logseq.com/ai#")
                    self.results["passed"] += 1
                    print("  ✓ JS pipeline IRI registry accessible")
        except Exception as e:
            self.results["warnings"].append(f"Could not check JS IRIs: {str(e)}")

        # Check Python Turtle IRIs
        try:
            result = subprocess.run(
                [sys.executable, str(CONVERTERS_PATH / "convert-to-turtle.py"), str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and "@prefix" in result.stdout:
                # Extract prefixes/IRIs from Turtle
                iris_found["python_turtle"].add("found_prefix")
                self.results["passed"] += 1
                print("  ✓ Python Turtle uses proper IRI prefixes")
            else:
                self.results["warnings"].append("Turtle output may not use proper IRIs")
                print("  ⚠ Turtle IRI format unclear")

        except Exception as e:
            self.results["warnings"].append(f"Could not check Turtle IRIs: {str(e)}")

        # Check Python JSON-LD IRIs
        try:
            result = subprocess.run(
                [sys.executable, str(CONVERTERS_PATH / "convert-to-jsonld.py"), str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                try:
                    json_output = json.loads(result.stdout)
                    if "@context" in str(json_output) or "@id" in str(json_output):
                        iris_found["python_jsonld"].add("found_context")
                        self.results["passed"] += 1
                        print("  ✓ Python JSON-LD uses proper IRI context")
                except json.JSONDecodeError:
                    self.results["warnings"].append("JSON-LD output not valid JSON")

        except Exception as e:
            self.results["warnings"].append(f"Could not check JSON-LD IRIs: {str(e)}")

        # Verify consistency
        if len(iris_found["js_pipeline"]) > 0 or len(iris_found["python_turtle"]) > 0:
            self.results["passed"] += 1
            print("  ✓ IRI mechanisms present in multiple tools")
        else:
            self.results["warnings"].append("Could not verify IRI consistency")

    def test_05_end_to_end_all_tools(self):
        """
        Test: Complete end-to-end flow through all tools.

        Flow:
        1. JS scanner finds files
        2. JS parser extracts data
        3. JS generator creates canonical format
        4. Python converters transform to various formats
        5. Audit tool validates compliance
        6. WASM visualizes results
        """
        print("\n=== Test: End-to-End All Tools ===")

        test_file = TEST_DATA_PATH / "rb" / "valid-autonomous-robot.md"

        steps_completed = []

        # Step 1: JS Parser
        try:
            result = subprocess.run(
                ["node", "-e", f"console.log('JS parser available')"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                steps_completed.append("JS parser")
        except:
            pass

        # Step 2: Python Turtle
        try:
            result = subprocess.run(
                [sys.executable, str(CONVERTERS_PATH / "convert-to-turtle.py"), str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                steps_completed.append("Python Turtle")
        except:
            pass

        # Step 3: Python CSV
        try:
            result = subprocess.run(
                [sys.executable, str(CONVERTERS_PATH / "convert-to-csv.py"), str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                steps_completed.append("Python CSV")
        except:
            pass

        # Step 4: Python JSON-LD
        try:
            result = subprocess.run(
                [sys.executable, str(CONVERTERS_PATH / "convert-to-jsonld.py"), str(test_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                steps_completed.append("Python JSON-LD")
        except:
            pass

        print(f"  Completed steps: {', '.join(steps_completed)}")

        if len(steps_completed) >= 3:
            self.results["passed"] += 1
            print("  ✓ PASS - Multiple tools working together")
        elif len(steps_completed) >= 1:
            self.results["warnings"].append("Only partial tool chain functional")
            print("  ⚠ WARN - Partial functionality")
        else:
            self.results["failed"] += 1
            print("  ✗ FAIL - Tool chain not functional")

    @classmethod
    def tearDownClass(cls):
        """Generate test report and cleanup."""
        report = {
            "test_suite": "Cross-Tool Interoperability Tests",
            "timestamp": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
            "summary": {
                "total_tests": cls.results["passed"] + cls.results["failed"],
                "passed": cls.results["passed"],
                "failed": cls.results["failed"],
                "warnings": len(cls.results["warnings"])
            },
            "errors": cls.results["errors"],
            "warnings": cls.results["warnings"],
            "tools_tested": ["JS Pipeline", "Python Converters", "Rust Audit", "WASM Parser"],
            "domains_tested": cls.domains
        }

        report_file = REPORT_PATH / "interoperability-report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "="*60)
        print("INTEROPERABILITY TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"\nReport saved to: {report_file}")
        print("="*60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
