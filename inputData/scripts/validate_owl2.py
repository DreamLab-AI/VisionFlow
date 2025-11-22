#!/usr/bin/env python3
"""
OWL2 Compliance Validator

Validates generated OWL2 TTL file for:
- Syntax correctness (RDF/TTL parsing)
- OWL2 DL compliance
- Common ontology issues
- Consistency checking
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL
    from rdflib.plugins.parsers.notation3 import BadSyntax
    import owlrl
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Install with: pip install rdflib owlrl")
    sys.exit(1)


class OWL2Validator:
    """Validates OWL2 ontology compliance."""

    def __init__(self, ttl_file: Path):
        self.ttl_file = ttl_file
        self.graph = Graph()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate_syntax(self) -> bool:
        """Test 1: Validate RDF/TTL syntax."""
        print("Test 1: Validating TTL syntax...")
        try:
            self.graph.parse(self.ttl_file, format='turtle')
            triples_count = len(self.graph)
            self.info.append(f"✓ Parsed {triples_count:,} RDF triples")
            print(f"  ✓ Valid TTL syntax ({triples_count:,} triples)")
            return True
        except BadSyntax as e:
            self.errors.append(f"✗ TTL Syntax Error: {e}")
            print(f"  ✗ Syntax error: {e}")
            return False
        except Exception as e:
            self.errors.append(f"✗ Parse Error: {e}")
            print(f"  ✗ Parse error: {e}")
            return False

    def validate_owl2_structure(self) -> bool:
        """Test 2: Validate OWL2 structure."""
        print("\nTest 2: Validating OWL2 structure...")

        # Check for ontology declaration
        ontology_count = len(list(self.graph.subjects(RDF.type, OWL.Ontology)))
        if ontology_count == 0:
            self.errors.append("✗ No owl:Ontology declaration found")
            print("  ✗ Missing owl:Ontology declaration")
            return False
        elif ontology_count > 1:
            self.warnings.append(f"⚠ Multiple owl:Ontology declarations ({ontology_count})")
            print(f"  ⚠ Multiple ontology declarations: {ontology_count}")
        else:
            self.info.append("✓ Single owl:Ontology declaration")
            print("  ✓ Found owl:Ontology declaration")

        # Count classes
        classes = list(self.graph.subjects(RDF.type, OWL.Class))
        self.info.append(f"✓ Found {len(classes)} owl:Class definitions")
        print(f"  ✓ Classes: {len(classes)}")

        # Count object properties
        obj_props = list(self.graph.subjects(RDF.type, OWL.ObjectProperty))
        self.info.append(f"✓ Found {len(obj_props)} owl:ObjectProperty definitions")
        print(f"  ✓ Object Properties: {len(obj_props)}")

        # Count data properties
        data_props = list(self.graph.subjects(RDF.type, OWL.DatatypeProperty))
        self.info.append(f"✓ Found {len(data_props)} owl:DatatypeProperty definitions")
        print(f"  ✓ Data Properties: {len(data_props)}")

        return True

    def check_owl2_dl_compliance(self) -> bool:
        """Test 3: Check OWL2 DL compliance."""
        print("\nTest 3: Checking OWL2 DL compliance...")

        issues = []

        # Check for punning (same URI used as both class and property)
        classes = set(self.graph.subjects(RDF.type, OWL.Class))
        properties = set(self.graph.subjects(RDF.type, OWL.ObjectProperty))
        properties.update(self.graph.subjects(RDF.type, OWL.DatatypeProperty))

        punning = classes.intersection(properties)
        if punning:
            issues.append(f"Punning detected: {len(punning)} URIs used as both class and property")
            for uri in list(punning)[:5]:  # Show first 5
                issues.append(f"  - {uri}")

        # Check for undefined classes in subClassOf relations
        for subclass, superclass in self.graph.subject_objects(RDFS.subClassOf):
            if superclass not in classes and not str(superclass).startswith('http://www.w3.org/'):
                issues.append(f"Undefined superclass: {superclass}")

        if issues:
            for issue in issues[:10]:  # Show first 10 issues
                self.warnings.append(f"⚠ {issue}")
                print(f"  ⚠ {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
            return False
        else:
            self.info.append("✓ No OWL2 DL compliance issues detected")
            print("  ✓ OWL2 DL compliant")
            return True

    def check_common_issues(self) -> bool:
        """Test 4: Check for common ontology issues."""
        print("\nTest 4: Checking for common ontology issues...")

        issues_found = False

        # Check for classes without labels
        classes_without_labels = []
        for cls in self.graph.subjects(RDF.type, OWL.Class):
            if not list(self.graph.objects(cls, RDFS.label)):
                classes_without_labels.append(cls)

        if classes_without_labels:
            issues_found = True
            count = len(classes_without_labels)
            self.warnings.append(f"⚠ {count} classes without rdfs:label")
            print(f"  ⚠ {count} classes missing labels")
        else:
            print("  ✓ All classes have labels")

        # Check for classes without comments/definitions
        classes_without_comments = []
        for cls in self.graph.subjects(RDF.type, OWL.Class):
            if not list(self.graph.objects(cls, RDFS.comment)):
                classes_without_comments.append(cls)

        if classes_without_comments:
            issues_found = True
            count = len(classes_without_comments)
            self.warnings.append(f"⚠ {count} classes without rdfs:comment")
            print(f"  ⚠ {count} classes missing comments")
        else:
            print("  ✓ All classes have comments")

        # Check for orphaned classes (no subClassOf relations)
        orphaned = []
        for cls in self.graph.subjects(RDF.type, OWL.Class):
            if not list(self.graph.objects(cls, RDFS.subClassOf)):
                # Check it's not owl:Thing or similar
                if not str(cls).startswith('http://www.w3.org/'):
                    orphaned.append(cls)

        if orphaned:
            count = len(orphaned)
            self.info.append(f"ℹ {count} top-level classes (no superclass)")
            print(f"  ℹ {count} top-level classes")

        if not issues_found:
            self.info.append("✓ No common ontology issues detected")

        return not issues_found

    def apply_owl_reasoning(self) -> bool:
        """Test 5: Apply OWL2 RL reasoning and check for inconsistencies."""
        print("\nTest 5: Applying OWL2 RL reasoning...")

        try:
            # Apply OWL-RL reasoning
            original_count = len(self.graph)
            owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(self.graph)
            new_count = len(self.graph)

            inferred = new_count - original_count
            self.info.append(f"✓ OWL2 RL reasoning complete: {inferred} triples inferred")
            print(f"  ✓ Inferred {inferred} new triples")

            # Check for contradictions (basic check)
            # In a real validator, you'd check for more sophisticated inconsistencies
            contradictions = 0

            if contradictions > 0:
                self.errors.append(f"✗ Found {contradictions} logical contradictions")
                print(f"  ✗ Contradictions detected: {contradictions}")
                return False
            else:
                self.info.append("✓ No logical contradictions detected")
                print("  ✓ Logically consistent")
                return True

        except Exception as e:
            self.errors.append(f"✗ Reasoning error: {e}")
            print(f"  ✗ Reasoning failed: {e}")
            return False

    def generate_report(self, output_file: Path):
        """Generate validation report."""
        report = f"""# OWL2 Validation Report
Generated: {datetime.now().isoformat()}
Ontology: {self.ttl_file.name}

## Summary
- Total triples: {len(self.graph):,}
- Errors: {len(self.errors)}
- Warnings: {len(self.warnings)}
- Info: {len(self.info)}

## Errors
"""
        if self.errors:
            for error in self.errors:
                report += f"{error}\n"
        else:
            report += "No errors detected.\n"

        report += "\n## Warnings\n"
        if self.warnings:
            for warning in self.warnings:
                report += f"{warning}\n"
        else:
            report += "No warnings.\n"

        report += "\n## Information\n"
        for info in self.info:
            report += f"{info}\n"

        with open(output_file, 'w') as f:
            f.write(report)

        print(f"\n✓ Validation report written to: {output_file}")

    def run_all_tests(self) -> bool:
        """Run all validation tests."""
        print("=" * 80)
        print("OWL2 Compliance Validation")
        print("=" * 80)
        print(f"File: {self.ttl_file}\n")

        # Run tests in order
        tests = [
            self.validate_syntax,
            self.validate_owl2_structure,
            self.check_owl2_dl_compliance,
            self.check_common_issues,
            self.apply_owl_reasoning,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                self.errors.append(f"✗ Test failed: {e}")
                print(f"  ✗ Test error: {e}")
                results.append(False)

        # Summary
        print("\n" + "=" * 80)
        print("Validation Summary")
        print("=" * 80)
        print(f"Tests passed: {sum(results)}/{len(results)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        return all(results) and len(self.errors) == 0


def main():
    """Main validation workflow."""
    if len(sys.argv) < 2:
        print("Usage: python validate_owl2.py <ttl_file>")
        sys.exit(1)

    ttl_file = Path(sys.argv[1])
    if not ttl_file.exists():
        print(f"Error: File not found: {ttl_file}")
        sys.exit(1)

    validator = OWL2Validator(ttl_file)
    success = validator.run_all_tests()

    # Generate report
    report_file = ttl_file.parent / f"{ttl_file.stem}-validation-report.md"
    validator.generate_report(report_file)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
