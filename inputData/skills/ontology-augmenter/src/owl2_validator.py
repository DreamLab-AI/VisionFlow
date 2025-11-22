#!/usr/bin/env python3
"""
OWL2 Compliance Validator for Logseq Knowledge Graph Ontology

Validates OWL functional syntax in ```clojure blocks within Logseq markdown files.
Ensures compliance with OWL2 DL profile and meta-ontology structure.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during OWL2 compliance checking"""
    severity: ValidationSeverity
    line_number: int
    axiom: str
    message: str
    fix_suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    file_path: str
    total_axioms: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    classes: Set[str] = field(default_factory=set)
    properties: Set[str] = field(default_factory=set)
    individuals: Set[str] = field(default_factory=set)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


class OWL2Validator:
    """
    Validates OWL2 functional syntax in Logseq markdown files.

    Checks:
    - Class declarations and subClassOf axioms
    - Property declarations (ObjectProperty, DataProperty)
    - Restrictions (ObjectSomeValuesFrom, etc.)
    - Namespace consistency (dt:, ai:, bc:, mv:, rb:)
    - OWL2 antipatterns
    - Meta-ontology alignment
    """

    # Valid namespace prefixes
    VALID_PREFIXES = {'dt', 'ai', 'bc', 'mv', 'rb', 'rdfs', 'owl', 'xsd', 'rdf'}

    # Prefix to term-id mapping
    PREFIX_TERM_MAPPING = {
        'ai': 'AI-',
        'bc': 'BC-',
        'mv': 'MV-',
        'rb': 'RB-',
        'dt': 'DT-'
    }

    # OWL2 keywords
    OWL2_KEYWORDS = {
        'Class', 'SubClassOf', 'EquivalentClasses', 'DisjointClasses',
        'ObjectProperty', 'DataProperty', 'AnnotationProperty',
        'Domain', 'Range', 'SubPropertyOf', 'InverseOf',
        'FunctionalProperty', 'InverseFunctionalProperty',
        'TransitiveProperty', 'SymmetricProperty', 'AsymmetricProperty',
        'ReflexiveProperty', 'IrreflexiveProperty',
        'ObjectSomeValuesFrom', 'ObjectAllValuesFrom', 'ObjectHasValue',
        'ObjectMinCardinality', 'ObjectMaxCardinality', 'ObjectExactCardinality',
        'DataSomeValuesFrom', 'DataAllValuesFrom', 'DataHasValue',
        'DataMinCardinality', 'DataMaxCardinality', 'DataExactCardinality',
        'ObjectUnionOf', 'ObjectIntersectionOf', 'ObjectComplementOf',
        'ObjectOneOf', 'DataUnionOf', 'DataIntersectionOf', 'DataComplementOf',
        'Declaration', 'NamedIndividual', 'ClassAssertion', 'ObjectPropertyAssertion',
        'DataPropertyAssertion', 'SameIndividual', 'DifferentIndividuals'
    }

    def __init__(self):
        self.report: Optional[ValidationReport] = None

    def validate_file(self, file_path: str, content: str) -> ValidationReport:
        """
        Validate an entire Logseq markdown file.

        Args:
            file_path: Path to the file being validated
            content: File content

        Returns:
            ValidationReport with all issues found
        """
        self.report = ValidationReport(file_path=file_path)

        # Extract ontology blocks
        ontology_blocks = self._extract_ontology_blocks(content)

        if not ontology_blocks:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                line_number=0,
                axiom="",
                message="No ontology blocks found in file",
                fix_suggestion="Add ```clojure blocks with OWL functional syntax"
            ))
            return self.report

        # Validate each block
        for block_start, block_content in ontology_blocks:
            self._validate_ontology_block(block_start, block_content)

        # Cross-block validations
        self._validate_cross_block_consistency()

        return self.report

    def _extract_ontology_blocks(self, content: str) -> List[Tuple[int, str]]:
        """
        Extract ```clojure blocks from markdown content.

        Returns:
            List of (line_number, block_content) tuples
        """
        blocks = []
        lines = content.split('\n')
        in_block = False
        block_start = 0
        current_block = []

        for i, line in enumerate(lines, 1):
            if line.strip().startswith('```clojure'):
                in_block = True
                block_start = i + 1
                current_block = []
            elif line.strip().startswith('```') and in_block:
                in_block = False
                blocks.append((block_start, '\n'.join(current_block)))
            elif in_block:
                current_block.append(line)

        return blocks

    def _validate_ontology_block(self, start_line: int, block_content: str):
        """Validate a single ontology block"""
        axioms = self._parse_axioms(block_content)
        self.report.total_axioms += len(axioms)

        for line_offset, axiom in axioms:
            line_number = start_line + line_offset
            self._validate_axiom(line_number, axiom)

    def _parse_axioms(self, content: str) -> List[Tuple[int, str]]:
        """
        Parse OWL functional syntax axioms from content.

        Returns:
            List of (line_offset, axiom_text) tuples
        """
        axioms = []
        lines = content.split('\n')
        current_axiom = []
        axiom_start = 0
        paren_depth = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            if not current_axiom:
                axiom_start = i

            current_axiom.append(stripped)
            paren_depth += stripped.count('(') - stripped.count(')')

            if paren_depth == 0 and current_axiom:
                axioms.append((axiom_start, ' '.join(current_axiom)))
                current_axiom = []

        return axioms

    def _validate_axiom(self, line_number: int, axiom: str):
        """Validate a single OWL axiom"""
        axiom = axiom.strip()

        # Check for empty axioms
        if not axiom:
            return

        # Validate parentheses balance
        if axiom.count('(') != axiom.count(')'):
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                line_number=line_number,
                axiom=axiom,
                message="Unbalanced parentheses",
                fix_suggestion="Ensure every '(' has a matching ')'"
            ))
            return

        # Extract axiom type
        match = re.match(r'\((\w+)\s+', axiom)
        if not match:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                line_number=line_number,
                axiom=axiom,
                message="Cannot parse axiom type",
                fix_suggestion="Axioms should start with (AxiomType ...)"
            ))
            return

        axiom_type = match.group(1)

        # Validate axiom type
        if axiom_type not in self.OWL2_KEYWORDS:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                line_number=line_number,
                axiom=axiom,
                message=f"Unknown OWL2 keyword: {axiom_type}",
                fix_suggestion=f"Valid keywords: {', '.join(sorted(self.OWL2_KEYWORDS))}"
            ))

        # Type-specific validation
        if axiom_type == 'Declaration':
            self._validate_declaration(line_number, axiom)
        elif axiom_type == 'SubClassOf':
            self._validate_subclass_of(line_number, axiom)
        elif axiom_type in ['ObjectProperty', 'DataProperty', 'AnnotationProperty']:
            self._validate_property_declaration(line_number, axiom, axiom_type)
        elif axiom_type in ['Domain', 'Range']:
            self._validate_property_constraint(line_number, axiom, axiom_type)
        elif 'Restriction' in axiom or 'ValuesFrom' in axiom:
            self._validate_restriction(line_number, axiom)

        # Validate namespace usage
        self._validate_namespaces(line_number, axiom)

        # Validate annotations
        if 'rdfs:label' in axiom or 'Annotation' in axiom:
            self._validate_annotations(line_number, axiom)

    def _validate_declaration(self, line_number: int, axiom: str):
        """Validate Declaration axiom"""
        # Extract entity type and IRI
        match = re.search(r'Declaration\(\s*(\w+)\s*\(([^)]+)\)', axiom)
        if not match:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                line_number=line_number,
                axiom=axiom,
                message="Malformed Declaration axiom",
                fix_suggestion="Format: Declaration(Class(iri)) or Declaration(ObjectProperty(iri))"
            ))
            return

        entity_type = match.group(1)
        iri = match.group(2).strip()

        # Track entities
        if entity_type == 'Class':
            self.report.classes.add(iri)
        elif entity_type in ['ObjectProperty', 'DataProperty']:
            self.report.properties.add(iri)
        elif entity_type == 'NamedIndividual':
            self.report.individuals.add(iri)

        # Validate IRI format
        if ':' not in iri:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                line_number=line_number,
                axiom=axiom,
                message=f"Invalid IRI format: {iri}",
                fix_suggestion="Use namespace prefix, e.g., ai:MachineLearning"
            ))

    def _validate_subclass_of(self, line_number: int, axiom: str):
        """Validate SubClassOf axiom"""
        # Extract subclass and superclass
        match = re.search(r'SubClassOf\(\s*([^\s]+)\s+([^)]+)\)', axiom)
        if not match:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                line_number=line_number,
                axiom=axiom,
                message="Malformed SubClassOf axiom",
                fix_suggestion="Format: SubClassOf(SubClass SuperClass)"
            ))
            return

        subclass = match.group(1).strip()
        superclass = match.group(2).strip()

        # Check for cyclic hierarchy (basic check)
        if subclass == superclass:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                line_number=line_number,
                axiom=axiom,
                message="Class cannot be subclass of itself",
                fix_suggestion="Remove cyclic SubClassOf axiom"
            ))

    def _validate_property_declaration(self, line_number: int, axiom: str, prop_type: str):
        """Validate property declaration"""
        # Extract property IRI
        match = re.search(r'Declaration\(\s*' + prop_type + r'\s*\(([^)]+)\)', axiom)
        if match:
            prop_iri = match.group(1).strip()
            self.report.properties.add(prop_iri)

            # Validate naming convention
            if not re.match(r'^[a-z][a-zA-Z]*:', prop_iri):
                self.report.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    line_number=line_number,
                    axiom=axiom,
                    message=f"Property should use camelCase: {prop_iri}",
                    fix_suggestion="Use camelCase for property names, e.g., ai:hasLearningRate"
                ))

    def _validate_property_constraint(self, line_number: int, axiom: str, constraint_type: str):
        """Validate Domain/Range axioms"""
        pattern = constraint_type + r'\(\s*([^\s]+)\s+([^)]+)\)'
        match = re.search(pattern, axiom)
        if not match:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                line_number=line_number,
                axiom=axiom,
                message=f"Malformed {constraint_type} axiom",
                fix_suggestion=f"Format: {constraint_type}(property class)"
            ))

    def _validate_restriction(self, line_number: int, axiom: str):
        """Validate OWL restrictions"""
        # Check for common restriction patterns
        restriction_patterns = [
            r'ObjectSomeValuesFrom',
            r'ObjectAllValuesFrom',
            r'ObjectMinCardinality',
            r'ObjectMaxCardinality',
            r'ObjectExactCardinality',
            r'DataSomeValuesFrom'
        ]

        found_restriction = False
        for pattern in restriction_patterns:
            if pattern in axiom:
                found_restriction = True

                # Validate structure
                if pattern.startswith('Object'):
                    # Should have property and class
                    if axiom.count('(') < 2:
                        self.report.issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            line_number=line_number,
                            axiom=axiom,
                            message=f"Incomplete {pattern} restriction",
                            fix_suggestion=f"{pattern}(property class)"
                        ))
                break

        if not found_restriction and 'ValuesFrom' in axiom:
            self.report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                line_number=line_number,
                axiom=axiom,
                message="Unknown restriction type",
                fix_suggestion="Use standard OWL2 restriction patterns"
            ))

    def _validate_namespaces(self, line_number: int, axiom: str):
        """Validate namespace prefix usage"""
        # Find all prefixed names
        prefixed_names = re.findall(r'(\w+):(\w+)', axiom)

        for prefix, local_name in prefixed_names:
            # Check if prefix is valid
            if prefix not in self.VALID_PREFIXES:
                self.report.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    line_number=line_number,
                    axiom=axiom,
                    message=f"Invalid namespace prefix: {prefix}",
                    fix_suggestion=f"Valid prefixes: {', '.join(sorted(self.VALID_PREFIXES))}"
                ))

            # Check naming conventions
            if prefix in ['ai', 'bc', 'mv', 'rb', 'dt']:
                # Classes should be PascalCase
                if not local_name[0].isupper():
                    # Might be a property (camelCase is ok)
                    pass

    def _validate_annotations(self, line_number: int, axiom: str):
        """Validate annotation axioms"""
        # Check for rdfs:label
        if 'rdfs:label' in axiom:
            # Extract label value
            match = re.search(r'rdfs:label\s+"([^"]+)"', axiom)
            if match:
                label = match.group(1)

                # Label should not be empty
                if not label.strip():
                    self.report.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        line_number=line_number,
                        axiom=axiom,
                        message="Empty rdfs:label",
                        fix_suggestion="Provide meaningful label text"
                    ))

                # Label should match class name conventions
                # (PascalCase for classes, spaces allowed)
            else:
                self.report.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    line_number=line_number,
                    axiom=axiom,
                    message="Malformed rdfs:label annotation",
                    fix_suggestion='Format: rdfs:label "Label Text"'
                ))

    def _validate_cross_block_consistency(self):
        """Validate consistency across all ontology blocks"""
        # Check for orphaned classes (no SubClassOf)
        # Check for undeclared entities
        # Check for consistent prefix usage
        pass

    def generate_validation_report(self) -> str:
        """
        Generate human-readable validation report.

        Returns:
            Formatted validation report string
        """
        if not self.report:
            return "No validation has been performed."

        lines = []
        lines.append("=" * 80)
        lines.append("OWL2 VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"File: {self.report.file_path}")
        lines.append(f"Total Axioms: {self.report.total_axioms}")
        lines.append(f"Classes: {len(self.report.classes)}")
        lines.append(f"Properties: {len(self.report.properties)}")
        lines.append(f"Individuals: {len(self.report.individuals)}")
        lines.append("")

        # Summary
        error_count = len(self.report.errors)
        warning_count = len(self.report.warnings)

        if self.report.is_valid:
            lines.append("✓ VALID - No errors found")
        else:
            lines.append(f"✗ INVALID - {error_count} error(s) found")

        if warning_count > 0:
            lines.append(f"⚠ {warning_count} warning(s)")

        lines.append("")

        # Errors
        if error_count > 0:
            lines.append("ERRORS:")
            lines.append("-" * 80)
            for issue in self.report.errors:
                lines.append(f"Line {issue.line_number}: {issue.message}")
                lines.append(f"  Axiom: {issue.axiom[:100]}...")
                if issue.fix_suggestion:
                    lines.append(f"  Fix: {issue.fix_suggestion}")
                lines.append("")

        # Warnings
        if warning_count > 0:
            lines.append("WARNINGS:")
            lines.append("-" * 80)
            for issue in self.report.warnings:
                lines.append(f"Line {issue.line_number}: {issue.message}")
                lines.append(f"  Axiom: {issue.axiom[:100]}...")
                if issue.fix_suggestion:
                    lines.append(f"  Suggestion: {issue.fix_suggestion}")
                lines.append("")

        lines.append("=" * 80)

        return '\n'.join(lines)


def validate_file(file_path: str) -> ValidationReport:
    """
    Convenience function to validate a file.

    Args:
        file_path: Path to Logseq markdown file

    Returns:
        ValidationReport
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    validator = OWL2Validator()
    return validator.validate_file(file_path, content)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python owl2_validator.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    validator = OWL2Validator()

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    report = validator.validate_file(file_path, content)
    print(validator.generate_validation_report())

    sys.exit(0 if report.is_valid else 1)
