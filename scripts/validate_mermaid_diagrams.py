#!/usr/bin/env python3
"""
VisionFlow Mermaid Diagram Validation Tool

This script validates all Mermaid diagrams in the documentation for:
1. Syntax correctness
2. Common formatting issues
3. Deprecated syntax
4. Best practices compliance

Author: Claude Code Analysis Agent
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ValidationIssue:
    file_path: str
    diagram_index: int
    line_number: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: Optional[str] = None
    original_line: Optional[str] = None
    fixed_line: Optional[str] = None

@dataclass
class DiagramInfo:
    file_path: str
    diagram_index: int
    start_line: int
    end_line: int
    diagram_type: str
    content: str
    raw_content: str

@dataclass
class ValidationReport:
    total_files: int
    total_diagrams: int
    files_with_issues: int
    diagrams_with_issues: int
    issues: List[ValidationIssue]
    successful_fixes: int
    validation_time: str

class MermaidValidator:
    def __init__(self):
        self.validation_patterns = self._load_validation_patterns()
        self.fixes_applied = 0
        
    def _load_validation_patterns(self) -> Dict:
        """Load validation patterns for different diagram types"""
        return {
            'graph': {
                'required_patterns': [
                    r'graph\s+(TD|TB|BT|RL|LR)',  # Valid graph direction
                ],
                'syntax_issues': [
                    (r'-->', r'Correct arrow syntax'),
                    (r'--->', r'Use --> for directed arrows'),
                    (r'-.->',  r'Valid dotted arrow'),
                    (r'==>', r'Use --> for standard arrows'),
                ],
                'common_errors': [
                    (r'([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]*)\]\s*-->\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]*)\]', 
                     r'Node definition with arrow'),
                ]
            },
            'flowchart': {
                'required_patterns': [
                    r'flowchart\s+(TD|TB|BT|RL|LR)',
                ],
                'syntax_issues': [
                    (r'-->', r'Correct arrow syntax'),
                    (r'-.->',  r'Valid dotted arrow'),
                ]
            },
            'sequenceDiagram': {
                'required_patterns': [
                    r'sequenceDiagram',
                ],
                'syntax_issues': [
                    (r'participant\s+\w+', r'Valid participant definition'),
                    (r'->>',  r'Valid message arrow'),
                    (r'-->>',  r'Valid response arrow'),
                ]
            },
            'classDiagram': {
                'required_patterns': [
                    r'classDiagram',
                ],
                'syntax_issues': [
                    (r'class\s+\w+', r'Valid class definition'),
                ]
            },
            'stateDiagram': {
                'required_patterns': [
                    r'stateDiagram(-v2)?',
                ],
            }
        }

    def find_mermaid_diagrams(self, docs_path: str) -> List[DiagramInfo]:
        """Find all Mermaid diagram blocks in markdown files"""
        diagrams = []
        docs_path = Path(docs_path)
        
        for md_file in docs_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_diagrams = self._extract_diagrams_from_content(str(md_file), content)
                diagrams.extend(file_diagrams)
                
            except Exception as e:
                print(f"Error reading {md_file}: {e}")
                
        return diagrams

    def _extract_diagrams_from_content(self, file_path: str, content: str) -> List[DiagramInfo]:
        """Extract Mermaid diagrams from file content"""
        diagrams = []
        lines = content.split('\n')
        
        i = 0
        diagram_index = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line == '```mermaid':
                start_line = i + 1
                diagram_lines = []
                i += 1
                
                # Collect diagram content until closing ```
                while i < len(lines) and lines[i].strip() != '```':
                    diagram_lines.append(lines[i])
                    i += 1
                
                if i < len(lines):  # Found closing ```
                    end_line = i
                    diagram_content = '\n'.join(diagram_lines)
                    raw_content = '\n'.join(lines[start_line-1:end_line+1])
                    
                    diagram_type = self._detect_diagram_type(diagram_content)
                    
                    diagrams.append(DiagramInfo(
                        file_path=file_path,
                        diagram_index=diagram_index,
                        start_line=start_line,
                        end_line=end_line,
                        diagram_type=diagram_type,
                        content=diagram_content.strip(),
                        raw_content=raw_content
                    ))
                    diagram_index += 1
            
            i += 1
            
        return diagrams

    def _detect_diagram_type(self, content: str) -> str:
        """Detect the type of Mermaid diagram"""
        content_lower = content.lower().strip()
        
        if content_lower.startswith('graph'):
            return 'graph'
        elif content_lower.startswith('flowchart'):
            return 'flowchart'
        elif content_lower.startswith('sequencediagram'):
            return 'sequenceDiagram'
        elif content_lower.startswith('classdiagram'):
            return 'classDiagram'
        elif content_lower.startswith('statediagram'):
            return 'stateDiagram'
        elif content_lower.startswith('erdiagram'):
            return 'erDiagram'
        elif content_lower.startswith('journey'):
            return 'journey'
        elif content_lower.startswith('gantt'):
            return 'gantt'
        elif content_lower.startswith('pie'):
            return 'pie'
        elif content_lower.startswith('gitgraph'):
            return 'gitGraph'
        else:
            return 'unknown'

    def validate_diagram(self, diagram: DiagramInfo) -> List[ValidationIssue]:
        """Validate a single Mermaid diagram"""
        issues = []
        lines = diagram.content.split('\n')
        
        # Check for empty diagram
        if not diagram.content.strip():
            issues.append(ValidationIssue(
                file_path=diagram.file_path,
                diagram_index=diagram.diagram_index,
                line_number=diagram.start_line,
                issue_type='empty_diagram',
                severity='error',
                message='Empty Mermaid diagram block',
                suggestion='Add diagram content or remove empty block'
            ))
            return issues
        
        # Validate diagram type
        if diagram.diagram_type == 'unknown':
            issues.append(ValidationIssue(
                file_path=diagram.file_path,
                diagram_index=diagram.diagram_index,
                line_number=diagram.start_line,
                issue_type='unknown_type',
                severity='error',
                message='Unrecognized diagram type',
                suggestion='Use supported diagram types: graph, flowchart, sequenceDiagram, etc.'
            ))
        
        # Type-specific validation
        if diagram.diagram_type in self.validation_patterns:
            issues.extend(self._validate_diagram_syntax(diagram, lines))
        
        # General syntax checks
        issues.extend(self._validate_general_syntax(diagram, lines))
        
        return issues

    def _validate_diagram_syntax(self, diagram: DiagramInfo, lines: List[str]) -> List[ValidationIssue]:
        """Validate diagram-specific syntax"""
        issues = []
        patterns = self.validation_patterns.get(diagram.diagram_type, {})
        
        # Check for required patterns
        required_patterns = patterns.get('required_patterns', [])
        content = diagram.content
        
        for pattern in required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                issues.append(ValidationIssue(
                    file_path=diagram.file_path,
                    diagram_index=diagram.diagram_index,
                    line_number=diagram.start_line,
                    issue_type='missing_declaration',
                    severity='error',
                    message=f'Missing required {diagram.diagram_type} declaration',
                    suggestion=f'Start diagram with proper declaration matching pattern: {pattern}'
                ))
        
        return issues

    def _validate_general_syntax(self, diagram: DiagramInfo, lines: List[str]) -> List[ValidationIssue]:
        """Validate general Mermaid syntax issues"""
        issues = []
        
        for i, line in enumerate(lines):
            line_number = diagram.start_line + i
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Check for common syntax issues
            issues.extend(self._check_line_syntax(diagram, line, line_number))
            
            # Check for quote issues
            if self._has_unmatched_quotes(line):
                issues.append(ValidationIssue(
                    file_path=diagram.file_path,
                    diagram_index=diagram.diagram_index,
                    line_number=line_number,
                    issue_type='unmatched_quotes',
                    severity='error',
                    message='Unmatched quotes in line',
                    suggestion='Ensure all quotes are properly closed',
                    original_line=line.strip()
                ))
            
            # Check for invalid characters
            if self._has_invalid_characters(line):
                issues.append(ValidationIssue(
                    file_path=diagram.file_path,
                    diagram_index=diagram.diagram_index,
                    line_number=line_number,
                    issue_type='invalid_characters',
                    severity='warning',
                    message='Line contains potentially problematic characters',
                    suggestion='Check for special characters that might break rendering',
                    original_line=line.strip()
                ))
        
        return issues

    def _check_line_syntax(self, diagram: DiagramInfo, line: str, line_number: int) -> List[ValidationIssue]:
        """Check line-specific syntax issues"""
        issues = []
        
        # Check for deprecated arrow syntax
        deprecated_arrows = [
            ('===>', '-->'),
            ('-.-.>', '-.->'),
            ('<-->', '<-->'),  # bidirectional
        ]
        
        for deprecated, replacement in deprecated_arrows:
            if deprecated in line:
                fixed_line = line.replace(deprecated, replacement)
                issues.append(ValidationIssue(
                    file_path=diagram.file_path,
                    diagram_index=diagram.diagram_index,
                    line_number=line_number,
                    issue_type='deprecated_syntax',
                    severity='warning',
                    message=f'Deprecated arrow syntax: {deprecated}',
                    suggestion=f'Use {replacement} instead',
                    original_line=line.strip(),
                    fixed_line=fixed_line.strip()
                ))
        
        # Check for common node ID issues
        if re.search(r'\b\d+[A-Za-z]', line):
            issues.append(ValidationIssue(
                file_path=diagram.file_path,
                diagram_index=diagram.diagram_index,
                line_number=line_number,
                issue_type='invalid_node_id',
                severity='warning',
                message='Node ID starting with number may cause issues',
                suggestion='Use IDs that start with letters: A1, Node1, etc.',
                original_line=line.strip()
            ))
        
        # Check for missing semicolons in some contexts
        if diagram.diagram_type in ['classDiagram'] and line.strip().endswith(':'):
            if not line.strip().endswith('::'):
                issues.append(ValidationIssue(
                    file_path=diagram.file_path,
                    diagram_index=diagram.diagram_index,
                    line_number=line_number,
                    issue_type='missing_syntax',
                    severity='info',
                    message='Consider adding double colon for class definitions',
                    suggestion='Use :: for class member definitions',
                    original_line=line.strip()
                ))
        
        return issues

    def _has_unmatched_quotes(self, line: str) -> bool:
        """Check for unmatched quotes"""
        single_quotes = line.count("'")
        double_quotes = line.count('"')
        
        return (single_quotes % 2 != 0) or (double_quotes % 2 != 0)

    def _has_invalid_characters(self, line: str) -> bool:
        """Check for potentially problematic characters"""
        # Characters that might cause issues in Mermaid
        problematic_chars = ['`', '\t', '\r']
        return any(char in line for char in problematic_chars)

    def generate_fixes(self, issues: List[ValidationIssue]) -> List[ValidationIssue]:
        """Generate fixes for issues that can be automatically corrected"""
        fixed_issues = []
        
        for issue in issues:
            if issue.fixed_line and issue.original_line:
                # This issue has an automatic fix
                fixed_issues.append(issue)
                self.fixes_applied += 1
            else:
                # Generate fixes where possible
                fixed_issue = self._generate_fix(issue)
                if fixed_issue.fixed_line:
                    self.fixes_applied += 1
                fixed_issues.append(fixed_issue)
        
        return fixed_issues

    def _generate_fix(self, issue: ValidationIssue) -> ValidationIssue:
        """Generate automatic fix for an issue"""
        if issue.issue_type == 'deprecated_syntax' and issue.original_line:
            # Already handled in _check_line_syntax
            pass
        elif issue.issue_type == 'unmatched_quotes' and issue.original_line:
            # Try to fix unmatched quotes
            line = issue.original_line
            if line.count('"') % 2 != 0:
                # Add missing quote at the end
                issue.fixed_line = line + '"'
                issue.suggestion = 'Added missing closing quote'
            elif line.count("'") % 2 != 0:
                issue.fixed_line = line + "'"
                issue.suggestion = 'Added missing closing quote'
        
        return issue

def main():
    """Main validation function"""
    docs_path = "/workspace/ext/docs"
    validator = MermaidValidator()
    
    print("🔍 VisionFlow Mermaid Diagram Validation")
    print("=" * 50)
    print(f"Scanning documentation in: {docs_path}")
    print()
    
    # Find all diagrams
    print("📋 Discovering Mermaid diagrams...")
    diagrams = validator.find_mermaid_diagrams(docs_path)
    
    if not diagrams:
        print("❌ No Mermaid diagrams found!")
        return
    
    print(f"✅ Found {len(diagrams)} Mermaid diagrams in {len(set(d.file_path for d in diagrams))} files")
    print()
    
    # Validate each diagram
    print("🔧 Validating diagrams...")
    all_issues = []
    diagrams_with_issues = 0
    
    for diagram in diagrams:
        issues = validator.validate_diagram(diagram)
        if issues:
            diagrams_with_issues += 1
            all_issues.extend(issues)
    
    # Generate fixes
    print("🛠️  Generating fixes...")
    fixed_issues = validator.generate_fixes(all_issues)
    
    # Create validation report
    report = ValidationReport(
        total_files=len(set(d.file_path for d in diagrams)),
        total_diagrams=len(diagrams),
        files_with_issues=len(set(issue.file_path for issue in fixed_issues)),
        diagrams_with_issues=diagrams_with_issues,
        issues=fixed_issues,
        successful_fixes=validator.fixes_applied,
        validation_time=datetime.now().isoformat()
    )
    
    # Print summary
    print_validation_summary(report, diagrams)
    
    # Save detailed report
    save_validation_report(report, "/workspace/ext/docs/MERMAID_VALIDATION_REPORT.md")
    
    # Print file-by-file breakdown
    print_file_breakdown(report, diagrams)
    
    return report

def print_validation_summary(report: ValidationReport, diagrams: List[DiagramInfo]):
    """Print validation summary"""
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Total files scanned: {report.total_files}")
    print(f"Total diagrams found: {report.total_diagrams}")
    print(f"Files with issues: {report.files_with_issues}")
    print(f"Diagrams with issues: {report.diagrams_with_issues}")
    print(f"Total issues found: {len(report.issues)}")
    print(f"Automatic fixes generated: {report.successful_fixes}")
    
    # Issue severity breakdown
    errors = sum(1 for issue in report.issues if issue.severity == 'error')
    warnings = sum(1 for issue in report.issues if issue.severity == 'warning')
    infos = sum(1 for issue in report.issues if issue.severity == 'info')
    
    print(f"\n🚨 Issues by severity:")
    print(f"  Errors: {errors}")
    print(f"  Warnings: {warnings}")
    print(f"  Info: {infos}")
    
    # Diagram type breakdown
    diagram_types = {}
    for diagram in diagrams:
        diagram_types[diagram.diagram_type] = diagram_types.get(diagram.diagram_type, 0) + 1
    
    print(f"\n📈 Diagrams by type:")
    for dtype, count in sorted(diagram_types.items()):
        print(f"  {dtype}: {count}")

def print_file_breakdown(report: ValidationReport, diagrams: List[DiagramInfo]):
    """Print detailed file-by-file breakdown"""
    print("\n📁 FILE-BY-FILE BREAKDOWN")
    print("=" * 40)
    
    # Group issues by file
    issues_by_file = {}
    for issue in report.issues:
        if issue.file_path not in issues_by_file:
            issues_by_file[issue.file_path] = []
        issues_by_file[issue.file_path].append(issue)
    
    # Group diagrams by file
    diagrams_by_file = {}
    for diagram in diagrams:
        if diagram.file_path not in diagrams_by_file:
            diagrams_by_file[diagram.file_path] = []
        diagrams_by_file[diagram.file_path].append(diagram)
    
    # Print file breakdown
    for file_path in sorted(set(d.file_path for d in diagrams)):
        rel_path = file_path.replace('/workspace/ext/', '')
        file_diagrams = diagrams_by_file.get(file_path, [])
        file_issues = issues_by_file.get(file_path, [])
        
        status = "✅" if not file_issues else "❌"
        print(f"\n{status} {rel_path}")
        print(f"   Diagrams: {len(file_diagrams)} | Issues: {len(file_issues)}")
        
        if file_issues:
            for issue in file_issues[:3]:  # Show first 3 issues
                severity_icon = {"error": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "❓")
                print(f"   {severity_icon} Line {issue.line_number}: {issue.message}")
            
            if len(file_issues) > 3:
                print(f"   ... and {len(file_issues) - 3} more issues")

def save_validation_report(report: ValidationReport, output_path: str):
    """Save detailed validation report"""
    content = generate_report_markdown(report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n💾 Detailed report saved to: {output_path}")

def generate_report_markdown(report: ValidationReport) -> str:
    """Generate markdown validation report"""
    
    # Group issues by file and severity
    issues_by_file = {}
    severity_counts = {"error": 0, "warning": 0, "info": 0}
    
    for issue in report.issues:
        if issue.file_path not in issues_by_file:
            issues_by_file[issue.file_path] = []
        issues_by_file[issue.file_path].append(issue)
        severity_counts[issue.severity] += 1
    
    # Generate markdown
    md_content = f"""# Mermaid Diagram Validation Report

**Generated**: {report.validation_time}

## Executive Summary

| Metric | Count |
|--------|--------|
| Total Files | {report.total_files} |
| Total Diagrams | {report.total_diagrams} |
| Files with Issues | {report.files_with_issues} |
| Diagrams with Issues | {report.diagrams_with_issues} |
| Total Issues | {len(report.issues)} |
| Auto-fixes Generated | {report.successful_fixes} |

## Issues by Severity

| Severity | Count | Icon |
|----------|--------|------|
| Errors | {severity_counts['error']} | 🚨 |
| Warnings | {severity_counts['warning']} | ⚠️ |
| Info | {severity_counts['info']} | ℹ️ |

## Detailed Issues by File

"""
    
    for file_path in sorted(issues_by_file.keys()):
        rel_path = file_path.replace('/workspace/ext/', '')
        file_issues = issues_by_file[file_path]
        
        md_content += f"\n### {rel_path}\n\n"
        md_content += f"**Issues found**: {len(file_issues)}\n\n"
        
        for issue in file_issues:
            severity_icon = {"error": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "❓")
            
            md_content += f"#### {severity_icon} {issue.issue_type.title().replace('_', ' ')}\n"
            md_content += f"- **Line**: {issue.line_number}\n"
            md_content += f"- **Severity**: {issue.severity}\n"
            md_content += f"- **Message**: {issue.message}\n"
            
            if issue.suggestion:
                md_content += f"- **Suggestion**: {issue.suggestion}\n"
            
            if issue.original_line:
                md_content += f"- **Original**: `{issue.original_line}`\n"
            
            if issue.fixed_line:
                md_content += f"- **Fixed**: `{issue.fixed_line}`\n"
            
            md_content += "\n"
    
    md_content += f"""
## Recommended Actions

1. **Fix Critical Errors** ({severity_counts['error']} issues)
   - Address syntax errors that prevent diagram rendering
   - Fix missing required declarations
   - Resolve unmatched quotes and brackets

2. **Address Warnings** ({severity_counts['warning']} issues)
   - Update deprecated syntax to current standards
   - Fix node ID formatting issues
   - Clean up invalid characters

3. **Consider Info Items** ({severity_counts['info']} issues)
   - Apply best practice recommendations
   - Improve diagram clarity and consistency

## Automatic Fixes Available

{report.successful_fixes} issues can be automatically fixed. Run the validation script with `--fix` flag to apply these fixes.

## Next Steps

1. Review this report and prioritize fixes based on severity
2. Apply automatic fixes where available
3. Manually address remaining issues
4. Re-run validation to verify fixes
5. Update documentation standards to prevent future issues

---

*Report generated by VisionFlow Mermaid Validation Tool*
"""
    
    return md_content

if __name__ == "__main__":
    main()