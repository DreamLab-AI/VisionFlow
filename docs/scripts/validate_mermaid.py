#!/usr/bin/env python3
"""
Mermaid Diagram Validation Script
Validates all Mermaid diagrams in documentation for syntax, quality, and consistency
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

class MermaidValidator:
    def __init__(self, docs_root: str):
        self.docs_root = Path(docs_root)
        self.results = {
            'total_files': 0,
            'files_with_diagrams': 0,
            'total_diagrams': 0,
            'syntax_errors': [],
            'warnings': [],
            'diagram_types': defaultdict(int),
            'files': []
        }

    def validate_all(self):
        """Scan all markdown files and validate Mermaid diagrams"""
        for md_file in self.docs_root.rglob('*.md'):
            self.results['total_files'] += 1
            self.validate_file(md_file)

        return self.results

    def validate_file(self, filepath: Path):
        """Validate all Mermaid diagrams in a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.results['syntax_errors'].append({
                'file': str(filepath.relative_to(self.docs_root)),
                'error': f'Cannot read file: {e}'
            })
            return

        # Find all Mermaid code blocks
        pattern = r'```mermaid\n(.*?)```'
        matches = list(re.finditer(pattern, content, re.DOTALL))

        if not matches:
            return

        self.results['files_with_diagrams'] += 1
        rel_path = str(filepath.relative_to(self.docs_root))

        file_info = {
            'path': rel_path,
            'diagrams': len(matches),
            'issues': []
        }

        for idx, match in enumerate(matches, 1):
            diagram = match.group(1)
            self.results['total_diagrams'] += 1

            # Validate diagram
            issues = self.validate_diagram(diagram, rel_path, idx)
            if issues:
                file_info['issues'].extend(issues)

        self.results['files'].append(file_info)

    def validate_diagram(self, diagram: str, filepath: str, diagram_num: int) -> List[Dict]:
        """Validate a single Mermaid diagram"""
        issues = []
        diagram_stripped = diagram.strip()

        # Check 1: Empty diagram
        if not diagram_stripped:
            issues.append({
                'file': filepath,
                'diagram': diagram_num,
                'severity': 'error',
                'message': 'Empty Mermaid diagram'
            })
            return issues

        # Check 2: Identify diagram type
        diagram_type = self.identify_diagram_type(diagram_stripped)
        if not diagram_type:
            issues.append({
                'file': filepath,
                'diagram': diagram_num,
                'severity': 'error',
                'message': 'Unknown or invalid diagram type'
            })
        else:
            self.results['diagram_types'][diagram_type] += 1

            # Check 3: Type-specific validation
            type_issues = self.validate_diagram_type(diagram_stripped, diagram_type, filepath, diagram_num)
            issues.extend(type_issues)

        # Check 4: Common syntax issues
        syntax_issues = self.check_syntax(diagram_stripped, filepath, diagram_num)
        issues.extend(syntax_issues)

        # Check 5: Styling consistency
        style_issues = self.check_styling(diagram_stripped, filepath, diagram_num)
        issues.extend(style_issues)

        return issues

    def identify_diagram_type(self, diagram: str) -> str:
        """Identify the type of Mermaid diagram"""
        first_line = diagram.split('\n')[0].strip()

        type_patterns = {
            'graph': r'^graph\s+(TB|BT|LR|RL|TD)',
            'flowchart': r'^flowchart\s+(TB|BT|LR|RL|TD)',
            'sequenceDiagram': r'^sequenceDiagram',
            'classDiagram': r'^classDiagram',
            'stateDiagram': r'^stateDiagram(-v2)?',
            'erDiagram': r'^erDiagram',
            'gantt': r'^gantt',
            'pie': r'^pie',
            'gitGraph': r'^gitGraph',
            'mindmap': r'^mindmap',
            'timeline': r'^timeline',
            'quadrantChart': r'^quadrantChart',
            'requirementDiagram': r'^requirementDiagram',
            'C4Context': r'^C4Context'
        }

        for diagram_type, pattern in type_patterns.items():
            if re.match(pattern, first_line):
                return diagram_type

        return None

    def validate_diagram_type(self, diagram: str, diagram_type: str, filepath: str, diagram_num: int) -> List[Dict]:
        """Validate type-specific syntax"""
        issues = []

        if diagram_type in ['graph', 'flowchart']:
            # Check for balanced brackets
            if diagram.count('[') != diagram.count(']'):
                issues.append({
                    'file': filepath,
                    'diagram': diagram_num,
                    'severity': 'error',
                    'message': 'Unbalanced square brackets in graph'
                })
            if diagram.count('(') != diagram.count(')'):
                issues.append({
                    'file': filepath,
                    'diagram': diagram_num,
                    'severity': 'error',
                    'message': 'Unbalanced parentheses in graph'
                })
            if diagram.count('{') != diagram.count('}'):
                issues.append({
                    'file': filepath,
                    'diagram': diagram_num,
                    'severity': 'error',
                    'message': 'Unbalanced curly braces in graph'
                })

        elif diagram_type == 'sequenceDiagram':
            # Check for participant definitions
            if not re.search(r'participant\s+\w+', diagram):
                issues.append({
                    'file': filepath,
                    'diagram': diagram_num,
                    'severity': 'warning',
                    'message': 'No participants defined in sequence diagram'
                })

        elif diagram_type == 'gantt':
            # Check for required sections
            if 'section' not in diagram.lower():
                issues.append({
                    'file': filepath,
                    'diagram': diagram_num,
                    'severity': 'warning',
                    'message': 'No sections defined in Gantt chart'
                })

        return issues

    def check_syntax(self, diagram: str, filepath: str, diagram_num: int) -> List[Dict]:
        """Check for common syntax errors"""
        issues = []

        # Check for unclosed strings
        lines = diagram.split('\n')
        for line_num, line in enumerate(lines, 1):
            # Count quotes (simple check)
            double_quotes = line.count('"')
            if double_quotes % 2 != 0:
                issues.append({
                    'file': filepath,
                    'diagram': diagram_num,
                    'severity': 'warning',
                    'message': f'Unclosed quotes on line {line_num}'
                })

        return issues

    def check_styling(self, diagram: str, filepath: str, diagram_num: int) -> List[Dict]:
        """Check styling consistency"""
        issues = []

        # Check for style definitions
        has_style = 'style ' in diagram or 'classDef' in diagram

        # Recommended colors from documentation
        recommended_colors = ['#FF6B35', '#4ECDC4', '#95E1D3', '#F7DC6F', '#BB8FCE', '#f96', '#FF9800', '#2196F3', '#4CAF50']

        # Extract used colors
        color_pattern = r'fill:(#[0-9A-Fa-f]{3,6})'
        used_colors = re.findall(color_pattern, diagram)

        # Check if non-standard colors are used
        non_standard_colors = [c for c in used_colors if c.upper() not in [rc.upper() for rc in recommended_colors]]
        if non_standard_colors and len(non_standard_colors) > 2:
            issues.append({
                'file': filepath,
                'diagram': diagram_num,
                'severity': 'info',
                'message': f'Using non-standard colors: {", ".join(set(non_standard_colors))}'
            })

        return issues

    def generate_report(self) -> str:
        """Generate validation report"""
        report = []
        report.append("=" * 80)
        report.append("MERMAID DIAGRAM VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary
        report.append("## SUMMARY")
        report.append(f"Total files scanned: {self.results['total_files']}")
        report.append(f"Files with Mermaid diagrams: {self.results['files_with_diagrams']}")
        report.append(f"Total diagrams: {self.results['total_diagrams']}")
        report.append("")

        # Diagram types breakdown
        report.append("## DIAGRAM TYPES")
        for dtype, count in sorted(self.results['diagram_types'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  - {dtype}: {count}")
        report.append("")

        # Issues summary
        critical_errors = sum(1 for f in self.results['files'] for i in f['issues'] if i['severity'] == 'error')
        warnings = sum(1 for f in self.results['files'] for i in f['issues'] if i['severity'] == 'warning')
        info = sum(1 for f in self.results['files'] for i in f['issues'] if i['severity'] == 'info')

        report.append("## ISSUES SUMMARY")
        report.append(f"Critical errors: {critical_errors}")
        report.append(f"Warnings: {warnings}")
        report.append(f"Info: {info}")
        report.append("")

        # Detailed issues
        if any(f['issues'] for f in self.results['files']):
            report.append("## DETAILED ISSUES")
            report.append("")

            for file_info in self.results['files']:
                if file_info['issues']:
                    report.append(f"### {file_info['path']}")
                    report.append(f"Diagrams: {file_info['diagrams']}")
                    report.append("")

                    for issue in file_info['issues']:
                        severity_marker = {
                            'error': '🔴 ERROR',
                            'warning': '🟡 WARNING',
                            'info': 'ℹ️  INFO'
                        }.get(issue['severity'], '❓ UNKNOWN')

                        report.append(f"  {severity_marker} [Diagram #{issue['diagram']}]: {issue['message']}")

                    report.append("")
        else:
            report.append("## ✅ NO ISSUES FOUND")
            report.append("")

        # Quality score
        total_checks = self.results['total_diagrams'] * 5  # 5 checks per diagram
        issues_count = critical_errors + warnings
        quality_score = max(0, (total_checks - issues_count) / total_checks * 100) if total_checks > 0 else 100

        report.append("## QUALITY METRICS")
        report.append(f"Quality Score: {quality_score:.1f}%")
        report.append(f"Diagrams without issues: {self.results['total_diagrams'] - len([f for f in self.results['files'] if f['issues']])}")
        report.append("")

        # Files processed
        report.append("## FILES PROCESSED")
        for file_info in sorted(self.results['files'], key=lambda x: x['diagrams'], reverse=True)[:20]:
            issue_count = len(file_info['issues'])
            status = "✅" if issue_count == 0 else f"⚠️  ({issue_count} issues)"
            report.append(f"  {status} {file_info['path']} - {file_info['diagrams']} diagram(s)")

        if len(self.results['files']) > 20:
            report.append(f"  ... and {len(self.results['files']) - 20} more files")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

def main():
    docs_root = Path(__file__).parent.parent
    validator = MermaidValidator(docs_root)

    print("Scanning documentation for Mermaid diagrams...")
    validator.validate_all()

    report = validator.generate_report()
    print(report)

    # Save report
    report_path = docs_root / 'scripts' / 'mermaid_validation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
