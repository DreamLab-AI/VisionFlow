#!/usr/bin/env python3
"""
Documentation Link Analyzer
Comprehensive analysis of all markdown links in /docs/ directory
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class LinkAnalyzer:
    def __init__(self, docs_dir: str, project_root: str):
        self.docs_dir = Path(docs_dir)
        self.project_root = Path(project_root)
        self.md_files: List[Path] = []
        self.links: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

        # Categories
        self.critical_issues: List[str] = []
        self.high_issues: List[str] = []
        self.medium_issues: List[str] = []
        self.low_issues: List[str] = []

        # Statistics
        self.stats = {
            'total_files': 0,
            'total_links': 0,
            'broken_links': 0,
            'valid_links': 0,
            'external_links': 0,
            'internal_doc_links': 0,
            'code_references': 0,
            'incorrect_paths': 0
        }

    def find_markdown_files(self):
        """Find all markdown files in docs directory"""
        self.md_files = sorted(self.docs_dir.rglob('*.md'))
        self.stats['total_files'] = len(self.md_files)
        print(f"Found {self.stats['total_files']} markdown files")

    def extract_links(self):
        """Extract all links from markdown files"""
        # Pattern to match markdown links: [text](path)
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        for md_file in self.md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        for match in link_pattern.finditer(line):
                            text = match.group(1)
                            path = match.group(2)
                            self.links[str(md_file)].append((line_num, text, path))
                            self.stats['total_links'] += 1
            except Exception as e:
                print(f"Error reading {md_file}: {e}")

        print(f"Extracted {self.stats['total_links']} links")

    def resolve_path(self, current_file: Path, link_path: str) -> Tuple[Path, bool]:
        """Resolve a link path relative to current file"""
        current_dir = current_file.parent

        # External URL
        if link_path.startswith(('http://', 'https://', 'ftp://')):
            self.stats['external_links'] += 1
            return None, True

        # Anchor link
        if link_path.startswith('#'):
            return None, True

        # Remove anchor from path
        clean_path = link_path.split('#')[0]
        if not clean_path:
            return None, True

        try:
            # Absolute path
            if clean_path.startswith('/'):
                target = self.project_root / clean_path.lstrip('/')
            else:
                # Relative path
                target = current_dir / clean_path

            # Normalize path
            target = target.resolve()
            return target, target.exists()
        except Exception as e:
            print(f"Error resolving path {link_path}: {e}")
            return None, False

    def categorize_issue(self, file: Path, line_num: int, text: str,
                        link_path: str, target: Path, exists: bool):
        """Categorize broken links by severity"""
        rel_file = file.relative_to(self.project_root)

        if not exists:
            self.stats['broken_links'] += 1

            # CRITICAL: Broken documentation link
            if link_path.endswith('.md'):
                self.stats['internal_doc_links'] += 1
                self.critical_issues.append(
                    f"**CRITICAL**: Broken doc link in `{rel_file}:{line_num}`\n"
                    f"  - Link text: \"{text}\"\n"
                    f"  - Path: `{link_path}`\n"
                    f"  - Resolved to: `{target}`\n"
                )

            # HIGH: Incorrect code path reference
            elif '/project/src/' in link_path or '/project/client/' in link_path or '/project/multi-agent-docker/' in link_path:
                self.stats['incorrect_paths'] += 1
                self.high_issues.append(
                    f"**HIGH**: Incorrect code path in `{rel_file}:{line_num}`\n"
                    f"  - Link text: \"{text}\"\n"
                    f"  - Path: `{link_path}`\n"
                    f"  - Issue: Should not contain '/project/' prefix\n"
                    f"  - Suggestion: Use relative path from docs root\n"
                )

            # HIGH: Broken code reference
            elif any(ext in link_path for ext in ['.rs', '.ts', '.js', '.py', '.toml', '.json']):
                self.stats['code_references'] += 1
                self.high_issues.append(
                    f"**HIGH**: Broken code reference in `{rel_file}:{line_num}`\n"
                    f"  - Link text: \"{text}\"\n"
                    f"  - Path: `{link_path}`\n"
                    f"  - Target: `{target}`\n"
                )

            # MEDIUM: Other broken links
            else:
                self.medium_issues.append(
                    f"**MEDIUM**: Broken link in `{rel_file}:{line_num}`\n"
                    f"  - Link text: \"{text}\"\n"
                    f"  - Path: `{link_path}`\n"
                    f"  - Target: `{target}`\n"
                )
        else:
            self.stats['valid_links'] += 1

    def analyze_links(self):
        """Analyze all extracted links"""
        for file_path, links in self.links.items():
            file = Path(file_path)
            for line_num, text, link_path in links:
                target, exists = self.resolve_path(file, link_path)

                if target is not None:
                    self.categorize_issue(file, line_num, text, link_path, target, exists)

    def check_path_patterns(self):
        """Check for common path pattern issues"""
        print("\nChecking for common path pattern issues...")

        for file in self.md_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    rel_file = file.relative_to(self.project_root)

                    # Check for /project/ prefix in paths
                    if '/project/src/' in content or '/project/client/' in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if '/project/' in line and '](' in line:
                                self.low_issues.append(
                                    f"**LOW**: Found '/project/' prefix in `{rel_file}:{i}`\n"
                                    f"  - Line: `{line.strip()}`\n"
                                    f"  - Suggestion: Use relative paths without '/project/'\n"
                                )
            except Exception as e:
                print(f"Error checking patterns in {file}: {e}")

    def generate_report(self) -> str:
        """Generate comprehensive report"""
        report = []
        report.append("# Documentation Link Analysis Report")
        report.append(f"Generated: {os.popen('date').read().strip()}")
        report.append("")

        # Summary Statistics
        report.append("## 📊 Summary Statistics")
        report.append("")
        report.append(f"- **Total markdown files**: {self.stats['total_files']}")
        report.append(f"- **Total links found**: {self.stats['total_links']}")
        report.append(f"- **Valid links**: {self.stats['valid_links']}")
        report.append(f"- **Broken links**: {self.stats['broken_links']}")
        report.append(f"- **External URLs**: {self.stats['external_links']}")
        report.append(f"- **Internal doc links**: {self.stats['internal_doc_links']}")
        report.append(f"- **Code references**: {self.stats['code_references']}")
        report.append(f"- **Incorrect paths**: {self.stats['incorrect_paths']}")
        report.append("")

        # Health Score
        if self.stats['total_links'] > 0:
            health_score = (self.stats['valid_links'] / (self.stats['total_links'] - self.stats['external_links'])) * 100
            report.append(f"### 🏥 Documentation Health Score: {health_score:.1f}%")
            report.append("")

        # Critical Issues
        if self.critical_issues:
            report.append(f"## 🔴 CRITICAL Issues ({len(self.critical_issues)})")
            report.append("")
            report.append("These are broken links to documentation files that must be fixed immediately.")
            report.append("")
            report.extend(self.critical_issues)
            report.append("")

        # High Priority Issues
        if self.high_issues:
            report.append(f"## 🟠 HIGH Priority Issues ({len(self.high_issues)})")
            report.append("")
            report.append("These are incorrect path references to code or broken code references.")
            report.append("")
            report.extend(self.high_issues)
            report.append("")

        # Medium Priority Issues
        if self.medium_issues:
            report.append(f"## 🟡 MEDIUM Priority Issues ({len(self.medium_issues)})")
            report.append("")
            report.append("These are other broken links that should be fixed.")
            report.append("")
            report.extend(self.medium_issues)
            report.append("")

        # Low Priority Issues
        if self.low_issues:
            report.append(f"## 🔵 LOW Priority Issues ({len(self.low_issues)})")
            report.append("")
            report.append("These are formatting inconsistencies or minor issues.")
            report.append("")
            report.extend(self.low_issues)
            report.append("")

        # Recommendations
        report.append("## 💡 Recommendations")
        report.append("")

        if self.critical_issues:
            report.append("1. **Fix critical broken documentation links immediately** - these break navigation")

        if self.stats['incorrect_paths'] > 0:
            report.append("2. **Update code path references** - remove '/project/' prefix and use relative paths")

        if self.stats['broken_links'] > 10:
            report.append("3. **Consider restructuring documentation** - high number of broken links indicates organizational issues")

        report.append("4. **Establish link validation in CI/CD** - prevent future broken links")
        report.append("5. **Use consistent path conventions** - prefer relative paths within docs")
        report.append("")

        return "\n".join(report)

    def run(self):
        """Run full analysis"""
        print("Starting documentation link analysis...")
        self.find_markdown_files()
        self.extract_links()
        self.analyze_links()
        self.check_path_patterns()
        return self.generate_report()


def main():
    docs_dir = "/home/devuser/workspace/project/docs"
    project_root = "/home/devuser/workspace/project"

    analyzer = LinkAnalyzer(docs_dir, project_root)
    report = analyzer.run()

    # Write report
    report_file = "/home/devuser/workspace/project/docs/link-analysis-report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ Report written to: {report_file}")
    print("\n" + "="*80)
    print(report)


if __name__ == "__main__":
    main()
