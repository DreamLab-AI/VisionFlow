#!/usr/bin/env python3
"""
Comprehensive Link Validation for VisionFlow Documentation
Validates ALL markdown links, builds link graph, identifies orphaned files
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

DOCS_ROOT = Path("/home/devuser/workspace/project/docs")
WORKING_DIR = Path("/home/devuser/workspace/project/docs/working")

# Regex patterns
MARKDOWN_LINK_PATTERN = r'\[([^\]]+)\]\(([^\)]+)\)'
ANCHOR_PATTERN = r'#([a-z0-9\-]+)'

class LinkValidator:
    def __init__(self):
        self.all_files: List[Path] = []
        self.links: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # source -> [(text, target, type)]
        self.inbound_links: Dict[str, Set[str]] = defaultdict(set)  # target -> sources
        self.broken_links: List[Dict] = []
        self.invalid_anchors: List[Dict] = []
        self.orphaned_files: List[str] = []
        self.isolated_files: List[str] = []
        self.headings: Dict[str, Set[str]] = defaultdict(set)  # file -> heading anchors

    def get_anchor_from_heading(self, heading: str) -> str:
        """Convert markdown heading to anchor ID (GitHub style)"""
        anchor = heading.lower()
        anchor = re.sub(r'[^\w\s\-]', '', anchor)  # Remove non-word chars except hyphens
        anchor = re.sub(r'\s+', '-', anchor)  # Spaces to hyphens
        return anchor

    def extract_headings(self, file_path: Path) -> Set[str]:
        """Extract all heading anchors from a markdown file"""
        anchors = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Match markdown headings: # Heading, ## Heading, etc.
                    heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
                    if heading_match:
                        heading_text = heading_match.group(2)
                        # Remove markdown formatting from heading
                        heading_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', heading_text)  # Remove links
                        heading_text = re.sub(r'[*_`]', '', heading_text)  # Remove emphasis
                        anchor = self.get_anchor_from_heading(heading_text)
                        anchors.add(anchor)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return anchors

    def collect_files(self):
        """Collect all markdown files in docs directory"""
        for root, dirs, files in os.walk(DOCS_ROOT):
            # Skip working directory itself during collection
            if Path(root) == WORKING_DIR:
                continue
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    self.all_files.append(file_path)
                    # Extract headings for anchor validation
                    self.headings[str(file_path.relative_to(DOCS_ROOT))] = self.extract_headings(file_path)

        print(f"Collected {len(self.all_files)} markdown files")

    def extract_links(self):
        """Extract all links from all markdown files"""
        for file_path in self.all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Find all markdown links
                    for match in re.finditer(MARKDOWN_LINK_PATTERN, content):
                        link_text = match.group(1)
                        link_target = match.group(2)

                        # Skip external links (http://, https://, mailto:)
                        if link_target.startswith(('http://', 'https://', 'mailto:')):
                            continue

                        source_rel = str(file_path.relative_to(DOCS_ROOT))

                        # Determine link type
                        if '#' in link_target:
                            link_type = 'anchor'
                        else:
                            link_type = 'internal'

                        self.links[source_rel].append((link_text, link_target, link_type))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print(f"Extracted {sum(len(v) for v in self.links.values())} links from {len(self.links)} files")

    def validate_links(self):
        """Validate all extracted links"""
        for source_file, file_links in self.links.items():
            source_path = DOCS_ROOT / source_file

            for link_text, link_target, link_type in file_links:
                # Split anchor from path
                if '#' in link_target:
                    target_path, anchor = link_target.split('#', 1)
                else:
                    target_path = link_target
                    anchor = None

                # Resolve relative path
                if target_path:
                    # Handle relative paths
                    if target_path.startswith('/'):
                        # Absolute path from docs root
                        resolved_path = DOCS_ROOT / target_path.lstrip('/')
                    else:
                        # Relative to current file
                        resolved_path = (source_path.parent / target_path).resolve()

                    # Check if file exists
                    if not resolved_path.exists():
                        self.broken_links.append({
                            'source': source_file,
                            'link_text': link_text,
                            'target': link_target,
                            'reason': 'File not found',
                            'resolved_path': str(resolved_path)
                        })
                        continue

                    # Skip links outside docs directory (cross-repository links)
                    try:
                        target_rel = str(resolved_path.relative_to(DOCS_ROOT))
                    except ValueError:
                        # Link points outside docs directory - skip it
                        continue

                    # Track inbound link
                    self.inbound_links[target_rel].add(source_file)

                    # Validate anchor if present
                    if anchor:
                        # Check if anchor exists in target file
                        target_headings = self.headings.get(target_rel, set())
                        if anchor not in target_headings:
                            self.invalid_anchors.append({
                                'source': source_file,
                                'target': target_rel,
                                'anchor': anchor,
                                'link_text': link_text,
                                'available_anchors': sorted(list(target_headings))[:10]  # First 10 for reference
                            })
                else:
                    # Anchor-only link (same file)
                    if anchor:
                        source_headings = self.headings.get(source_file, set())
                        if anchor not in source_headings:
                            self.invalid_anchors.append({
                                'source': source_file,
                                'target': source_file,
                                'anchor': anchor,
                                'link_text': link_text,
                                'available_anchors': sorted(list(source_headings))[:10]
                            })

    def find_orphaned_files(self):
        """Find files with no inbound links (except INDEX.md and README.md)"""
        for file_path in self.all_files:
            file_rel = str(file_path.relative_to(DOCS_ROOT))

            # Skip root-level index files
            if file_path.name in ['README.md', 'INDEX.md', 'OVERVIEW.md']:
                continue

            # Check if file has inbound links
            if file_rel not in self.inbound_links or len(self.inbound_links[file_rel]) == 0:
                self.orphaned_files.append(file_rel)

    def find_isolated_files(self):
        """Find files with no outbound links"""
        all_files_rel = {str(f.relative_to(DOCS_ROOT)) for f in self.all_files}
        files_with_links = set(self.links.keys())

        self.isolated_files = sorted(list(all_files_rel - files_with_links))

    def build_link_graph(self) -> Dict:
        """Build complete link graph"""
        graph = {}

        for source_file, file_links in self.links.items():
            targets = []
            for _, link_target, _ in file_links:
                # Remove anchors for graph
                target_path = link_target.split('#')[0] if '#' in link_target else link_target
                if target_path:  # Skip anchor-only links
                    targets.append(target_path)
            graph[source_file] = targets

        return graph

    def calculate_link_health(self) -> float:
        """Calculate overall link health percentage"""
        total_links = sum(len(v) for v in self.links.values())
        if total_links == 0:
            return 0.0

        broken_count = len(self.broken_links) + len(self.invalid_anchors)
        healthy_count = total_links - broken_count

        return (healthy_count / total_links) * 100

    def generate_report(self):
        """Generate comprehensive JSON and Markdown reports"""
        # JSON report
        json_report = {
            'total_files': len(self.all_files),
            'total_links': sum(len(v) for v in self.links.values()),
            'broken_links': self.broken_links,
            'invalid_anchors': self.invalid_anchors,
            'orphaned_files': self.orphaned_files,
            'isolated_files': self.isolated_files,
            'link_graph': self.build_link_graph(),
            'link_health_percentage': self.calculate_link_health(),
            'inbound_link_counts': {
                file: len(sources) for file, sources in self.inbound_links.items()
            }
        }

        json_path = WORKING_DIR / 'hive-link-validation.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2)

        print(f"JSON report written to {json_path}")

        # Markdown report
        md_report = self.generate_markdown_report()
        md_path = WORKING_DIR / 'hive-link-validation.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)

        print(f"Markdown report written to {md_path}")

        return json_report

    def generate_markdown_report(self) -> str:
        """Generate human-readable markdown report"""
        lines = []
        lines.append("# Link Validation Report")
        lines.append("")
        lines.append(f"**Generated**: {Path(__file__).name}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Files**: {len(self.all_files)}")
        lines.append(f"- **Total Links**: {sum(len(v) for v in self.links.values())}")
        lines.append(f"- **Broken Links**: {len(self.broken_links)}")
        lines.append(f"- **Invalid Anchors**: {len(self.invalid_anchors)}")
        lines.append(f"- **Orphaned Files**: {len(self.orphaned_files)}")
        lines.append(f"- **Isolated Files**: {len(self.isolated_files)}")
        lines.append(f"- **Link Health**: {self.calculate_link_health():.2f}%")
        lines.append("")

        # Broken Links
        if self.broken_links:
            lines.append("## Broken Links")
            lines.append("")
            for broken in self.broken_links[:50]:  # First 50
                lines.append(f"- **{broken['source']}**")
                lines.append(f"  - Link: `[{broken['link_text']}]({broken['target']})`")
                lines.append(f"  - Reason: {broken['reason']}")
                lines.append("")

        # Invalid Anchors
        if self.invalid_anchors:
            lines.append("## Invalid Anchors")
            lines.append("")
            for invalid in self.invalid_anchors[:50]:  # First 50
                lines.append(f"- **{invalid['source']}**")
                lines.append(f"  - Link: `[{invalid['link_text']}](#{invalid['anchor']})`")
                lines.append(f"  - Target: {invalid['target']}")
                if invalid['available_anchors']:
                    lines.append(f"  - Available anchors: {', '.join(invalid['available_anchors'][:5])}")
                lines.append("")

        # Orphaned Files
        if self.orphaned_files:
            lines.append("## Orphaned Files (No Inbound Links)")
            lines.append("")
            for orphan in sorted(self.orphaned_files)[:50]:  # First 50
                lines.append(f"- `{orphan}`")
            lines.append("")

        # Isolated Files
        if self.isolated_files:
            lines.append("## Isolated Files (No Outbound Links)")
            lines.append("")
            for isolated in sorted(self.isolated_files)[:50]:  # First 50
                lines.append(f"- `{isolated}`")
            lines.append("")

        # Top Referenced Files
        lines.append("## Top Referenced Files")
        lines.append("")
        top_referenced = sorted(
            self.inbound_links.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20]

        for file, sources in top_referenced:
            lines.append(f"- **{file}** ({len(sources)} inbound links)")
        lines.append("")

        return '\n'.join(lines)

    def run(self):
        """Run complete validation pipeline"""
        print("Starting link validation...")
        self.collect_files()
        self.extract_links()
        self.validate_links()
        self.find_orphaned_files()
        self.find_isolated_files()
        report = self.generate_report()

        print("\n=== VALIDATION COMPLETE ===")
        print(f"Link Health: {report['link_health_percentage']:.2f}%")
        print(f"Broken Links: {len(report['broken_links'])}")
        print(f"Invalid Anchors: {len(report['invalid_anchors'])}")
        print(f"Orphaned Files: {len(report['orphaned_files'])}")
        print(f"Isolated Files: {len(report['isolated_files'])}")

        return report

if __name__ == '__main__':
    validator = LinkValidator()
    validator.run()
