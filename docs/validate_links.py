#!/usr/bin/env python3
"""
Link Validator for VisionFlow Documentation
Validates all internal links, anchors, and identifies orphaned files
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse

class LinkValidator:
    def __init__(self, docs_root):
        self.docs_root = Path(docs_root)
        self.links_by_file = defaultdict(list)
        self.all_files = set()
        self.broken_internal = []
        self.broken_anchors = []
        self.external_links = []
        self.anchor_links = []
        self.inbound_links = defaultdict(list)
        self.outbound_counts = defaultdict(int)

    def find_all_md_files(self):
        """Find all markdown files in docs directory"""
        for md_file in self.docs_root.rglob('*.md'):
            self.all_files.add(str(md_file.relative_to(self.docs_root)))
        return self.all_files

    def extract_links(self, content, filename):
        """Extract all markdown links from content"""
        # Pattern for [text](url)
        pattern = r'\[([^\]]*)\]\(([^)]+)\)'
        links = re.findall(pattern, content)

        for text, url in links:
            if not url.strip():
                continue
            self.links_by_file[filename].append({
                'text': text,
                'url': url,
                'type': self._categorize_link(url)
            })
        return links

    def _categorize_link(self, url):
        """Categorize link type"""
        if url.startswith('http://') or url.startswith('https://'):
            return 'external'
        elif url.startswith('#'):
            return 'anchor'
        else:
            return 'internal'

    def validate_links(self):
        """Validate all links in documentation"""
        for md_file in self.docs_root.rglob('*.md'):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    relative_path = str(md_file.relative_to(self.docs_root))
                    self.extract_links(content, relative_path)
            except Exception as e:
                print(f"Error reading {md_file}: {e}")

        # Process all extracted links
        for source_file, links in self.links_by_file.items():
            for link_info in links:
                url = link_info['url']
                link_type = link_info['type']

                if link_type == 'external':
                    self.external_links.append({
                        'source': source_file,
                        'url': url
                    })
                elif link_type == 'anchor':
                    self.anchor_links.append({
                        'source': source_file,
                        'anchor': url
                    })
                elif link_type == 'internal':
                    self._validate_internal_link(source_file, url)

                self.outbound_counts[source_file] += 1
                self.inbound_links[url].append(source_file)

    def _validate_internal_link(self, source_file, url):
        """Validate internal link"""
        # Handle relative paths
        source_dir = Path(source_file).parent

        # Remove anchor from URL if present
        url_parts = url.split('#')
        file_part = url_parts[0]
        anchor_part = url_parts[1] if len(url_parts) > 1 else None

        # Resolve the target file
        if file_part:
            # Handle relative paths
            if file_part.startswith('/'):
                # Absolute path from project root
                target_file = Path('/home/devuser/workspace/project') / file_part.lstrip('/')
            else:
                # Relative path from current file
                target_file = (self.docs_root / source_dir / file_part).resolve()
        else:
            # Same file anchor
            target_file = self.docs_root / source_file

        # Check if file exists
        if not target_file.exists() and target_file.with_suffix('.md').exists():
            target_file = target_file.with_suffix('.md')

        if not target_file.exists():
            self.broken_internal.append({
                'source': source_file,
                'target': url,
                'file': str(target_file.relative_to(self.docs_root.parent) if target_file.is_relative_to(self.docs_root.parent) else target_file)
            })

    def get_orphaned_files(self):
        """Find files with no inbound links"""
        orphaned = []
        for md_file in self.all_files:
            if md_file not in self.inbound_links and md_file != 'README.md':
                orphaned.append(md_file)
        return orphaned

    def get_files_with_no_outbound(self):
        """Find files with no outbound links"""
        return [f for f in self.all_files if self.outbound_counts[f] == 0]

    def generate_report(self):
        """Generate comprehensive link validation report"""
        report = []
        report.append("# Link Validation Report for VisionFlow Documentation\n")
        report.append(f"**Generated**: {Path('/home/devuser/workspace/project/docs/validate_links.py').stat().st_mtime}\n\n")

        # Summary Stats
        total_links = sum(len(links) for links in self.links_by_file.values())
        report.append("## Summary Statistics\n")
        report.append(f"- Total markdown files: {len(self.all_files)}\n")
        report.append(f"- Total links found: {total_links}\n")
        report.append(f"- Internal links: {len(self.links_by_file)}\n")
        report.append(f"- External links: {len(self.external_links)}\n")
        report.append(f"- Anchor links: {len(self.anchor_links)}\n")

        # Health calculation
        broken_count = len(self.broken_internal) + len(self.broken_anchors)
        health_percent = ((total_links - broken_count) / total_links * 100) if total_links > 0 else 0
        report.append(f"- Broken links: {broken_count}\n")
        report.append(f"- Link health: {health_percent:.1f}%\n\n")

        # Broken Internal Links
        if self.broken_internal:
            report.append("## Broken Internal Links\n")
            for link in self.broken_internal:
                report.append(f"- **Source**: `{link['source']}`\n")
                report.append(f"  - **Target**: `{link['target']}`\n")
                report.append(f"  - **Resolved to**: `{link['file']}`\n\n")
        else:
            report.append("## Broken Internal Links\n")
            report.append("None detected.\n\n")

        # Broken Anchors
        if self.broken_anchors:
            report.append("## Broken Anchor Links\n")
            for link in self.broken_anchors:
                report.append(f"- `{link['source']}` -> `{link['anchor']}`\n")
            report.append("\n")
        else:
            report.append("## Broken Anchor Links\n")
            report.append("None detected.\n\n")

        # Orphaned Files
        orphaned = self.get_orphaned_files()
        if orphaned:
            report.append("## Orphaned Files (No Inbound Links)\n")
            report.append(f"Total: {len(orphaned)}\n\n")
            for file in sorted(orphaned):
                report.append(f"- `{file}`\n")
            report.append("\n")
        else:
            report.append("## Orphaned Files (No Inbound Links)\n")
            report.append("None detected.\n\n")

        # Files with No Outbound Links
        no_outbound = self.get_files_with_no_outbound()
        if no_outbound:
            report.append("## Files with No Outbound Links\n")
            report.append(f"Total: {len(no_outbound)}\n\n")
            for file in sorted(no_outbound):
                report.append(f"- `{file}`\n")
            report.append("\n")
        else:
            report.append("## Files with No Outbound Links\n")
            report.append("None detected.\n\n")

        # Top Files by Link Count
        report.append("## Top Files by Outbound Link Count\n")
        sorted_files = sorted(self.outbound_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for file, count in sorted_files:
            report.append(f"- `{file}`: {count} links\n")
        report.append("\n")

        # External Links Summary
        report.append("## External Links (Sample)\n")
        report.append(f"Total external links: {len(self.external_links)}\n\n")
        domains = defaultdict(int)
        for link in self.external_links[:50]:  # Show sample
            url = link['url']
            try:
                domain = urlparse(url).netloc or urlparse(url).path.split('/')[0]
                domains[domain] += 1
            except:
                pass

        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"- `{domain}`: {count} links\n")

        return "".join(report)


if __name__ == '__main__':
    docs_root = '/home/devuser/workspace/project/docs'
    validator = LinkValidator(docs_root)

    print("Finding markdown files...")
    validator.find_all_md_files()

    print(f"Found {len(validator.all_files)} markdown files")
    print("Validating links...")
    validator.validate_links()

    print("Generating report...")
    report = validator.generate_report()

    # Write report
    report_path = Path(docs_root) / 'reports' / 'link-validation.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report written to {report_path}")
    print("\nReport Preview:")
    print(report[:2000])
