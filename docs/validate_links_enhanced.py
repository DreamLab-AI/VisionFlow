#!/usr/bin/env python3
"""
Enhanced Link Validator for VisionFlow Documentation
Provides detailed analysis with categorization and recommendations
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse
from datetime import datetime

class EnhancedLinkValidator:
    def __init__(self, docs_root):
        self.docs_root = Path(docs_root)
        self.project_root = Path('/home/devuser/workspace/project')
        self.links_by_file = defaultdict(list)
        self.all_files = set()
        self.broken_internal = []
        self.broken_anchors = []
        self.external_links = []
        self.anchor_links = []
        self.inbound_links = defaultdict(list)
        self.outbound_counts = defaultdict(int)
        self.file_stats = defaultdict(lambda: {'lines': 0, 'links': 0})

    def find_all_md_files(self):
        """Find all markdown files in docs directory"""
        for md_file in self.docs_root.rglob('*.md'):
            rel_path = str(md_file.relative_to(self.docs_root))
            self.all_files.add(rel_path)
        return self.all_files

    def extract_links(self, content, filename):
        """Extract all markdown links from content"""
        pattern = r'\[([^\]]*)\]\(([^)]+)\)'
        links = re.findall(pattern, content)
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
                with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    relative_path = str(md_file.relative_to(self.docs_root))

                    # Store file stats
                    self.file_stats[relative_path]['lines'] = len(content.split('\n'))

                    # Extract links
                    links = self.extract_links(content, relative_path)
                    self.file_stats[relative_path]['links'] = len(links)

                    for text, url in links:
                        if not url.strip():
                            continue

                        link_type = self._categorize_link(url)
                        self.links_by_file[relative_path].append({
                            'text': text,
                            'url': url,
                            'type': link_type
                        })

                        if link_type == 'external':
                            self.external_links.append({'source': relative_path, 'url': url})
                        elif link_type == 'anchor':
                            self.anchor_links.append({'source': relative_path, 'anchor': url})
                        elif link_type == 'internal':
                            self._validate_internal_link(relative_path, url)

                        self.outbound_counts[relative_path] += 1

            except Exception as e:
                print(f"Error reading {md_file}: {e}")

    def _validate_internal_link(self, source_file, url):
        """Validate internal link"""
        source_dir = Path(source_file).parent
        url_parts = url.split('#')
        file_part = url_parts[0]

        if file_part:
            if file_part.startswith('/'):
                target_file = self.project_root / file_part.lstrip('/')
            else:
                target_file = (self.docs_root / source_dir / file_part).resolve()
        else:
            target_file = self.docs_root / source_file

        # Try with .md extension if not found
        if not target_file.exists() and not target_file.suffix == '.md':
            target_file_md = target_file.with_suffix('.md')
            if target_file_md.exists():
                target_file = target_file_md

        if not target_file.exists():
            try:
                if target_file.is_relative_to(self.docs_root.parent):
                    resolved = str(target_file.relative_to(self.docs_root.parent))
                else:
                    resolved = str(target_file)
            except:
                resolved = str(target_file)

            self.broken_internal.append({
                'source': source_file,
                'target': url,
                'file': resolved,
                'type': 'missing_file'
            })

    def get_orphaned_files(self):
        """Find files with no inbound links"""
        orphaned = []
        entry_points = {'README.md', 'INDEX.md', 'NAVIGATION.md', 'OVERVIEW.md'}

        for md_file in self.all_files:
            has_inbound = False
            for links in self.links_by_file.values():
                for link in links:
                    if link['type'] == 'internal':
                        url = link['url'].split('#')[0]
                        if not url or url == md_file or url.lstrip('/') == md_file:
                            has_inbound = True
                            break
                if has_inbound:
                    break

            if not has_inbound and md_file not in entry_points:
                orphaned.append(md_file)

        return orphaned

    def get_files_with_no_outbound(self):
        """Find files with no outbound links"""
        return [f for f in self.all_files if self.outbound_counts[f] == 0]

    def categorize_broken_links(self):
        """Categorize broken links by pattern"""
        categories = {
            'missing_docs_files': [],
            'missing_subdirectories': [],
            'wrong_paths': [],
            'external_refs': []
        }

        for link in self.broken_internal:
            target = link['file']
            if 'docs/' in target:
                if 'guides/' in target or 'explanations/' in target or 'reference/' in target:
                    categories['missing_subdirectories'].append(link)
                else:
                    categories['missing_docs_files'].append(link)
            elif '../' in link['target']:
                categories['wrong_paths'].append(link)
            else:
                categories['external_refs'].append(link)

        return categories

    def generate_detailed_report(self):
        """Generate comprehensive link validation report"""
        report = []
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report.append("# Link Validation Report - VisionFlow Documentation\n\n")
        report.append(f"**Date**: {now}\n")
        report.append(f"**Documentation Root**: `/docs`\n")
        report.append(f"**Project Root**: `/home/devuser/workspace/project`\n\n")

        # Executive Summary
        total_links = sum(len(links) for links in self.links_by_file.values())
        broken_count = len(self.broken_internal) + len(self.broken_anchors)
        health_percent = ((total_links - broken_count) / total_links * 100) if total_links > 0 else 0

        report.append("## Executive Summary\n\n")
        report.append(f"**Link Health Score**: {health_percent:.1f}%\n\n")
        report.append(f"| Metric | Count |\n")
        report.append(f"|--------|-------|\n")
        report.append(f"| Total Files | {len(self.all_files)} |\n")
        report.append(f"| Total Links | {total_links} |\n")
        report.append(f"| Valid Links | {total_links - broken_count} |\n")
        report.append(f"| Broken Links | {broken_count} |\n")
        report.append(f"| Orphaned Files | {len(self.get_orphaned_files())} |\n")
        report.append(f"| Files with No Links | {len(self.get_files_with_no_outbound())} |\n\n")

        # Link Type Breakdown
        report.append("## Link Type Distribution\n\n")
        report.append(f"- **Internal Links**: {len([l for l in self.links_by_file.values() for li in l if li['type'] == 'internal'])}\n")
        report.append(f"- **External Links**: {len(self.external_links)}\n")
        report.append(f"- **Anchor Links**: {len(self.anchor_links)}\n\n")

        # Broken Links Analysis
        report.append("## Broken Links Analysis\n\n")
        categories = self.categorize_broken_links()

        if self.broken_internal:
            report.append(f"### Missing Internal Files ({len(self.broken_internal)})\n\n")
            for cat_name, cat_links in categories.items():
                if cat_links:
                    report.append(f"**{cat_name.replace('_', ' ').title()}**: {len(cat_links)} links\n\n")
                    for link in sorted(cat_links, key=lambda x: x['source'])[:10]:
                        report.append(f"- **File**: `{link['source']}`\n")
                        report.append(f"  - **Link**: `{link['target']}`\n")
                        report.append(f"  - **Expected**: `{link['file']}`\n\n")
                    if len(cat_links) > 10:
                        report.append(f"  ... and {len(cat_links) - 10} more\n\n")
        else:
            report.append("### Missing Internal Files\nNone detected.\n\n")

        if self.broken_anchors:
            report.append(f"### Broken Anchor Links ({len(self.broken_anchors)})\n\n")
            for link in self.broken_anchors[:20]:
                report.append(f"- `{link['source']}` -> `{link['anchor']}`\n")
            if len(self.broken_anchors) > 20:
                report.append(f"... and {len(self.broken_anchors) - 20} more\n\n")
        else:
            report.append("### Broken Anchor Links\nNone detected.\n\n")

        # Orphaned Files
        orphaned = self.get_orphaned_files()
        if orphaned:
            report.append(f"## Orphaned Files ({len(orphaned)})\n\n")
            report.append("Files with no inbound links (potential candidates for deletion or linking):\n\n")
            for file in sorted(orphaned)[:30]:
                report.append(f"- `{file}`\n")
            if len(orphaned) > 30:
                report.append(f"... and {len(orphaned) - 30} more\n\n")
        else:
            report.append("## Orphaned Files\nNone detected.\n\n")

        # Files with No Outbound Links
        no_outbound = self.get_files_with_no_outbound()
        if no_outbound:
            report.append(f"## Files with No Outbound Links ({len(no_outbound)})\n\n")
            report.append("These files don't link to other documentation:\n\n")
            for file in sorted(no_outbound)[:20]:
                report.append(f"- `{file}`\n")
            if len(no_outbound) > 20:
                report.append(f"... and {len(no_outbound) - 20} more\n\n")
        else:
            report.append("## Files with No Outbound Links\nNone detected.\n\n")

        # Top Files by Links
        report.append("## Top Files by Link Volume\n\n")
        sorted_files = sorted(self.outbound_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for file, count in sorted_files:
            report.append(f"- `{file}`: {count} links\n")
        report.append("\n")

        # External Link Domains
        report.append("## External Link Sources\n\n")
        domains = defaultdict(int)
        for link in self.external_links:
            url = link['url']
            try:
                parsed = urlparse(url)
                domain = parsed.netloc or parsed.path.split('/')[0]
                domains[domain] += 1
            except:
                domains['unknown'] += 1

        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:15]:
            report.append(f"- `{domain}`: {count} links\n")
        report.append("\n")

        # Recommendations
        report.append("## Recommendations\n\n")
        if len(self.broken_internal) > 50:
            report.append(f"1. **High Priority**: Address {len(self.broken_internal)} broken internal links\n")
            report.append("   - Create missing documentation files\n")
            report.append("   - Update incorrect link paths\n\n")
        if len(orphaned) > 20:
            report.append(f"2. **Medium Priority**: Review {len(orphaned)} orphaned files\n")
            report.append("   - Consider adding inbound links from other documents\n")
            report.append("   - Or remove if no longer relevant\n\n")
        if len(no_outbound) > 10:
            report.append(f"3. **Low Priority**: Consider adding internal links to {len(no_outbound)} standalone files\n")
            report.append("   - Improve navigation between related topics\n\n")

        report.append("## Statistics by Directory\n\n")
        dir_stats = defaultdict(lambda: {'files': 0, 'links': 0})
        for file in self.all_files:
            dir_name = Path(file).parent
            if str(dir_name) == '.':
                dir_name = 'root'
            dir_stats[str(dir_name)]['files'] += 1
            dir_stats[str(dir_name)]['links'] += self.outbound_counts[file]

        for dir_name in sorted(dir_stats.keys()):
            stats = dir_stats[dir_name]
            report.append(f"- `{dir_name}/`: {stats['files']} files, {stats['links']} total links\n")

        return "".join(report)


if __name__ == '__main__':
    docs_root = '/home/devuser/workspace/project/docs'
    validator = EnhancedLinkValidator(docs_root)

    print("Step 1: Finding markdown files...")
    validator.find_all_md_files()
    print(f"  Found {len(validator.all_files)} files")

    print("Step 2: Validating links...")
    validator.validate_links()
    print(f"  Extracted links from {len(validator.links_by_file)} files")

    print("Step 3: Generating report...")
    report = validator.generate_detailed_report()

    # Write report
    report_path = Path(docs_root) / 'reports' / 'link-validation.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print(f"Report size: {len(report)} characters")
