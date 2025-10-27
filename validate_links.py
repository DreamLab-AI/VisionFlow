#!/usr/bin/env python3
"""
Comprehensive documentation link validator.
Checks forward links (A->B) and backward links (B->A).
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List

class LinkValidator:
    def __init__(self, docs_root: str):
        self.docs_root = Path(docs_root)
        self.files: Set[Path] = set()
        self.forward_links: Dict[Path, Set[Tuple[Path, str]]] = defaultdict(set)
        self.backward_links: Dict[Path, Set[Tuple[Path, str]]] = defaultdict(set)
        self.broken_links: List[Tuple[str, Path, str]] = []
        self.missing_backlinks: List[Tuple[Path, Path]] = []

    def discover_files(self):
        """Find all markdown files"""
        for md_file in self.docs_root.rglob("*.md"):
            self.files.add(md_file)
        print(f"✅ Found {len(self.files)} markdown files")

    def extract_links(self):
        """Extract all markdown links from files"""
        # Pattern for markdown links: [text](path)
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        for md_file in self.files:
            try:
                # Try to read with UTF-8, fallback to latin-1
                try:
                    content = md_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    content = md_file.read_text(encoding='latin-1')

                matches = link_pattern.findall(content)

                for text, link in matches:
                    # Only process relative markdown links
                    if not link.startswith('http') and link.endswith('.md'):
                        # Resolve relative paths
                        if link.startswith('./'):
                            target = (md_file.parent / link).resolve()
                        elif link.startswith('../'):
                            target = (md_file.parent / link).resolve()
                        else:
                            # Relative to docs root
                            target = (self.docs_root / link).resolve()

                        self.forward_links[md_file].add((target, link))

                        # Only track backward link if target is within docs
                        if target.is_relative_to(self.docs_root) if hasattr(target, 'is_relative_to') else str(target).startswith(str(self.docs_root)):
                            self.backward_links[target].add((md_file, link))

            except Exception as e:
                print(f"⚠️  Error processing {md_file}: {e}")

    def validate_links(self):
        """Check if all referenced links exist"""
        for source, targets in self.forward_links.items():
            for target, link in targets:
                # Check if target exists
                if not target.exists():
                    src_rel = source.relative_to(self.docs_root).__str__()
                    # Try to get relative target path
                    try:
                        tgt_rel = target.relative_to(self.docs_root).__str__()
                    except ValueError:
                        tgt_rel = str(target)
                    self.broken_links.append((src_rel, tgt_rel, link))

    def generate_report(self) -> str:
        """Generate validation report"""
        report = []
        report.append("\n" + "="*80)
        report.append("📋 DOCUMENTATION LINK VALIDATION REPORT")
        report.append("="*80 + "\n")

        # Summary stats
        total_files = len(self.files)
        total_forward_links = sum(len(links) for links in self.forward_links.values())
        total_backward_links = sum(len(links) for links in self.backward_links.values())

        report.append(f"📊 STATISTICS:")
        report.append(f"  • Total markdown files: {total_files}")
        report.append(f"  • Total forward links: {total_forward_links}")
        report.append(f"  • Total backward links: {total_backward_links}")
        report.append(f"  • Files with links: {len(self.forward_links)}")
        report.append(f"  • Files linked to: {len(self.backward_links)}")

        # Broken links
        report.append(f"\n🔗 LINK VALIDATION RESULTS:")
        if self.broken_links:
            report.append(f"  ❌ BROKEN LINKS ({len(self.broken_links)}):")
            for source, target, link in sorted(self.broken_links):
                report.append(f"     • {source}")
                report.append(f"       └─> {link} (target not found: {target})")
        else:
            report.append(f"  ✅ All forward links are valid!")

        # Files by link count
        report.append(f"\n📍 FILES BY FORWARD LINK COUNT:")
        links_by_file = sorted([(f, len(links)) for f, links in self.forward_links.items()],
                              key=lambda x: x[1], reverse=True)

        if links_by_file:
            report.append(f"  Top 10 files with most links:")
            for file, count in links_by_file[:10]:
                rel_path = file.relative_to(self.docs_root)
                report.append(f"    • {rel_path}: {count} links")

        # Files by backward link count (entry points)
        report.append(f"\n🎯 MOST REFERENCED FILES (Entry Points):")
        backlinks_by_file = sorted([(f, len(links)) for f, links in self.backward_links.items()],
                                  key=lambda x: x[1], reverse=True)

        if backlinks_by_file:
            report.append(f"  Top 10 most referenced files:")
            for file, count in backlinks_by_file[:10]:
                rel_path = file.relative_to(self.docs_root)
                report.append(f"    • {rel_path}: referenced by {count} files")

        # Navigation structure validation
        report.append(f"\n🗺️  NAVIGATION STRUCTURE VALIDATION:")

        # Check key navigation files exist
        key_files = [
            "README.md",
            "docs/concepts/README.md",
            "docs/reference/README.md",
            "docs/reference/api/README.md",
            "docs/reference/architecture/README.md",
            "docs/getting-started/01-installation.md",
            "docs/guides/developer/README.md",
        ]

        for key_file in key_files:
            path = self.docs_root / key_file
            if path.exists():
                links_from = len(self.forward_links.get(path, set()))
                links_to = len(self.backward_links.get(path, set()))
                report.append(f"  ✅ {key_file}")
                report.append(f"     └─ {links_from} outgoing links, {links_to} incoming links")
            else:
                report.append(f"  ❌ {key_file} (MISSING)")

        # Orphaned files (no incoming links)
        report.append(f"\n👻 POTENTIALLY ORPHANED FILES (no incoming links):")
        orphaned = []
        for file in self.files:
            if file.relative_to(self.docs_root).name not in ["README.md", "index.md"]:
                if file not in self.backward_links or len(self.backward_links[file]) == 0:
                    orphaned.append(file)

        if orphaned and len(orphaned) <= 20:
            for file in sorted(orphaned)[:20]:
                report.append(f"    • {file.relative_to(self.docs_root)}")
            if len(orphaned) > 20:
                report.append(f"    ... and {len(orphaned) - 20} more")
        elif not orphaned:
            report.append("  ✅ No orphaned files found!")
        else:
            report.append(f"  ⚠️  {len(orphaned)} orphaned files (showing first 20)")
            for file in sorted(orphaned)[:20]:
                report.append(f"    • {file.relative_to(self.docs_root)}")

        # Links health
        report.append(f"\n📈 OVERALL HEALTH:")
        if self.broken_links:
            health = f"⚠️  ISSUES FOUND: {len(self.broken_links)} broken link(s)"
        else:
            health = f"✅ EXCELLENT: All {total_forward_links} forward links are valid!"
        report.append(f"  {health}")

        report.append("\n" + "="*80 + "\n")

        return "\n".join(report)

    def validate_and_report(self) -> Tuple[str, bool]:
        """Run full validation and return report with status"""
        self.discover_files()
        self.extract_links()
        self.validate_links()
        report = self.generate_report()
        is_healthy = len(self.broken_links) == 0
        return report, is_healthy

def main():
    docs_root = "/home/devuser/workspace/project/docs"

    validator = LinkValidator(docs_root)
    report, is_healthy = validator.validate_and_report()

    print(report)

    # Write report to file
    report_path = Path(docs_root) / "LINK_VALIDATION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write("# Documentation Link Validation Report\n\n")
        f.write("**Generated:** Automated link validation\n\n")
        f.write(report)

    print(f"📄 Report saved to: {report_path}")

    return 0 if is_healthy else 1

if __name__ == "__main__":
    exit(main())
