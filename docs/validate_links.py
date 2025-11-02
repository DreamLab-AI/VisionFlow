#!/usr/bin/env python3
"""
Documentation Link Validation Script for VisionFlow
Validates internal markdown links and anchor references
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set

class LinkValidator:
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir)
        self.markdown_files = []
        self.broken_links = []
        self.broken_anchors = []
        self.valid_files = set()

    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files in the docs directory"""
        markdown_files = []
        for root, dirs, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith('.md'):
                    markdown_files.append(Path(root) / file)
        return markdown_files

    def extract_links(self, file_path: Path) -> List[Tuple[str, int]]:
        """Extract all markdown links from a file"""
        links = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all markdown links [text](target)
            link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'

            for line_num, line in enumerate(content.split('\n'), 1):
                matches = re.finditer(link_pattern, line)
                for match in matches:
                    target = match.group(2)
                    # Skip external URLs and mailto links
                    if not target.startswith(('http://', 'https://', 'mailto:', '#')):
                        links.append((target, line_num))

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        return links

    def extract_anchors(self, file_path: Path) -> Set[str]:
        """Extract all heading anchors from a file"""
        anchors = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all headings (# ## ### etc.)
            heading_pattern = r'^(#{1,6})\s+(.+)$'

            for line in content.split('\n'):
                match = re.match(heading_pattern, line)
                if match:
                    heading = match.group(2).strip()
                    # Convert to anchor format (lowercase, hyphens, no special chars)
                    anchor = re.sub(r'[^\w\s-]', '', heading.lower())
                    anchor = re.sub(r'[-\s]+', '-', anchor).strip('-')
                    anchors.add(anchor)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        return anchors

    def validate_internal_links(self):
        """Validate all internal documentation links"""
        print("Finding all markdown files...")
        self.markdown_files = self.find_markdown_files()
        self.valid_files = {str(f.relative_to(self.docs_dir)) for f in self.markdown_files}

        print(f"Found {len(self.markdown_files)} markdown files")
        print("\nValidating internal links...")

        # Extract all anchors from all files
        file_anchors = {}
        for file_path in self.markdown_files:
            relative_path = str(file_path.relative_to(self.docs_dir))
            file_anchors[relative_path] = self.extract_anchors(file_path)

        total_links = 0

        for file_path in self.markdown_files:
            relative_path = str(file_path.relative_to(self.docs_dir))
            links = self.extract_links(file_path)

            for target, line_num in links:
                total_links += 1

                # Handle anchor links (#anchor)
                if target.startswith('#'):
                    anchor = target[1:]
                    if anchor not in file_anchors.get(relative_path, set()):
                        self.broken_anchors.append((relative_path, anchor, line_num))

                # Handle file paths and file+anchor combos
                elif '#' in target:
                    file_target, anchor = target.split('#', 1)
                    target_file = self.resolve_path(file_target, relative_path)

                    if target_file not in self.valid_files:
                        # Check if it might be a directory link
                        if not file_target.endswith('/'):
                            dir_target = self.resolve_path(file_target + '/', relative_path)
                            if any(f.startswith(dir_target) for f in self.valid_files):
                                continue  # Valid directory link
                        self.broken_links.append((relative_path, target, line_num, "File not found"))
                    elif anchor and anchor not in file_anchors.get(target_file, set()):
                        self.broken_anchors.append((relative_path, anchor, line_num, f"in {target_file}"))

                # Handle file paths only
                else:
                    # Skip directory links ending with /
                    if target.endswith('/'):
                        # Check if any files exist in this directory
                        dir_target = self.resolve_path(target, relative_path)
                        if any(f.startswith(dir_target) for f in self.valid_files):
                            continue  # Valid directory link
                        else:
                            self.broken_links.append((relative_path, target, line_num, "Directory not found"))
                    else:
                        target_file = self.resolve_path(target, relative_path)
                        if target_file not in self.valid_files:
                            # Try with .md extension
                            target_file_md = target_file + '.md'
                            if target_file_md not in self.valid_files:
                                self.broken_links.append((relative_path, target, line_num, "File not found"))

        print(f"Checked {total_links} links")

    def resolve_path(self, target: str, source_file: str) -> str:
        """Resolve relative path to absolute path within docs"""
        source_dir = str(Path(source_file).parent)

        if target.startswith('./'):
            target = target[2:]
        elif target.startswith('../'):
            # Handle parent directory references
            parts = source_dir.split('/')
            target_parts = target.split('/')

            while target_parts and target_parts[0] == '..':
                if parts:
                    parts.pop()
                target_parts.pop(0)

            target = '/'.join(parts + target_parts)
        else:
            # Handle relative paths
            if source_dir:
                target = f"{source_dir}/{target}"

        # Normalize path separators
        target = target.replace('\\', '/')

        # Remove .md extension if present for consistency
        if target.endswith('.md'):
            pass  # Keep the extension for file checking

        return target

    def generate_report(self) -> str:
        """Generate a validation report"""
        report = []
        report.append("# Documentation Link Validation Report")
        report.append(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## Summary")
        report.append(f"- Total markdown files: {len(self.markdown_files)}")
        report.append(f"- Broken internal links: {len(self.broken_links)}")
        report.append(f"- Broken anchor links: {len(self.broken_anchors)}")
        report.append("")

        if self.broken_links:
            report.append("## Broken Internal Links")
            report.append("")
            for source_file, target, line_num, error in self.broken_links:
                report.append(f"### {source_file}:{line_num}")
                report.append(f"- **Target:** `{target}`")
                report.append(f"- **Error:** {error}")
                report.append("")

        if self.broken_anchors:
            report.append("## Broken Anchor Links")
            report.append("")
            for source_file, anchor, line_num, *extra in self.broken_anchors:
                location = f" in {extra[0]}" if extra else ""
                report.append(f"### {source_file}:{line_num}")
                report.append(f"- **Anchor:** `#{anchor}`")
                report.append(f"- **Error:** Heading not found{location}")
                report.append("")

        if not self.broken_links and not self.broken_anchors:
            report.append("✅ All internal links and anchors are valid!")

        return "\n".join(report)

    def save_report(self, output_file: str = "LINK_VALIDATION_REPORT.md"):
        """Save validation report to file"""
        report = self.generate_report()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nValidation report saved to: {output_file}")

        # Print summary to console
        if self.broken_links or self.broken_anchors:
            print(f"\n❌ Found {len(self.broken_links)} broken links and {len(self.broken_anchors)} broken anchors")
            return False
        else:
            print("\n✅ All links are valid!")
            return True

if __name__ == "__main__":
    validator = LinkValidator()
    validator.validate_internal_links()
    success = validator.save_report()
    sys.exit(0 if success else 1)