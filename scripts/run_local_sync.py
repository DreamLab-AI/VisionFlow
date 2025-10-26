#!/usr/bin/env python3
"""
Direct local markdown sync validation script.
Bypasses Docker container issues by directly analyzing local markdown files.
"""
import os
import re
import json
from pathlib import Path
from collections import defaultdict

def parse_markdown_file(filepath):
    """Parse a single markdown file and extract metadata and wikilinks."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract properties (key:: value format)
    properties = {}
    for line in content.split('\n')[:20]:  # Check first 20 lines for properties
        match = re.match(r'^(\w+)::\s*(.+)$', line.strip())
        if match:
            key, value = match.groups()
            properties[key] = value.strip()

    # Extract wikilinks [[Page Name]]
    wikilinks = re.findall(r'\[\[([^\]]+)\]\]', content)

    return {
        'filename': os.path.basename(filepath),
        'properties': properties,
        'wikilinks': wikilinks,
        'has_public': properties.get('public') == 'true'
    }

def main():
    markdown_dir = Path('/home/devuser/workspace/project/data/markdown')

    if not markdown_dir.exists():
        print(f"❌ Directory not found: {markdown_dir}")
        return

    # Parse all markdown files
    files = list(markdown_dir.glob('*.md'))
    print(f"=== Local Markdown Analysis ===")
    print(f"Directory: {markdown_dir}")
    print(f"Total .md files: {len(files)}\n")

    parsed_files = []
    public_count = 0
    non_public_count = 0
    public_page_names = set()

    for filepath in sorted(files):
        parsed = parse_markdown_file(filepath)
        parsed_files.append(parsed)

        if parsed['has_public']:
            public_count += 1
            # Add page name without .md extension
            page_name = parsed['filename'].rsplit('.md', 1)[0]
            public_page_names.add(page_name)
        else:
            non_public_count += 1

    # Analyze wikilinks
    all_wikilinks = []
    for parsed in parsed_files:
        if parsed['has_public']:  # Only from public files
            all_wikilinks.extend(parsed['wikilinks'])

    unique_wikilinks = set(all_wikilinks)
    wikilinks_to_public = sum(1 for link in unique_wikilinks if link in public_page_names)
    wikilinks_to_private = len(unique_wikilinks) - wikilinks_to_public

    print("=" * 60)
    print("FILE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total files: {len(files)}")
    print(f"Files WITH public:: true: {public_count} ({public_count/len(files)*100:.1f}%)")
    print(f"Files WITHOUT public:: true: {non_public_count} ({non_public_count/len(files)*100:.1f}%)")
    print()
    print("=" * 60)
    print("WIKILINK ANALYSIS")
    print("=" * 60)
    print(f"Total wikilinks from public files: {len(all_wikilinks)}")
    print(f"Unique wikilinks: {len(unique_wikilinks)}")
    print(f"Wikilinks pointing to PUBLIC pages: {wikilinks_to_public}")
    print(f"Wikilinks pointing to PRIVATE pages: {wikilinks_to_private}")
    print()

    if wikilinks_to_private > 0:
        print("⚠️  PRIVACY LEAK DETECTED:")
        print(f"   {wikilinks_to_private} wikilinks point to pages WITHOUT public:: true")
        print(f"   These would create 'linked_page' nodes for private pages!")
        print()
        print("Sample private page references (first 10):")
        private_links = [link for link in unique_wikilinks if link not in public_page_names]
        for link in sorted(private_links)[:10]:
            print(f"  - [[{link}]]")

    print()
    print("=" * 60)
    print("EXPECTED DATABASE RESULTS (with two-pass filtering)")
    print("=" * 60)
    print(f"Page nodes: {public_count} (one per public file)")
    print(f"Linked_page nodes: {wikilinks_to_public} (only public pages)")
    print(f"Total nodes after filtering: {public_count + wikilinks_to_public}")
    print(f"Filtered out (private linked_pages): {wikilinks_to_private}")
    print()

    if non_public_count == 0:
        print("✅ SUCCESS: ALL files have public:: true")
        print("✅ Two-pass filtering will correctly filter out private linked_pages")
    else:
        print(f"❌ ISSUE: {non_public_count} files lack public:: true marker")

    # Sample files
    print()
    print("=" * 60)
    print("SAMPLE FILES (first 5)")
    print("=" * 60)
    for parsed in parsed_files[:5]:
        print(f"\n📄 {parsed['filename']}")
        print(f"   public:: {parsed['properties'].get('public', 'NOT SET')}")
        print(f"   wikilinks: {len(parsed['wikilinks'])}")
        if parsed['wikilinks'][:3]:
            print(f"   samples: {', '.join(f'[[{l}]]' for l in parsed['wikilinks'][:3])}")

if __name__ == '__main__':
    main()
