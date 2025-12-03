#!/usr/bin/env python3
"""
Diátaxis Phase 3b: Fix internal links after restructuring
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# Build file index
def build_file_index():
    """Build a map of filename -> new path"""
    docs_dir = Path('docs')
    file_index = {}

    for md_file in docs_dir.rglob('*.md'):
        # Store both the filename and the relative path
        rel_path = md_file.relative_to(docs_dir)
        filename = md_file.name

        # Store in index (handle duplicates by keeping the non-archive version)
        if filename in file_index:
            # Prefer non-archive paths
            if '/archive/' not in str(rel_path):
                file_index[filename] = rel_path
        else:
            file_index[filename] = rel_path

    return file_index

# Common link patterns that need fixing
LINK_REPLACEMENTS = {
    # Old architecture paths
    '../concepts/architecture/': '../../explanations/architecture/',
    '../architecture/': '../../explanations/architecture/',
    './concepts/architecture/': './explanations/architecture/',
    './architecture/': './explanations/architecture/',
    '/docs/concepts/architecture/': '/docs/explanations/architecture/',
    '/docs/architecture/': '/docs/explanations/architecture/',

    # Old API paths
    '../api/': '../../reference/api/',
    './api/': './reference/api/',
    '/docs/api/': '/docs/reference/api/',

    # Old features paths
    '../features/': '../../guides/features/',
    './features/': './guides/features/',
    '/docs/features/': '/docs/guides/features/',

    # Old getting-started paths
    '../getting-started/': '../../tutorials/',
    './getting-started/': './tutorials/',
    '/docs/getting-started/': '/docs/tutorials/',

    # Old concepts paths
    '../concepts/': '../../explanations/ontology/',
    './concepts/': './explanations/ontology/',
    '/docs/concepts/': '/docs/explanations/ontology/',

    # Specific file renames
    '02-first-graph-and-agents.md': '02-first-graph.md',
    'ONTOLOGY_ARCHITECTURE_ANALYSIS.md': 'ontology-analysis.md',
    'DEEPSEEK_API_VERIFIED.md': 'deepseek-verification.md',
    'DEEPSEEK_SKILL_DEPLOYMENT.md': 'deepseek-deployment.md',
    'PAGINATION_BUG_FIX.md': 'github-pagination-fix.md',
    'LOCAL_FILE_SYNC_STRATEGY.md': 'local-file-sync-strategy.md',
    'hexagonal-cqrs-architecture.md': 'hexagonal-cqrs.md',
    '00-architecture-overview.md': 'system-overview.md',
    '04-database-schemas.md': 'schemas.md',
    'ontology-reasoning.md': 'reasoning-engine.md',
    'binary-protocol-specification.md': 'binary-websocket.md',
}

def fix_links_in_file(filepath, file_index):
    """Fix internal links in a markdown file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        # Apply pattern replacements
        for old_pattern, new_pattern in LINK_REPLACEMENTS.items():
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                changes.append(f"  - Replaced: {old_pattern} -> {new_pattern}")

        # Find all markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

        def fix_link(match):
            text = match.group(1)
            url = match.group(2)

            # Skip external links
            if url.startswith('http://') or url.startswith('https://'):
                return match.group(0)

            # Skip anchors
            if url.startswith('#'):
                return match.group(0)

            return match.group(0)  # For now, just return as-is

        content = re.sub(link_pattern, fix_link, content)

        # Write back if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
        return False, []

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False, []

def main():
    docs_dir = Path('docs')

    print("=== Diátaxis Phase 3b: Fixing Internal Links ===\n")

    # Build file index
    print("Building file index...")
    file_index = build_file_index()
    print(f"Indexed {len(file_index)} files\n")

    # Find all markdown files
    md_files = list(docs_dir.rglob('*.md'))

    fixed = 0
    unchanged = 0

    for md_file in md_files:
        # Skip archive for now
        if '/archive/' in str(md_file):
            unchanged += 1
            continue

        changed, changes = fix_links_in_file(md_file, file_index)
        if changed:
            print(f"✓ Fixed links: {md_file}")
            for change in changes[:5]:  # Show first 5 changes
                print(change)
            if len(changes) > 5:
                print(f"  ... and {len(changes) - 5} more changes")
            fixed += 1
        else:
            unchanged += 1

    print(f"\n=== Summary ===")
    print(f"Files with fixed links: {fixed}")
    print(f"Files unchanged: {unchanged}")
    print(f"Total files processed: {len(md_files)}")

if __name__ == '__main__':
    main()
