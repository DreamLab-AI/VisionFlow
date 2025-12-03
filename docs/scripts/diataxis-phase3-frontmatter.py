#!/usr/bin/env python3
"""
Diátaxis Phase 3: Add frontmatter to all markdown files
"""

import os
import re
from pathlib import Path
from datetime import datetime

# Determine document type based on path
def get_doc_type(filepath):
    path_str = str(filepath)
    if '/tutorials/' in path_str:
        return 'tutorial'
    elif '/guides/' in path_str:
        return 'guide'
    elif '/explanations/' in path_str:
        return 'explanation'
    elif '/reference/' in path_str:
        return 'reference'
    elif '/archive/' in path_str:
        return 'archive'
    else:
        return 'document'

# Extract title from first heading or filename
def extract_title(content, filename):
    # Try to find first # heading
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fall back to filename
    return filename.replace('-', ' ').replace('_', ' ').title()

# Extract first paragraph as description
def extract_description(content):
    # Remove frontmatter if exists
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

    # Find first paragraph (non-heading, non-empty)
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('#'):
            # Get this line and maybe the next few
            desc_lines = []
            for j in range(i, min(i+3, len(lines))):
                l = lines[j].strip()
                if l and not l.startswith('#'):
                    desc_lines.append(l)
                else:
                    break
            description = ' '.join(desc_lines)
            # Truncate to reasonable length
            if len(description) > 200:
                description = description[:197] + '...'
            return description
    return "Documentation"

# Check if file already has frontmatter
def has_frontmatter(content):
    return content.startswith('---\n')

# Add frontmatter to file
def add_frontmatter(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already has frontmatter
        if has_frontmatter(content):
            return False

        filename = filepath.stem
        doc_type = get_doc_type(filepath)
        title = extract_title(content, filename)
        description = extract_description(content)

        # Determine status
        status = 'stable'
        if '/archive/' in str(filepath):
            status = 'archived'
        elif '/working/' in str(filepath):
            status = 'draft'

        frontmatter = f"""---
title: {title}
description: {description}
type: {doc_type}
status: {status}
---

"""

        new_content = frontmatter + content

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    docs_dir = Path('docs')

    print("=== Diátaxis Phase 3: Adding Frontmatter ===\n")

    # Find all markdown files
    md_files = list(docs_dir.rglob('*.md'))

    added = 0
    skipped = 0

    for md_file in md_files:
        # Skip README files in root or specific directories
        if md_file.name == 'README.md' and md_file.parent == docs_dir:
            skipped += 1
            continue

        if add_frontmatter(md_file):
            print(f"✓ Added frontmatter: {md_file}")
            added += 1
        else:
            print(f"⊘ Skipped (has frontmatter): {md_file}")
            skipped += 1

    print(f"\n=== Summary ===")
    print(f"Frontmatter added: {added}")
    print(f"Skipped: {skipped}")
    print(f"Total files: {len(md_files)}")

if __name__ == '__main__':
    main()
