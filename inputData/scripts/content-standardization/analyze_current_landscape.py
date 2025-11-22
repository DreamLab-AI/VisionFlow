#!/usr/bin/env python3
"""
Analyze Current Landscape sections for quality and detect template copy-paste.
"""

import re
from pathlib import Path
from collections import Counter
import hashlib
import json

def extract_current_landscape(content: str) -> str:
    """Extract Current Landscape section."""
    match = re.search(
        r'##\s+Current Landscape.*?\n(.*?)(?=\n##\s+|\Z)',
        content,
        re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else ""


def is_generic_template(section: str) -> bool:
    """Check if section is generic template text."""
    # Generic template indicators
    generic_phrases = [
        r'\[.*?\]',  # Template placeholders like [Year], [Example]
        r'Industry adoption (?:is )?growing',
        r'(?:Manchester|Leeds|Newcastle|Sheffield).*(?:Manchester|Leeds|Newcastle|Sheffield).*(?:Manchester|Leeds|Newcastle|Sheffield)',  # Lists cities without content
        r'Standards.*development.*progress',
        r'Recent.*advances.*transformed?',
        r'The emergence of.*has created',
    ]

    matches = sum(1 for pattern in generic_phrases if re.search(pattern, section, re.IGNORECASE))

    # If 3+ generic indicators, likely template
    # If very short (<100 chars), likely placeholder
    return matches >= 3 or (len(section) < 100 and '[' in section)


def section_hash(section: str) -> str:
    """Get hash of normalized section for duplicate detection."""
    # Normalize whitespace and capitalization for comparison
    normalized = ' '.join(section.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:8]


def analyze_corpus(pages_dir: Path):
    """Analyze all Current Landscape sections."""
    markdown_files = list(pages_dir.rglob("*.md"))

    sections = {}
    hashes = Counter()
    generic_files = []
    duplicate_groups = {}

    for file_path in markdown_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            section = extract_current_landscape(content)

            if section:
                # Store section
                rel_path = str(file_path.relative_to(pages_dir.parent.parent))
                sections[rel_path] = section

                # Check for generic template
                if is_generic_template(section):
                    generic_files.append(rel_path)

                # Track duplicates
                section_id = section_hash(section)
                hashes[section_id] += 1

                if section_id not in duplicate_groups:
                    duplicate_groups[section_id] = []
                duplicate_groups[section_id].append(rel_path)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Find duplicate groups (3+ files with same content)
    significant_duplicates = {
        h: files for h, files in duplicate_groups.items()
        if len(files) >= 3
    }

    return {
        'total_files_with_section': len(sections),
        'generic_template_files': generic_files,
        'duplicate_groups': significant_duplicates,
        'sections': sections
    }


def main():
    pages_dir = Path(__file__).parent.parent.parent / "mainKnowledgeGraph" / "pages"

    print("Analyzing Current Landscape sections...")
    results = analyze_corpus(pages_dir)

    print(f"\nFiles with Current Landscape: {results['total_files_with_section']}")
    print(f"Generic/template sections: {len(results['generic_template_files'])}")
    print(f"Duplicate groups: {len(results['duplicate_groups'])}\n")

    if results['generic_template_files']:
        print("Generic template files (first 20):")
        for filepath in results['generic_template_files'][:20]:
            print(f"  - {filepath}")

    if results['duplicate_groups']:
        print(f"\nDuplicate Current Landscape sections:")
        for hash_id, files in list(results['duplicate_groups'].items())[:5]:
            print(f"\n  Group {hash_id} ({len(files)} files):")
            for filepath in files[:5]:
                print(f"    - {filepath}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")

    # Save detailed report
    report_path = Path(__file__).parent / "reports" / "current_landscape_analysis.json"
    report_path.parent.mkdir(exist_ok=True, parents=True)

    report = {
        'summary': {
            'total_files': results['total_files_with_section'],
            'generic_templates': len(results['generic_template_files']),
            'duplicate_groups': len(results['duplicate_groups'])
        },
        'generic_files': results['generic_template_files'],
        'duplicate_groups': {
            hash_id: files
            for hash_id, files in results['duplicate_groups'].items()
        }
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n✓ Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
