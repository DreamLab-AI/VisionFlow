#!/usr/bin/env python3
"""
Identify and fix copy-paste errors in markdown files.
These are sections with generic content that doesn't match the file's domain/topic.
"""

import re
from pathlib import Path
import json
from typing import Dict, List, Tuple
import sys

# Problematic copy-paste patterns that appear in wrong contexts
COPYPASTE_PATTERNS = [
    # Generic blockchain content in non-blockchain files
    (r'(?:### )?Current Landscape[\s\S]{0,500}blockchain(?:[\s\S]{0,1000}decentrali[sz](?:ed|ation)[\s\S]{0,500})?', 'blockchain'),

    # Generic AI content in non-AI files
    (r'(?:### )?Current Landscape[\s\S]{0,500}machine learning[\s\S]{0,500}neural networks?[\s\S]{0,500}', 'ai'),

    # Generic metaverse content in non-metaverse files
    (r'(?:### )?Current Landscape[\s\S]{0,500}virtual worlds?[\s\S]{0,500}immersive experiences?[\s\S]{0,500}', 'metaverse'),

    # Generic robotics content in non-robotics files
    (r'(?:### )?Current Landscape[\s\S]{0,500}(?:autonomous|robotic) systems[\s\S]{0,500}sensors[\s\S]{0,500}', 'robotics'),
]

# Standard generic paragraphs to remove entirely
GENERIC_PARAGRAPHS = [
    r"The rapid evolution of distributed ledger technology has created new opportunities[\s\S]{0,500}?",
    r"Recent advances in artificial intelligence have transformed[\s\S]{0,500}?",
    r"The emergence of virtual worlds and immersive technologies[\s\S]{0,500}?",
    r"Blockchain technology's core innovation lies in[\s\S]{0,500}?",
]


def detect_domain_from_filename(filename: str) -> str:
    """Detect the expected domain from filename."""
    filename_lower = filename.lower()

    if filename_lower.startswith('ai-') or 'ai ' in filename_lower or 'artificial-intelligence' in filename_lower:
        return 'ai'
    elif filename_lower.startswith('bc-') or 'blockchain' in filename_lower or 'crypto' in filename_lower:
        return 'blockchain'
    elif filename_lower.startswith('rb-') or 'robot' in filename_lower:
        return 'robotics'
    elif filename_lower.startswith('mv-') or 'metaverse' in filename_lower or 'virtual-reality' in filename_lower or 'augmented-reality' in filename_lower:
        return 'metaverse'
    elif filename_lower.startswith('tc-') or 'telecollaboration' in filename_lower or 'collaboration' in filename_lower:
        return 'telecollaboration'
    elif filename_lower.startswith('dt-') or 'disruptive' in filename_lower:
        return 'disruptive-tech'

    return 'unknown'


def detect_ontology_domain(content: str) -> str:
    """Detect domain from ontology block."""
    # Look for owl:class or rdf:about with domain prefix
    if match := re.search(r'owl:class::\s*([a-z]+):', content, re.IGNORECASE):
        prefix = match.group(1).lower()
        if prefix == 'ai': return 'ai'
        elif prefix == 'bc': return 'blockchain'
        elif prefix == 'rb': return 'robotics'
        elif prefix == 'mv': return 'metaverse'
        elif prefix == 'tc': return 'telecollaboration'
        elif prefix == 'dt': return 'disruptive-tech'

    return 'unknown'


def has_copypaste_error(file_path: Path) -> Tuple[bool, List[str]]:
    """Check if file has copy-paste errors."""
    try:
        content = file_path.read_text(encoding='utf-8')

        # Detect expected domain
        filename_domain = detect_domain_from_filename(file_path.name)
        ontology_domain = detect_ontology_domain(content)
        expected_domain = ontology_domain if ontology_domain != 'unknown' else filename_domain

        issues = []

        # Check for mismatched domain content
        for pattern, pattern_domain in COPYPASTE_PATTERNS:
            if pattern_domain != expected_domain:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(f"Contains {pattern_domain} content in {expected_domain} file")

        # Check for generic paragraphs
        for generic_pattern in GENERIC_PARAGRAPHS:
            if re.search(generic_pattern, content, re.IGNORECASE):
                issues.append("Contains generic copied paragraph")

        return len(issues) > 0, issues

    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False, []


def remove_copypaste_sections(file_path: Path, dry_run: bool = True) -> bool:
    """Remove copy-paste sections from a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content

        # Detect expected domain
        filename_domain = detect_domain_from_filename(file_path.name)
        ontology_domain = detect_ontology_domain(content)
        expected_domain = ontology_domain if ontology_domain != 'unknown' else filename_domain

        # Remove Current Landscape sections with wrong domain content
        # Strategy: Remove entire Current Landscape section if it contains mismatched content
        if expected_domain != 'unknown':
            for pattern, pattern_domain in COPYPASTE_PATTERNS:
                if pattern_domain != expected_domain:
                    # Check if this pattern exists
                    if re.search(pattern, content, re.IGNORECASE):
                        # Remove the entire Current Landscape section
                        content = re.sub(
                            r'### Current Landscape\s*[\s\S]*?(?=###|\Z)',
                            '',
                            content,
                            count=1
                        )
                        break

        # Remove standalone generic paragraphs
        for generic_pattern in GENERIC_PARAGRAPHS:
            content = re.sub(generic_pattern, '', content, flags=re.IGNORECASE)

        # Clean up multiple blank lines
        content = re.sub(r'\n{4,}', '\n\n\n', content)

        # Save if changes were made
        if content != original:
            if not dry_run:
                file_path.write_text(content, encoding='utf-8')
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Find and optionally fix copy-paste errors."""
    import argparse

    parser = argparse.ArgumentParser(description='Detect and fix copy-paste errors')
    parser.add_argument('--directory', default='../../mainKnowledgeGraph/pages', help='Pages directory')
    parser.add_argument('--apply', action='store_true', help='Apply fixes (default: dry run)')
    parser.add_argument('--report', help='Output report JSON file')

    args = parser.parse_args()

    pages_dir = Path(args.directory)
    if not pages_dir.exists():
        print(f"Error: Directory not found: {pages_dir}", file=sys.stderr)
        sys.exit(1)

    markdown_files = list(pages_dir.rglob("*.md"))
    print(f"Scanning {len(markdown_files)} markdown files...")

    files_with_errors = []

    for file_path in markdown_files:
        has_error, issues = has_copypaste_error(file_path)
        if has_error:
            files_with_errors.append({
                'file': str(file_path.relative_to(pages_dir.parent.parent)),
                'issues': issues
            })

    print(f"\nFound {len(files_with_errors)} files with copy-paste errors\n")

    if files_with_errors:
        print("Files with copy-paste errors:")
        for item in files_with_errors[:10]:  # Show first 10
            print(f"  - {item['file']}")
            for issue in item['issues']:
                print(f"    • {issue}")

    if args.apply:
        print(f"\nApplying fixes...")
        fixed_count = 0
        for item in files_with_errors:
            file_path = Path(item['file'])
            if remove_copypaste_sections(file_path, dry_run=False):
                fixed_count += 1
                print(f"✓ Fixed: {file_path.name}")

        print(f"\n✓ Fixed {fixed_count} files")
    else:
        print(f"\nDry run mode. Use --apply to fix errors.")

    if args.report:
        report = {
            'total_files': len(markdown_files),
            'files_with_errors': len(files_with_errors),
            'files': files_with_errors
        }
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\n✓ Report saved to {args.report}")


if __name__ == "__main__":
    main()
