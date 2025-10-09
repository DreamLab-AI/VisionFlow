#!/usr/bin/env python3
"""
Generate topics.json from markdown filenames in the data/markdown directory.

This script extracts topic names from .md filenames, normalizes them,
and creates the topics.json file used by the web-summary-mcp-server
for semantic topic matching.

Usage:
    python3 generate-topics-from-markdown.py /path/to/data/markdown /path/to/output/topics.json
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Set


def normalize_filename_to_topic(filename: str) -> str:
    """
    Convert a markdown filename to a normalized topic string.

    Examples:
        "AI Video.md" -> "ai video"
        "Large language models.md" -> "large language models"
        "3D and 4D.md" -> "3d and 4d"
        "Stable Diffusion.md" -> "stable diffusion"
    """
    # Remove .md extension
    topic = filename.replace('.md', '')

    # Remove .backup extension if present
    topic = topic.replace('.backup', '')

    # Convert to lowercase for normalization
    topic = topic.lower()

    # Remove any remaining extensions or special suffixes
    topic = re.sub(r'\.(backup|tmp|old)$', '', topic)

    return topic.strip()


def should_exclude_topic(topic: str) -> bool:
    """
    Determine if a topic should be excluded from the list.

    Exclusion criteria:
    - Too generic (single words like "test", "debug")
    - Meta topics that aren't real content
    - System/technical topics
    """
    exclude_patterns = [
        r'^debug\b',
        r'^test\b',
        r'^introduction to me$',
        r'^suggested reading order$',
        r'^revision list$',
        r'^recent projects$',
        r'^project\s',  # Project-specific pages
    ]

    for pattern in exclude_patterns:
        if re.match(pattern, topic, re.IGNORECASE):
            return True

    return False


def extract_topics_from_directory(markdown_dir: Path) -> List[str]:
    """
    Extract all topics from markdown filenames in the directory.

    Args:
        markdown_dir: Path to directory containing .md files

    Returns:
        Sorted list of unique normalized topic strings
    """
    topics: Set[str] = set()

    if not markdown_dir.exists():
        print(f"Error: Directory not found: {markdown_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all .md files (excluding backups)
    md_files = [f for f in markdown_dir.glob('*.md') if not f.name.endswith('.backup')]

    print(f"Found {len(md_files)} markdown files in {markdown_dir}", file=sys.stderr)

    for md_file in md_files:
        filename = md_file.name
        topic = normalize_filename_to_topic(filename)

        # Skip excluded topics
        if should_exclude_topic(topic):
            print(f"  Excluding: {filename} -> {topic}", file=sys.stderr)
            continue

        topics.add(topic)
        print(f"  Adding: {filename} -> {topic}", file=sys.stderr)

    # Sort topics alphabetically for consistency
    return sorted(list(topics))


def generate_topics_json(topics: List[str], output_path: Path) -> None:
    """
    Generate topics.json file with the extracted topics.

    Args:
        topics: List of topic strings
        output_path: Path to write topics.json
    """
    topics_data = {
        "topics": topics,
        "metadata": {
            "count": len(topics),
            "generated_from": "markdown filenames",
            "description": "Automatically generated topic list for web-summary-mcp-server semantic matching"
        }
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON with proper formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(topics_data, f, indent=2, ensure_ascii=False)
        f.write('\n')  # Add trailing newline

    print(f"\nGenerated {output_path} with {len(topics)} topics", file=sys.stderr)


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python3 generate-topics-from-markdown.py <markdown_dir> <output_json>", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python3 generate-topics-from-markdown.py ../data/markdown ./core-assets/config/topics.json", file=sys.stderr)
        sys.exit(1)

    markdown_dir = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()

    print(f"Parsing markdown directory: {markdown_dir}", file=sys.stderr)
    print(f"Output file: {output_path}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    # Extract topics
    topics = extract_topics_from_directory(markdown_dir)

    if not topics:
        print("\nError: No topics found!", file=sys.stderr)
        sys.exit(1)

    # Generate JSON
    generate_topics_json(topics, output_path)

    # Print summary
    print("-" * 60, file=sys.stderr)
    print(f"Success! Generated {len(topics)} topics:", file=sys.stderr)
    print(f"  First 10: {', '.join(topics[:10])}", file=sys.stderr)
    if len(topics) > 10:
        print(f"  ... and {len(topics) - 10} more", file=sys.stderr)


if __name__ == '__main__':
    main()
