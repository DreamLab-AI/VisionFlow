#!/usr/bin/env python3
"""
Automated Link Update Script for Documentation Restructure

This script updates internal links across all markdown files to reflect
the new documentation structure following Phase 4 of the Diataxis migration.

Usage:
    python update-links.py [--dry-run] [--verbose]

Options:
    --dry-run   Show what would be changed without making changes
    --verbose   Show detailed information about each change
"""

import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# Configuration
DOCS_ROOT = Path(__file__).parent.parent
REDIRECT_MAP_PATH = DOCS_ROOT / ".analysis" / "redirect-map.json"

# Link patterns to find markdown links
LINK_PATTERN = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
REFERENCE_LINK_PATTERN = re.compile(r'\[([^\]]+)\]:\s*(.+)')

# Statistics tracking
stats = {
    "files_processed": 0,
    "files_modified": 0,
    "links_updated": 0,
    "broken_links_found": [],
    "changes": []
}

def load_redirect_map() -> Dict:
    """Load the redirect map from JSON file."""
    if REDIRECT_MAP_PATH.exists():
        with open(REDIRECT_MAP_PATH, 'r') as f:
            return json.load(f)
    return {"patterns": {}}

def build_replacement_rules(redirect_map: Dict) -> List[Tuple[re.Pattern, str, str]]:
    """Build regex replacement rules from the redirect map."""
    rules = []

    patterns = redirect_map.get("patterns", {})

    # ADR path changes
    if "adr_paths" in patterns:
        for mapping in patterns["adr_paths"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # SCREAMING_CASE to kebab-case (context-aware)
    if "screaming_case" in patterns:
        for mapping in patterns["screaming_case"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            context = mapping.get("context", "")
            if context:
                # Match with context prefix
                pattern = re.compile(rf'({re.escape(context)}){old}')
                rules.append((pattern, rf'\g<1>{new}', mapping.get("reason", "")))
            else:
                rules.append((re.compile(old), new, mapping.get("reason", "")))

    # Reference restructure
    if "reference_restructure" in patterns:
        for mapping in patterns["reference_restructure"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # Case sensitivity fixes
    if "case_sensitivity" in patterns:
        for mapping in patterns["case_sensitivity"]["mappings"]:
            old = mapping["old"]
            new = mapping["new"]
            rules.append((re.compile(old, re.IGNORECASE), new, mapping.get("reason", "")))

    # Architecture moves
    if "architecture_moves" in patterns:
        for mapping in patterns["architecture_moves"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # Path normalization
    if "path_normalization" in patterns:
        for mapping in patterns["path_normalization"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # Explanations to architecture/concepts
    if "explanations_architecture" in patterns:
        for mapping in patterns["explanations_architecture"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # Explanations to concepts
    if "explanations_to_concepts" in patterns:
        for mapping in patterns["explanations_to_concepts"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # Ontology moves
    if "ontology_moves" in patterns:
        for mapping in patterns["ontology_moves"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # Physics moves
    if "physics_moves" in patterns:
        for mapping in patterns["physics_moves"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # DeepSeek location
    if "deepseek_location" in patterns:
        for mapping in patterns["deepseek_location"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    # Getting started/tutorials
    if "getting_started_tutorials" in patterns:
        for mapping in patterns["getting_started_tutorials"]["mappings"]:
            old = re.escape(mapping["old"])
            new = mapping["new"]
            rules.append((re.compile(old), new, mapping.get("reason", "")))

    return rules

def get_hardcoded_rules() -> List[Tuple[re.Pattern, str, str]]:
    """Return hardcoded replacement rules for common patterns."""
    return [
        # Case sensitivity - readme.md to README.md
        (re.compile(r'/readme\.md', re.IGNORECASE), '/README.md', 'Case sensitivity fix'),
        (re.compile(r'\(readme\.md\)', re.IGNORECASE), '(README.md)', 'Case sensitivity fix'),

        # Double docs/ prefix - comprehensive patterns
        (re.compile(r'\(docs/diagrams/'), '(diagrams/', 'Remove doubled docs/ prefix'),
        (re.compile(r'\(docs/docs/'), '(docs/', 'Remove doubled docs/ prefix'),
        (re.compile(r'\(docs/getting-started/'), '(getting-started/', 'Remove doubled docs/ prefix'),
        (re.compile(r'\(docs/guides/'), '(guides/', 'Remove doubled docs/ prefix'),
        (re.compile(r'\(docs/reference/'), '(reference/', 'Remove doubled docs/ prefix'),
        (re.compile(r'\(docs/architecture/'), '(architecture/', 'Remove doubled docs/ prefix'),
        (re.compile(r'\(docs/concepts/'), '(concepts/', 'Remove doubled docs/ prefix'),
        (re.compile(r'\(docs/\)'), '(./', 'Remove doubled docs/ prefix'),

        # Explanations to concepts - comprehensive mapping
        (re.compile(r'explanations/architecture/hexagonal-cqrs\.md'), 'concepts/hexagonal-architecture.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/database-architecture\.md'), 'architecture/database.md', 'Architecture consolidation'),
        (re.compile(r'explanations/architecture/adapter-patterns\.md'), 'concepts/adapter-patterns.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/analytics-visualization\.md'), 'concepts/analytics-visualization.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/event-driven-architecture\.md'), 'concepts/event-driven-architecture.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/gpu-semantic-forces\.md'), 'concepts/gpu-semantic-forces.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/hierarchical-visualization\.md'), 'concepts/hierarchical-visualization.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/integration-patterns\.md'), 'concepts/integration-patterns.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/ontology-analysis\.md'), 'concepts/ontology-analysis.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/ontology-physics-integration\.md'), 'concepts/ontology-physics-integration.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/ontology-reasoning-pipeline\.md'), 'concepts/ontology-reasoning-pipeline.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/ontology-storage-architecture\.md'), 'concepts/ontology-storage.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/pipeline-integration\.md'), 'concepts/pipeline-integration.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/pipeline-sequence-diagrams\.md'), 'concepts/pipeline-sequence.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/reasoning-data-flow\.md'), 'concepts/reasoning-data-flow.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/ruvector-integration\.md'), 'concepts/ruvector-integration.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/semantic-forces-system\.md'), 'concepts/semantic-forces-system.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/semantic-physics-system\.md'), 'concepts/semantic-physics-system.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/semantic-physics\.md'), 'concepts/semantic-physics.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/services-layer\.md'), 'concepts/services-layer.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/stress-majorization\.md'), 'concepts/stress-majorization.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/xr-immersive-system\.md'), 'concepts/xr-immersive-system.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/github-sync-service-design\.md'), 'concepts/github-sync-service.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/cqrs-directive-template\.md'), 'concepts/cqrs-directive.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/api-handlers-reference\.md'), 'reference/api/handlers.md', 'API reference'),
        (re.compile(r'explanations/architecture/quick-reference\.md'), 'concepts/quick-reference.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/reasoning-tests-summary\.md'), 'concepts/reasoning-tests.md', 'Explanations to concepts'),
        (re.compile(r'explanations/architecture/README\.md'), 'concepts/README.md', 'Explanations to concepts'),

        # Legacy architecture file redirects
        (re.compile(r'ARCHITECTURE_OVERVIEW\.md'), 'architecture/overview.md', 'Legacy redirect'),
        (re.compile(r'ARCHITECTURE_COMPLETE\.md'), 'architecture/overview.md', 'Legacy redirect'),
        (re.compile(r'DEVELOPER_JOURNEY\.md'), 'architecture/developer-journey.md', 'Legacy redirect'),

        # ADR path standardization
        (re.compile(r'explanations/architecture/decisions/(\d+)-'), r'architecture/adr/ADR-\1-', 'ADR path standardization'),

        # Reference files SCREAMING_CASE to kebab-case (within reference/)
        (re.compile(r'(reference/)API_REFERENCE\.md'), r'\1api/README.md', 'Reference restructure'),
        (re.compile(r'(reference/)CONFIGURATION_REFERENCE\.md'), r'\1configuration/README.md', 'Reference restructure'),
        (re.compile(r'(reference/)PROTOCOL_REFERENCE\.md'), r'\1protocols/README.md', 'Reference restructure'),
        (re.compile(r'(reference/)DATABASE_SCHEMA_REFERENCE\.md'), r'\1database/README.md', 'Reference restructure'),
        (re.compile(r'(reference/)ERROR_REFERENCE\.md'), r'\1error-codes.md', 'Reference restructure'),
        (re.compile(r'(reference/)INDEX\.md'), r'\1README.md', 'Standard naming'),

        # Standalone reference file names (when in same directory)
        (re.compile(r'\./API_REFERENCE\.md'), './api/README.md', 'Reference restructure'),
        (re.compile(r'\./CONFIGURATION_REFERENCE\.md'), './configuration/README.md', 'Reference restructure'),
        (re.compile(r'\./PROTOCOL_REFERENCE\.md'), './protocols/README.md', 'Reference restructure'),
        (re.compile(r'\./DATABASE_SCHEMA_REFERENCE\.md'), './database/README.md', 'Reference restructure'),
        (re.compile(r'\./ERROR_REFERENCE\.md'), './error-codes.md', 'Reference restructure'),

        # Relative paths without ./
        (re.compile(r'\(API_REFERENCE\.md'), '(api/README.md', 'Reference restructure'),
        (re.compile(r'\(CONFIGURATION_REFERENCE\.md'), '(configuration/README.md', 'Reference restructure'),
        (re.compile(r'\(PROTOCOL_REFERENCE\.md'), '(protocols/README.md', 'Reference restructure'),
        (re.compile(r'\(DATABASE_SCHEMA_REFERENCE\.md'), '(database/README.md', 'Reference restructure'),
        (re.compile(r'\(ERROR_REFERENCE\.md'), '(error-codes.md', 'Reference restructure'),

        # Parent directory references
        (re.compile(r'\.\./API_REFERENCE\.md'), '../api/README.md', 'Reference restructure'),
        (re.compile(r'\.\./CONFIGURATION_REFERENCE\.md'), '../configuration/README.md', 'Reference restructure'),
        (re.compile(r'\.\./PROTOCOL_REFERENCE\.md'), '../protocols/README.md', 'Reference restructure'),
        (re.compile(r'\.\./DATABASE_SCHEMA_REFERENCE\.md'), '../database/README.md', 'Reference restructure'),
        (re.compile(r'\.\./ERROR_REFERENCE\.md'), '../error-codes.md', 'Reference restructure'),

        # Architecture SCREAMING_CASE
        (re.compile(r'architecture/HEXAGONAL_ARCHITECTURE_STATUS\.md'), 'concepts/hexagonal-architecture.md', 'Conceptual content'),
        (re.compile(r'architecture/PROTOCOL_MATRIX\.md'), 'reference/protocols/matrix.md', 'Reference content'),

        # Explanations to architecture consolidation
        (re.compile(r'explanations/architecture/gpu/'), 'architecture/gpu/', 'GPU architecture consolidation'),
        (re.compile(r'explanations/architecture/core/client\.md'), 'architecture/client/overview.md', 'Architecture consolidation'),
        (re.compile(r'explanations/architecture/core/server\.md'), 'architecture/server/overview.md', 'Architecture consolidation'),
        (re.compile(r'explanations/architecture/components/websocket-protocol\.md'), 'architecture/protocols/websocket.md', 'Protocol documentation'),

        # DeepSeek location fix
        (re.compile(r'guides/features/deepseek-'), 'guides/ai-models/deepseek-', 'DeepSeek location fix'),

        # Getting started path fixes
        (re.compile(r'getting-started/01-installation\.md'), 'getting-started/installation.md', 'Remove numbered prefix'),
        (re.compile(r'getting-started/02-first-graph-and-agents\.md'), 'getting-started/first-graph.md', 'Simplify name'),
        (re.compile(r'getting-started/02-first-graph\.md'), 'getting-started/first-graph.md', 'Simplify name'),
        (re.compile(r'tutorials/01-installation\.md'), 'getting-started/installation.md', 'Tutorials to getting-started'),
        (re.compile(r'tutorials/02-first-graph\.md'), 'getting-started/first-graph.md', 'Tutorials to getting-started'),
    ]

def process_file(filepath: Path, rules: List[Tuple[re.Pattern, str, str]],
                 dry_run: bool = False, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Process a single markdown file and update links."""
    changes = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False, []

    original_content = content

    for pattern, replacement, reason in rules:
        matches = pattern.findall(content)
        if matches:
            new_content = pattern.sub(replacement, content)
            if new_content != content:
                for match in matches:
                    if isinstance(match, tuple):
                        match = ''.join(match)
                    new_text = pattern.sub(replacement, match if isinstance(match, str) else str(match))
                    changes.append(f"  {match} -> {new_text} ({reason})")
                content = new_content

    if content != original_content:
        if not dry_run:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                print(f"Error writing {filepath}: {e}")
                return False, changes
        return True, changes

    return False, changes

def check_link_exists(link: str, source_file: Path) -> bool:
    """Check if a linked file exists."""
    if link.startswith('http://') or link.startswith('https://'):
        return True  # Skip external links
    if link.startswith('#'):
        return True  # Skip anchor-only links

    # Remove anchor from link
    link_path = link.split('#')[0]
    if not link_path:
        return True

    # Resolve relative path
    if link_path.startswith('/'):
        target = DOCS_ROOT / link_path.lstrip('/')
    else:
        target = source_file.parent / link_path

    return target.exists()

def find_all_links(filepath: Path) -> List[Tuple[str, str]]:
    """Find all markdown links in a file."""
    links = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find inline links [text](url)
        for match in LINK_PATTERN.finditer(content):
            links.append((match.group(1), match.group(2)))

        # Find reference links [text]: url
        for match in REFERENCE_LINK_PATTERN.finditer(content):
            links.append((match.group(1), match.group(2)))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return links

def verify_links(verbose: bool = False) -> List[Tuple[Path, str]]:
    """Verify all links and return broken ones."""
    broken = []

    for filepath in DOCS_ROOT.rglob("*.md"):
        if ".analysis" in str(filepath) or "scripts" in str(filepath):
            continue

        links = find_all_links(filepath)
        for text, url in links:
            if not check_link_exists(url, filepath):
                broken.append((filepath, url))
                if verbose:
                    print(f"Broken: {filepath.relative_to(DOCS_ROOT)} -> {url}")

    return broken

def main():
    dry_run = "--dry-run" in sys.argv
    verbose = "--verbose" in sys.argv

    print("=" * 60)
    print("Documentation Link Update Script")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Docs root: {DOCS_ROOT}")
    print()

    # Load redirect map and build rules
    redirect_map = load_redirect_map()
    rules_from_map = build_replacement_rules(redirect_map)
    hardcoded_rules = get_hardcoded_rules()

    # Combine rules (hardcoded take precedence for specificity)
    all_rules = hardcoded_rules + rules_from_map

    print(f"Loaded {len(all_rules)} replacement rules")
    print()

    # Process all markdown files
    print("Processing files...")
    print("-" * 40)

    files_modified = 0
    total_changes = 0
    all_changes = []

    for filepath in sorted(DOCS_ROOT.rglob("*.md")):
        # Skip analysis and script directories
        rel_path = filepath.relative_to(DOCS_ROOT)
        if str(rel_path).startswith('.analysis') or str(rel_path).startswith('scripts'):
            continue

        stats["files_processed"] += 1
        modified, changes = process_file(filepath, all_rules, dry_run, verbose)

        if modified:
            files_modified += 1
            total_changes += len(changes)
            all_changes.append((rel_path, changes))

            if verbose:
                print(f"\n{rel_path}:")
                for change in changes:
                    print(change)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files modified: {files_modified}")
    print(f"Links updated: {total_changes}")

    if all_changes and not verbose:
        print()
        print("Modified files:")
        for rel_path, changes in all_changes[:20]:  # Show first 20
            print(f"  {rel_path} ({len(changes)} changes)")
        if len(all_changes) > 20:
            print(f"  ... and {len(all_changes) - 20} more files")

    # Verify links after update
    print()
    print("Verifying links...")
    broken_links = verify_links(verbose)

    if broken_links:
        print()
        print(f"Found {len(broken_links)} broken links:")
        shown = 0
        for filepath, url in broken_links[:30]:  # Show first 30
            print(f"  {filepath.relative_to(DOCS_ROOT)}: {url}")
            shown += 1
        if len(broken_links) > 30:
            print(f"  ... and {len(broken_links) - 30} more")
    else:
        print("All links verified successfully!")

    print()
    print("=" * 60)

    return 0 if not broken_links else 1

if __name__ == "__main__":
    sys.exit(main())
