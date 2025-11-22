#!/usr/bin/env python3
"""
Fix incorrect draft→draught conversions.
Keep "draught" only in appropriate contexts (beer, air flow).
Revert to "draft" for documents, maturity levels, etc.
"""

import re
from pathlib import Path
import sys

def fix_draft_in_file(file_path: Path) -> bool:
    """Fix incorrect draught conversions in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content

        # Patterns where "draught" should be reverted to "draft"
        # 1. Maturity field
        content = re.sub(r'(\bmaturity::\s*)draught\b', r'\1draft', content)

        # 2. Status field
        content = re.sub(r'(\*\*Status\*\*:?\s*)draught\b', r'\1draft', content, flags=re.IGNORECASE)
        content = re.sub(r'(\bStatus:?\s*)draught\b', r'\1draft', content, flags=re.IGNORECASE)

        # 3. Maturity heading/field
        content = re.sub(r'(\*\*Maturity\*\*:?\s*)draught\b', r'\1draft', content, flags=re.IGNORECASE)
        content = re.sub(r'(\bMaturity:?\s*)draught\b', r'\1draft', content, flags=re.IGNORECASE)

        # 4. "drafting" (present participle - should always be draft)
        content = re.sub(r'\bdraughting\b', 'drafting', content, flags=re.IGNORECASE)

        # 5. "drafted" (past tense - should always be draft)
        content = re.sub(r'\bdraughted\b', 'drafted', content, flags=re.IGNORECASE)

        # 6. "draughter" → "drafter"
        content = re.sub(r'\bdraughter\b', 'drafter', content, flags=re.IGNORECASE)

        # 7. Common phrases with "draft" that should never be "draught"
        # Simple replacements first
        content = re.sub(r'\bfirst\s+draught\b', 'first draft', content, flags=re.IGNORECASE)
        content = re.sub(r'\binitial\s+draught\b', 'initial draft', content, flags=re.IGNORECASE)
        content = re.sub(r'\bfinal\s+draught\b', 'final draft', content, flags=re.IGNORECASE)
        content = re.sub(r'\brough\s+draught\b', 'rough draft', content, flags=re.IGNORECASE)
        content = re.sub(r'\bpreliminary\s+draught\b', 'preliminary draft', content, flags=re.IGNORECASE)
        content = re.sub(r'\bworking\s+draught\b', 'working draft', content, flags=re.IGNORECASE)

        # Patterns with capture groups
        content = re.sub(r'\ba\s+draught\s+(document|version|copy|text)\b', r'a draft \1', content, flags=re.IGNORECASE)
        content = re.sub(r'\bthe\s+draught\s+(document|version|copy|text|legislation|law|regulation)\b', r'the draft \1', content, flags=re.IGNORECASE)
        content = re.sub(r'\bdraughts\s+of\s+(reports|documents|contracts|agreements|proposals|plans)\b', r'drafts of \1', content, flags=re.IGNORECASE)

        phrases_to_fix = []

        for pattern, replacement in phrases_to_fix:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

        # Save if changes were made
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Fix draft conversions in all markdown files."""
    pages_dir = Path(__file__).parent.parent.parent / "mainKnowledgeGraph" / "pages"

    if not pages_dir.exists():
        print(f"Error: Directory not found: {pages_dir}", file=sys.stderr)
        sys.exit(1)

    markdown_files = list(pages_dir.rglob("*.md"))
    print(f"Found {len(markdown_files)} markdown files")

    fixed_count = 0
    for file_path in markdown_files:
        if fix_draft_in_file(file_path):
            fixed_count += 1
            print(f"✓ Fixed: {file_path.name}")

    print(f"\n✓ Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
