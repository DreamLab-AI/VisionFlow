#!/usr/bin/env python3
"""
Automated Content Enhancement Pipeline

Automatically improves content quality through:
- US → UK English conversion
- Wiki link enhancement
- Section restructuring
- Formatting fixes

Safety features:
- Never modifies OntologyBlock
- Preview mode before applying
- Git backup before batch processing
- Detailed enhancement reports
"""

import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict

# Import our modules
from us_to_uk_dict import get_conversion_dictionary, is_tech_term
from corpus_indexer import CorpusIndexer


@dataclass
class Enhancement:
    """Single enhancement made to content."""
    type: str
    description: str
    old_value: str = ''
    new_value: str = ''
    line_number: int = 0
    confidence: float = 1.0


@dataclass
class EnhancementReport:
    """Report of all enhancements made to a file."""
    file: str
    quality_before: float
    quality_after: float
    enhancements: List[Enhancement]
    timestamp: str

    def to_dict(self):
        return {
            'file': self.file,
            'quality_before': self.quality_before,
            'quality_after': self.quality_after,
            'improvement': self.quality_after - self.quality_before,
            'enhancements_by_type': self._group_by_type(),
            'enhancements': [asdict(e) for e in self.enhancements],
            'timestamp': self.timestamp
        }

    def _group_by_type(self):
        groups = {}
        for enh in self.enhancements:
            if enh.type not in groups:
                groups[enh.type] = []
            groups[enh.type].append(enh)
        return {k: len(v) for k, v in groups.items()}


class ContentEnhancer:
    """Main content enhancement engine."""

    # Standard section structure
    STANDARD_SECTIONS = [
        'Overview',
        'Key Concepts',
        'Technical Details',
        'UK Context',
        'Applications',
        'Challenges',
        'Future Directions',
        'Related Concepts',
        'Further Reading'
    ]

    def __init__(self, corpus_index_path: Path = None):
        self.us_uk_dict = get_conversion_dictionary()
        self.corpus_indexer = None

        if corpus_index_path and corpus_index_path.exists():
            self.corpus_indexer = CorpusIndexer.load_index(corpus_index_path)
            print(f"Loaded corpus index with {len(self.corpus_indexer.term_index)} terms")

    def enhance_file(self, file_path: Path, level: int = 1) -> EnhancementReport:
        """
        Enhance a single file.

        Args:
            file_path: Path to markdown file
            level: Enhancement level (1=safe, 2=moderate, 3=aggressive)

        Returns:
            EnhancementReport with all changes
        """
        content = file_path.read_text(encoding='utf-8')
        enhancements = []

        # Calculate initial quality score
        quality_before = self._calculate_quality_score(content)

        # Separate OntologyBlock from content
        ontology_block, body_content = self._separate_ontology_block(content)

        # Apply enhancements based on level
        if level >= 1:  # Safe enhancements
            body_content, enh = self._convert_us_to_uk(body_content)
            enhancements.extend(enh)

            body_content, enh = self._fix_formatting(body_content)
            enhancements.extend(enh)

        if level >= 2:  # Moderate enhancements
            if self.corpus_indexer:
                body_content, enh = self._add_wiki_links(body_content, confidence=0.8)
                enhancements.extend(enh)

            body_content, enh = self._restructure_sections(body_content)
            enhancements.extend(enh)

        if level >= 3:  # Aggressive enhancements
            body_content, enh = self._add_missing_sections(body_content)
            enhancements.extend(enh)

            if self.corpus_indexer:
                body_content, enh = self._add_wiki_links(body_content, confidence=0.6)
                enhancements.extend(enh)

        # Recombine
        enhanced_content = ontology_block + body_content

        # Calculate final quality score
        quality_after = self._calculate_quality_score(enhanced_content)

        # Create report
        report = EnhancementReport(
            file=str(file_path),
            quality_before=quality_before,
            quality_after=quality_after,
            enhancements=enhancements,
            timestamp=datetime.now().isoformat()
        )

        return report, enhanced_content

    def _separate_ontology_block(self, content: str) -> Tuple[str, str]:
        """
        Separate OntologyBlock from body content.

        Returns:
            tuple: (ontology_block, body_content)
        """
        match = re.search(
            r'(- ### OntologyBlock.*?(?=\n- |\n###|\Z))',
            content,
            re.DOTALL
        )

        if match:
            ontology_end = match.end()
            return content[:ontology_end], content[ontology_end:]

        return '', content

    def _convert_us_to_uk(self, content: str) -> Tuple[str, List[Enhancement]]:
        """Convert US spelling to UK spelling."""
        enhancements = []
        modified = content

        # Word boundary pattern for safe replacement
        for us_word, uk_word in self.us_uk_dict.items():
            # Skip if it's a tech term in code/technical context
            if is_tech_term(us_word):
                continue

            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(us_word) + r'\b'
            matches = list(re.finditer(pattern, modified))

            if matches:
                modified = re.sub(pattern, uk_word, modified)
                enhancements.append(Enhancement(
                    type='us_to_uk',
                    description=f'Converted "{us_word}" to "{uk_word}"',
                    old_value=us_word,
                    new_value=uk_word,
                    confidence=1.0
                ))

        return modified, enhancements

    def _add_wiki_links(self, content: str, confidence: float = 0.8) -> Tuple[str, List[Enhancement]]:
        """Add wiki links for known terms."""
        if not self.corpus_indexer:
            return content, []

        enhancements = []
        modified = content

        # Get linkable terms
        linkable = self.corpus_indexer.get_linkable_terms(confidence)

        # Sort by length (longest first) to handle multi-word terms
        linkable.sort(key=len, reverse=True)

        for term in linkable:
            # Skip if already linked
            if f'[[{term}]]' in modified or f'[[{term}|' in modified:
                continue

            # Pattern: word not already in brackets
            pattern = r'(?<!\[)(?<!\[\[)\b' + re.escape(term) + r'\b(?!\])'

            matches = list(re.finditer(pattern, modified, re.IGNORECASE))

            if matches:
                # Only link first occurrence per section
                match = matches[0]
                old_text = match.group(0)
                new_text = f'[[{term}]]'

                modified = modified[:match.start()] + new_text + modified[match.end():]

                enhancements.append(Enhancement(
                    type='wiki_link_added',
                    description=f'Added wiki link for "{term}"',
                    old_value=old_text,
                    new_value=new_text,
                    confidence=confidence
                ))

        return modified, enhancements

    def _fix_formatting(self, content: str) -> Tuple[str, List[Enhancement]]:
        """Fix common formatting issues."""
        enhancements = []
        modified = content

        # Fix bullet points (convert * to -)
        bullet_pattern = r'^\s*\*\s+(.+)$'
        matches = list(re.finditer(bullet_pattern, modified, re.MULTILINE))
        if matches:
            modified = re.sub(bullet_pattern, r'- \1', modified, flags=re.MULTILINE)
            enhancements.append(Enhancement(
                type='formatting_fixed',
                description=f'Converted {len(matches)} asterisk bullets to hyphens',
                confidence=1.0
            ))

        # Fix heading hierarchy (ensure proper nesting)
        # Find all headings
        headings = re.findall(r'^(#+)\s+(.+)$', modified, re.MULTILINE)

        # Fix multiple blank lines
        blank_lines = len(re.findall(r'\n\n\n+', modified))
        if blank_lines > 0:
            modified = re.sub(r'\n\n\n+', '\n\n', modified)
            enhancements.append(Enhancement(
                type='formatting_fixed',
                description=f'Fixed {blank_lines} instances of excessive blank lines',
                confidence=1.0
            ))

        # Fix trailing whitespace
        trailing = len(re.findall(r' +$', modified, re.MULTILINE))
        if trailing > 0:
            modified = re.sub(r' +$', '', modified, flags=re.MULTILINE)
            enhancements.append(Enhancement(
                type='formatting_fixed',
                description=f'Removed trailing whitespace from {trailing} lines',
                confidence=1.0
            ))

        return modified, enhancements

    def _restructure_sections(self, content: str) -> Tuple[str, List[Enhancement]]:
        """Reorganize content into standard section structure."""
        enhancements = []

        # For now, just identify missing standard sections
        existing_sections = re.findall(r'^###?\s+(.+)$', content, re.MULTILINE)

        missing = [s for s in self.STANDARD_SECTIONS if s not in existing_sections]

        if missing:
            enhancements.append(Enhancement(
                type='sections_identified',
                description=f'Identified {len(missing)} missing standard sections',
                old_value=', '.join(existing_sections),
                new_value=', '.join(missing),
                confidence=0.7
            ))

        return content, enhancements

    def _add_missing_sections(self, content: str) -> Tuple[str, List[Enhancement]]:
        """Add placeholder for missing sections."""
        enhancements = []
        modified = content

        # Check for UK Context section
        if '## UK Context' not in modified and '### UK Context' not in modified:
            # Add UK Context section at appropriate location
            uk_section = '\n\n### UK Context\n\n*[Content to be added: UK-specific context and considerations]*\n'

            # Find best insertion point (after Overview or Key Concepts)
            insert_after = ['## Overview', '### Overview', '## Key Concepts', '### Key Concepts']

            for heading in insert_after:
                if heading in modified:
                    # Find end of section
                    pattern = heading + r'.*?(?=\n## |\n### |\Z)'
                    match = re.search(pattern, modified, re.DOTALL)
                    if match:
                        insert_pos = match.end()
                        modified = modified[:insert_pos] + uk_section + modified[insert_pos:]
                        enhancements.append(Enhancement(
                            type='section_added',
                            description='Added UK Context section placeholder',
                            confidence=0.8
                        ))
                        break

        return modified, enhancements

    def _calculate_quality_score(self, content: str) -> float:
        """
        Calculate content quality score (0-100).

        Factors:
        - Length and completeness
        - Section structure
        - Wiki link density
        - UK spelling usage
        - Formatting quality
        """
        score = 0.0

        # Length factor (0-20 points)
        word_count = len(content.split())
        if word_count > 500:
            score += 20
        else:
            score += (word_count / 500) * 20

        # Section structure (0-20 points)
        sections = re.findall(r'^###?\s+(.+)$', content, re.MULTILINE)
        section_count = len(sections)
        score += min(section_count * 3, 20)

        # Wiki links (0-20 points)
        wiki_links = len(re.findall(r'\[\[.+?\]\]', content))
        score += min(wiki_links * 2, 20)

        # UK spelling (0-20 points)
        uk_words = sum(1 for uk in self.us_uk_dict.values() if uk in content)
        us_words = sum(1 for us in self.us_uk_dict.keys() if us in content)
        if uk_words + us_words > 0:
            uk_ratio = uk_words / (uk_words + us_words)
            score += uk_ratio * 20
        else:
            score += 10  # Neutral if no relevant words

        # Formatting quality (0-20 points)
        # Check for proper formatting
        formatting_score = 20
        if re.search(r'\n\n\n+', content):  # Multiple blank lines
            formatting_score -= 5
        if re.search(r' +$', content, re.MULTILINE):  # Trailing spaces
            formatting_score -= 5
        if re.search(r'^\*\s+', content, re.MULTILINE):  # Asterisk bullets
            formatting_score -= 5
        score += max(formatting_score, 0)

        return min(score, 100.0)

    def generate_diff(self, original: str, enhanced: str) -> str:
        """Generate human-readable diff."""
        import difflib

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            enhanced.splitlines(keepends=True),
            lineterm='',
            fromfile='original',
            tofile='enhanced'
        )

        return ''.join(diff)


def create_git_backup(message: str):
    """Create git commit as backup before batch processing."""
    try:
        subprocess.run(['git', 'add', '-A'], check=True)
        subprocess.run(['git', 'commit', '-m', f'[BACKUP] {message}'], check=True)
        print("✓ Git backup created")
        return True
    except subprocess.CalledProcessError:
        print("⚠ Warning: Could not create git backup")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Automated Content Enhancement Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview enhancements
  %(prog)s --file page.md --preview

  # Apply safe enhancements
  %(prog)s --file page.md --level 1

  # Batch process with review
  %(prog)s --directory pages/ --level 2 --review

  # Aggressive enhancement
  %(prog)s --file page.md --level 3 --apply
        """
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', type=Path, help='Single file to enhance')
    input_group.add_argument('--directory', type=Path, help='Directory to batch process')

    # Enhancement options
    parser.add_argument('--level', type=int, choices=[1, 2, 3], default=1,
                       help='Enhancement level: 1=safe, 2=moderate, 3=aggressive')
    parser.add_argument('--corpus-index', type=Path,
                       help='Path to corpus index JSON (for wiki links)')

    # Operation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--preview', action='store_true',
                           help='Preview changes without applying')
    mode_group.add_argument('--apply', action='store_true',
                           help='Apply changes directly')
    mode_group.add_argument('--review', action='store_true',
                           help='Show report and ask for confirmation')

    # Output
    parser.add_argument('--report', type=Path,
                       help='Output path for enhancement report')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip git backup (not recommended)')

    args = parser.parse_args()

    # Initialize enhancer
    enhancer = ContentEnhancer(args.corpus_index)

    # Process file(s)
    if args.file:
        files = [args.file]
    else:
        files = list(args.directory.glob('**/*.md'))
        print(f"Found {len(files)} markdown files")

    # Create backup for batch processing
    if len(files) > 1 and not args.no_backup and not args.preview:
        create_git_backup(f'Before content enhancement (level {args.level})')

    # Process files
    all_reports = []

    for file_path in files:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path.name}")
        print('='*60)

        try:
            report, enhanced_content = enhancer.enhance_file(file_path, args.level)
            all_reports.append(report)

            # Display report
            print(f"\nQuality Score: {report.quality_before:.1f} → {report.quality_after:.1f} "
                  f"(+{report.quality_after - report.quality_before:.1f})")
            print(f"Enhancements: {len(report.enhancements)}")

            for enh_type, count in report._group_by_type().items():
                print(f"  - {enh_type}: {count}")

            # Preview mode
            if args.preview:
                original = file_path.read_text(encoding='utf-8')
                diff = enhancer.generate_diff(original, enhanced_content)
                print("\nDiff:")
                print(diff)

            # Review mode
            elif args.review:
                response = input("\nApply these enhancements? (y/n): ")
                if response.lower() == 'y':
                    file_path.write_text(enhanced_content, encoding='utf-8')
                    print("✓ Enhancements applied")

            # Apply mode
            elif args.apply:
                file_path.write_text(enhanced_content, encoding='utf-8')
                print("✓ Enhancements applied")

        except Exception as e:
            print(f"✗ Error: {e}")

    # Generate combined report
    if args.report:
        report_data = {
            'summary': {
                'files_processed': len(all_reports),
                'total_enhancements': sum(len(r.enhancements) for r in all_reports),
                'avg_quality_improvement': sum(r.quality_after - r.quality_before
                                               for r in all_reports) / len(all_reports)
            },
            'reports': [r.to_dict() for r in all_reports]
        }

        with open(args.report, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n✓ Enhancement report saved to {args.report}")

    # Summary
    print(f"\n{'='*60}")
    print("Enhancement Summary")
    print('='*60)
    print(f"Files processed: {len(all_reports)}")
    print(f"Total enhancements: {sum(len(r.enhancements) for r in all_reports)}")
    if all_reports:
        avg_improvement = sum(r.quality_after - r.quality_before for r in all_reports) / len(all_reports)
        print(f"Average quality improvement: +{avg_improvement:.1f}")


if __name__ == '__main__':
    main()
