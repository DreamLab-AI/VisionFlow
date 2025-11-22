#!/usr/bin/env python3
"""
Content Quality Analyzer for Logseq Knowledge Base
Analyzes and scores content quality across multiple dimensions.
"""

import re
import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import csv

# UK to US English mapping for detection
US_TO_UK_SPELLING = {
    'analyze': 'analyse',
    'analyzing': 'analysing',
    'analyzed': 'analysed',
    'analyzer': 'analyser',
    'organize': 'organise',
    'organizing': 'organising',
    'organized': 'organised',
    'organization': 'organisation',
    'color': 'colour',
    'favor': 'favour',
    'behavior': 'behaviour',
    'center': 'centre',
    'fiber': 'fibre',
    'theater': 'theatre',
    'meter': 'metre',
    'liter': 'litre',
    'defense': 'defence',
    'offense': 'offence',
    'license': 'licence',
    'practice': 'practise',  # when used as verb
    'optimize': 'optimise',
    'optimizing': 'optimising',
    'optimized': 'optimised',
    'optimization': 'optimisation',
    'realize': 'realise',
    'realized': 'realised',
    'realization': 'realisation',
    'recognize': 'recognise',
    'recognized': 'recognised',
    'recognition': 'recognition',
    'customize': 'customise',
    'customized': 'customised',
    'customization': 'customisation',
    'utilize': 'utilise',
    'utilized': 'utilised',
    'utilization': 'utilisation',
    'visualize': 'visualise',
    'visualization': 'visualisation',
    'categorize': 'categorise',
    'categorized': 'categorised',
    'categorization': 'categorisation',
    'initialize': 'initialise',
    'initialized': 'initialised',
    'initialization': 'initialisation',
    'normalize': 'normalise',
    'normalized': 'normalised',
    'normalization': 'normalisation',
}

# Required and optional sections
REQUIRED_SECTIONS = [
    'Technical Overview',
    'Detailed Explanation'
]

OPTIONAL_SECTIONS = [
    'UK Context',
    'Historical Background',
    'Applications and Use Cases',
    'Technical Details',
    'Best Practices',
    'Common Pitfalls',
    'Related Concepts',
    'Further Reading',
    'Examples',
    'Implementation'
]


@dataclass
class QualityIssue:
    """Represents a quality issue found in content."""
    type: str
    description: str
    line: Optional[int] = None
    word: Optional[str] = None
    suggestion: Optional[str] = None
    severity: str = 'medium'  # low, medium, high


@dataclass
class QualityScores:
    """Breakdown of quality scores."""
    completeness: float  # /30
    depth: float  # /25
    formatting: float  # /20
    uk_english: float  # /10
    wiki_linking: float  # /15

    @property
    def total(self) -> float:
        """Calculate total score."""
        return self.completeness + self.depth + self.formatting + self.uk_english + self.wiki_linking


@dataclass
class QualityReport:
    """Complete quality assessment report."""
    file_path: str
    scores: QualityScores
    issues: List[QualityIssue]
    recommendations: List[str]
    metadata: Dict

    @property
    def overall_score(self) -> float:
        """Get overall score out of 100."""
        return self.scores.total

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'file': self.file_path,
            'overall_score': round(self.overall_score, 1),
            'grade': self.get_grade(),
            'scores': {
                'completeness': f"{self.scores.completeness}/30",
                'depth': f"{self.scores.depth}/25",
                'formatting': f"{self.scores.formatting}/20",
                'uk_english': f"{self.scores.uk_english}/10",
                'wiki_linking': f"{self.scores.wiki_linking}/15"
            },
            'issues': [
                {
                    'type': issue.type,
                    'description': issue.description,
                    'line': issue.line,
                    'word': issue.word,
                    'suggestion': issue.suggestion,
                    'severity': issue.severity
                }
                for issue in self.issues
            ],
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }

    def get_grade(self) -> str:
        """Get letter grade based on score."""
        score = self.overall_score
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'B+'
        elif score >= 75:
            return 'B'
        elif score >= 70:
            return 'C+'
        elif score >= 65:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'


class ContentQualityAnalyzer:
    """Analyzes content quality for Logseq markdown files."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def analyze_file(self, file_path: str) -> QualityReport:
        """Analyze a single file and return quality report."""
        if self.verbose:
            print(f"Analyzing: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Initialize scores and issues
        issues = []
        recommendations = []

        # Run all checks
        completeness_score, completeness_issues = self.check_completeness(content)
        depth_score, depth_issues = self.check_depth(content)
        formatting_score, formatting_issues = self.check_formatting(content)
        uk_english_score, uk_issues = self.check_uk_english(content)
        wiki_linking_score, wiki_issues = self.check_wiki_linking(content)

        # Combine issues
        issues.extend(completeness_issues)
        issues.extend(depth_issues)
        issues.extend(formatting_issues)
        issues.extend(uk_issues)
        issues.extend(wiki_issues)

        # Generate recommendations
        recommendations = self.generate_recommendations(
            completeness_score, depth_score, formatting_score,
            uk_english_score, wiki_linking_score, issues
        )

        # Create scores object
        scores = QualityScores(
            completeness=completeness_score,
            depth=depth_score,
            formatting=formatting_score,
            uk_english=uk_english_score,
            wiki_linking=wiki_linking_score
        )

        # Collect metadata
        metadata = self.collect_metadata(content, file_path)

        return QualityReport(
            file_path=file_path,
            scores=scores,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )

    def detect_sections(self, content: str) -> List[str]:
        """Find all ## Heading sections."""
        sections = []
        for line in content.split('\n'):
            match = re.match(r'^##\s+(.+)$', line.strip())
            if match:
                sections.append(match.group(1).strip())
        return sections

    def count_wiki_links(self, content: str) -> int:
        """Count [[WikiLink]] instances."""
        return len(re.findall(r'\[\[([^\]]+)\]\]', content))

    def detect_us_english(self, content: str) -> List[Tuple[str, int, str]]:
        """Find US spelling patterns and return (word, line_number, suggestion)."""
        issues = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Skip code blocks
            if line.strip().startswith('```'):
                continue

            # Check for US spellings (case-insensitive word boundaries)
            for us_word, uk_word in US_TO_UK_SPELLING.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(us_word) + r'\b'
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append((us_word, line_num, uk_word))

        return issues

    def measure_depth(self, content: str) -> Dict:
        """Measure content depth metrics."""
        # Remove code blocks for accurate word count
        content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)

        words = content_no_code.split()
        word_count = len(words)

        # Count paragraphs (non-empty lines that aren't headings or bullets)
        lines = content.split('\n')
        paragraphs = [
            line for line in lines
            if line.strip() and not line.strip().startswith(('#', '-', '*', '```'))
        ]
        paragraph_count = len(paragraphs)

        # Count code blocks
        code_blocks = len(re.findall(r'```', content)) // 2

        # Detect technical terms (simple heuristic: capitalized words, camelCase, snake_case)
        technical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b\w+_\w+\b', content))

        return {
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'code_blocks': code_blocks,
            'technical_terms': technical_terms
        }

    def check_logseq_formatting(self, content: str) -> List[str]:
        """Check Logseq markdown formatting issues."""
        issues = []
        lines = content.split('\n')

        in_code_block = False
        has_hyphen_bullets = False
        has_proper_headings = False

        for line_num, line in enumerate(lines, 1):
            # Track code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            # Check for hyphen bullets
            if re.match(r'^\s*-\s+', line):
                has_hyphen_bullets = True

            # Check heading hierarchy
            if re.match(r'^#{1,6}\s+', line):
                has_proper_headings = True

        if not has_hyphen_bullets:
            issues.append("Missing hyphen-defined bullet points (Logseq style)")

        if not has_proper_headings:
            issues.append("Missing proper heading hierarchy")

        return issues

    def check_completeness(self, content: str) -> Tuple[float, List[QualityIssue]]:
        """Check completeness (30 points)."""
        score = 0.0
        issues = []

        sections = self.detect_sections(content)

        # Required sections (10 points each)
        for required in REQUIRED_SECTIONS:
            if required in sections:
                score += 10
            else:
                issues.append(QualityIssue(
                    type='missing_section',
                    description=f"Missing required section: {required}",
                    severity='high'
                ))

        # Optional sections (10 points for 3+)
        optional_count = sum(1 for opt in OPTIONAL_SECTIONS if opt in sections)
        if optional_count >= 3:
            score += 10
        elif optional_count >= 2:
            score += 7
        elif optional_count >= 1:
            score += 4
        else:
            issues.append(QualityIssue(
                type='missing_optional_sections',
                description=f"Only {optional_count} optional sections found (recommended: 3+)",
                severity='medium'
            ))

        return score, issues

    def check_depth(self, content: str) -> Tuple[float, List[QualityIssue]]:
        """Check content depth (25 points)."""
        score = 0.0
        issues = []

        metrics = self.measure_depth(content)
        sections = self.detect_sections(content)

        # Extract Detailed Explanation section
        detailed_section_words = 0
        if 'Detailed Explanation' in sections:
            # Find content between ## Detailed Explanation and next ##
            match = re.search(
                r'##\s+Detailed Explanation\s*\n(.*?)(?=\n##|\Z)',
                content,
                re.DOTALL
            )
            if match:
                detailed_content = match.group(1)
                detailed_section_words = len(detailed_content.split())

        # Detailed Explanation > 500 words (10 points)
        if detailed_section_words >= 500:
            score += 10
        elif detailed_section_words >= 350:
            score += 7
        elif detailed_section_words >= 200:
            score += 4
        else:
            issues.append(QualityIssue(
                type='shallow_explanation',
                description=f"Detailed Explanation section has {detailed_section_words} words (recommended: 500+)",
                severity='high'
            ))

        # Technical depth (10 points)
        if metrics['code_blocks'] >= 2 and metrics['technical_terms'] >= 5:
            score += 10
        elif metrics['code_blocks'] >= 1 or metrics['technical_terms'] >= 3:
            score += 6
        else:
            issues.append(QualityIssue(
                type='low_technical_depth',
                description=f"Low technical depth: {metrics['code_blocks']} code blocks, {metrics['technical_terms']} technical terms",
                severity='medium'
            ))

        # Examples provided (5 points)
        has_examples = 'Examples' in sections or 'Example' in sections or metrics['code_blocks'] > 0
        if has_examples:
            score += 5
        else:
            issues.append(QualityIssue(
                type='missing_examples',
                description="No examples or code blocks found",
                severity='medium'
            ))

        return score, issues

    def check_formatting(self, content: str) -> Tuple[float, List[QualityIssue]]:
        """Check formatting quality (20 points)."""
        score = 0.0
        issues = []

        formatting_issues = self.check_logseq_formatting(content)

        # Proper Logseq markdown (5 points)
        if len(formatting_issues) == 0:
            score += 5
        else:
            for issue_desc in formatting_issues:
                issues.append(QualityIssue(
                    type='formatting_issue',
                    description=issue_desc,
                    severity='low'
                ))

        # Hyphen-defined blocks (5 points)
        if not any('hyphen' in issue for issue in formatting_issues):
            score += 5

        # Proper heading hierarchy (5 points)
        headings = re.findall(r'^(#{1,6})\s+', content, re.MULTILINE)
        if headings:
            # Check if headings start with ## and don't skip levels
            heading_levels = [len(h) for h in headings]
            if heading_levels and min(heading_levels) == 2:
                score += 5
            elif heading_levels:
                score += 3

        # Code blocks formatted correctly (5 points)
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', content, re.DOTALL)
        if code_blocks:
            properly_formatted = all(lang.strip() != '' for lang, _ in code_blocks)
            if properly_formatted:
                score += 5
            else:
                score += 3
                issues.append(QualityIssue(
                    type='code_formatting',
                    description="Some code blocks missing language specifiers",
                    severity='low'
                ))
        elif '```' in content:
            issues.append(QualityIssue(
                type='code_formatting',
                description="Malformed code blocks detected",
                severity='medium'
            ))

        return score, issues

    def check_uk_english(self, content: str) -> Tuple[float, List[QualityIssue]]:
        """Check UK English usage (10 points)."""
        score = 10.0
        issues = []

        us_spellings = self.detect_us_english(content)

        # Deduct 1 point per US spelling found (minimum 0)
        score = max(0, 10 - len(us_spellings))

        for word, line_num, suggestion in us_spellings:
            issues.append(QualityIssue(
                type='us_english',
                description=f"US English spelling detected",
                line=line_num,
                word=word,
                suggestion=suggestion,
                severity='low'
            ))

        return score, issues

    def check_wiki_linking(self, content: str) -> Tuple[float, List[QualityIssue]]:
        """Check wiki linking quality (15 points)."""
        score = 0.0
        issues = []

        wiki_link_count = self.count_wiki_links(content)

        # Scale: 5+ links = 15 points
        if wiki_link_count >= 5:
            score = 15
        elif wiki_link_count >= 4:
            score = 12
        elif wiki_link_count >= 3:
            score = 9
        elif wiki_link_count >= 2:
            score = 6
        elif wiki_link_count >= 1:
            score = 3
        else:
            issues.append(QualityIssue(
                type='low_wiki_linking',
                description="No wiki links found (recommended: 5+)",
                severity='medium'
            ))

        if wiki_link_count < 5 and wiki_link_count > 0:
            issues.append(QualityIssue(
                type='low_wiki_linking',
                description=f"Only {wiki_link_count} wiki links found (recommended: 5+)",
                severity='low'
            ))

        return score, issues

    def generate_recommendations(self, completeness: float, depth: float,
                                formatting: float, uk_english: float,
                                wiki_linking: float, issues: List[QualityIssue]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Prioritize by score
        if completeness < 20:
            missing_sections = [i for i in issues if i.type == 'missing_section']
            if missing_sections:
                recommendations.append(
                    f"Add required sections: {', '.join([i.description.split(': ')[1] for i in missing_sections])}"
                )

        if depth < 15:
            recommendations.append(
                "Expand Detailed Explanation section to 500+ words with more technical depth"
            )

        if formatting < 15:
            recommendations.append(
                "Improve formatting: use hyphen bullets, proper headings, and code block language specifiers"
            )

        us_english_issues = [i for i in issues if i.type == 'us_english']
        if us_english_issues:
            count = len(us_english_issues)
            recommendations.append(
                f"Convert US English to UK English ({count} instance{'s' if count != 1 else ''})"
            )

        if wiki_linking < 10:
            recommendations.append(
                "Add more wiki links to related concepts (target: 5+ links)"
            )

        # General recommendations
        if not recommendations:
            recommendations.append("Content quality is excellent! Consider minor refinements from issues list.")

        return recommendations

    def collect_metadata(self, content: str, file_path: str) -> Dict:
        """Collect metadata about the file."""
        metrics = self.measure_depth(content)
        sections = self.detect_sections(content)

        return {
            'file_size_bytes': len(content.encode('utf-8')),
            'line_count': len(content.split('\n')),
            'word_count': metrics['word_count'],
            'section_count': len(sections),
            'wiki_link_count': self.count_wiki_links(content),
            'code_block_count': metrics['code_blocks'],
            'sections': sections
        }

    def generate_markdown_report(self, report: QualityReport) -> str:
        """Generate a markdown formatted report."""
        md = []
        md.append(f"# Quality Report: {Path(report.file_path).name}")
        md.append("")
        md.append(f"**Overall Score**: {report.overall_score:.1f}/100 (Grade: {report.get_grade()})")
        md.append("")

        # Scores breakdown
        md.append("## Score Breakdown")
        md.append("")
        md.append(f"- **Completeness**: {report.scores.completeness}/30")
        md.append(f"- **Depth**: {report.scores.depth}/25")
        md.append(f"- **Formatting**: {report.scores.formatting}/20")
        md.append(f"- **UK English**: {report.scores.uk_english}/10")
        md.append(f"- **Wiki Linking**: {report.scores.wiki_linking}/15")
        md.append("")

        # Issues
        if report.issues:
            md.append("## Issues Found")
            md.append("")

            # Group by severity
            high = [i for i in report.issues if i.severity == 'high']
            medium = [i for i in report.issues if i.severity == 'medium']
            low = [i for i in report.issues if i.severity == 'low']

            for severity, issue_list in [('High', high), ('Medium', medium), ('Low', low)]:
                if issue_list:
                    md.append(f"### {severity} Priority")
                    md.append("")
                    for issue in issue_list:
                        line_info = f" (line {issue.line})" if issue.line else ""
                        suggestion = f" → suggest: '{issue.suggestion}'" if issue.suggestion else ""
                        md.append(f"- **{issue.type}**: {issue.description}{line_info}{suggestion}")
                    md.append("")

        # Recommendations
        md.append("## Recommendations")
        md.append("")
        for i, rec in enumerate(report.recommendations, 1):
            md.append(f"{i}. {rec}")
        md.append("")

        # Metadata
        md.append("## Metadata")
        md.append("")
        md.append(f"- **File Size**: {report.metadata['file_size_bytes']:,} bytes")
        md.append(f"- **Lines**: {report.metadata['line_count']:,}")
        md.append(f"- **Words**: {report.metadata['word_count']:,}")
        md.append(f"- **Sections**: {report.metadata['section_count']}")
        md.append(f"- **Wiki Links**: {report.metadata['wiki_link_count']}")
        md.append(f"- **Code Blocks**: {report.metadata['code_block_count']}")

        return '\n'.join(md)


class BatchProcessor:
    """Process multiple files and generate statistics."""

    def __init__(self, analyzer: ContentQualityAnalyzer):
        self.analyzer = analyzer

    def process_directory(self, directory: str, pattern: str = "*.md") -> List[QualityReport]:
        """Process all markdown files in a directory."""
        reports = []
        path = Path(directory)

        for file_path in path.rglob(pattern):
            if file_path.is_file():
                try:
                    report = self.analyzer.analyze_file(str(file_path))
                    reports.append(report)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)

        return reports

    def generate_summary(self, reports: List[QualityReport]) -> Dict:
        """Generate summary statistics."""
        if not reports:
            return {}

        scores = [r.overall_score for r in reports]

        # Grade distribution
        grades = defaultdict(int)
        for report in reports:
            grades[report.get_grade()] += 1

        # Files needing improvement
        needs_improvement = [
            r.file_path for r in reports
            if r.overall_score < 70
        ]

        # Most common issues
        issue_types = defaultdict(int)
        for report in reports:
            for issue in report.issues:
                issue_types[issue.type] += 1

        return {
            'total_files': len(reports),
            'average_score': sum(scores) / len(scores),
            'median_score': sorted(scores)[len(scores) // 2],
            'min_score': min(scores),
            'max_score': max(scores),
            'grade_distribution': dict(grades),
            'files_needing_improvement': needs_improvement,
            'most_common_issues': dict(sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:10])
        }

    def export_csv(self, reports: List[QualityReport], output_file: str):
        """Export reports to CSV for analysis."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'File', 'Overall Score', 'Grade', 'Completeness', 'Depth',
                'Formatting', 'UK English', 'Wiki Linking', 'Issue Count',
                'Word Count', 'Section Count', 'Wiki Link Count'
            ])

            for report in reports:
                writer.writerow([
                    report.file_path,
                    f"{report.overall_score:.1f}",
                    report.get_grade(),
                    f"{report.scores.completeness:.1f}",
                    f"{report.scores.depth:.1f}",
                    f"{report.scores.formatting:.1f}",
                    f"{report.scores.uk_english:.1f}",
                    f"{report.scores.wiki_linking:.1f}",
                    len(report.issues),
                    report.metadata['word_count'],
                    report.metadata['section_count'],
                    report.metadata['wiki_link_count']
                ])


def create_visualizations(reports: List[QualityReport], output_dir: str):
    """Create visualization charts (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualizations.")
        print("Install with: pip install matplotlib")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Score distribution histogram
    scores = [r.overall_score for r in reports]
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.title('Content Quality Score Distribution')
    plt.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Score component breakdown
    components = ['Completeness', 'Depth', 'Formatting', 'UK English', 'Wiki Linking']
    max_scores = [30, 25, 20, 10, 15]
    avg_scores = [
        np.mean([r.scores.completeness for r in reports]),
        np.mean([r.scores.depth for r in reports]),
        np.mean([r.scores.formatting for r in reports]),
        np.mean([r.scores.uk_english for r in reports]),
        np.mean([r.scores.wiki_linking for r in reports])
    ]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(components))
    width = 0.35
    plt.bar(x - width/2, avg_scores, width, label='Average', alpha=0.8)
    plt.bar(x + width/2, max_scores, width, label='Maximum', alpha=0.8)
    plt.xlabel('Quality Component')
    plt.ylabel('Score')
    plt.title('Average Scores by Component')
    plt.xticks(x, components, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / 'component_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Issue frequency
    issue_types = defaultdict(int)
    for report in reports:
        for issue in report.issues:
            issue_types[issue.type] += 1

    if issue_types:
        sorted_issues = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:10]
        issue_names, issue_counts = zip(*sorted_issues)

        plt.figure(figsize=(12, 6))
        plt.barh(range(len(issue_names)), issue_counts, alpha=0.8)
        plt.yticks(range(len(issue_names)), issue_names)
        plt.xlabel('Frequency')
        plt.title('Most Common Quality Issues')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_path / 'issue_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze content quality for Logseq knowledge base',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  %(prog)s --file path/to/file.md

  # Analyze all files in a directory
  %(prog)s --directory mainKnowledgeGraph/pages

  # Analyze domain-specific files
  %(prog)s --domain ai --output reports/ai_quality.json

  # Generate CSV export
  %(prog)s --directory pages --csv-output quality_report.csv

  # Generate visualizations
  %(prog)s --directory pages --visualize --viz-output charts/
        """
    )

    parser.add_argument('--file', help='Path to a single file to analyze')
    parser.add_argument('--directory', help='Path to directory to analyze')
    parser.add_argument('--domain', help='Analyze files from specific domain')
    parser.add_argument('--output', help='Output file for JSON report')
    parser.add_argument('--markdown', help='Output file for markdown report')
    parser.add_argument('--csv-output', help='Output CSV file for batch analysis')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization charts')
    parser.add_argument('--viz-output', default='visualizations/', help='Output directory for visualizations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--min-score', type=float, help='Only show files below this score')

    args = parser.parse_args()

    # Validate arguments
    if not (args.file or args.directory or args.domain):
        parser.error("Must specify --file, --directory, or --domain")

    analyzer = ContentQualityAnalyzer(verbose=args.verbose)

    # Single file analysis
    if args.file:
        report = analyzer.analyze_file(args.file)

        # Output
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"JSON report saved to {args.output}")

        if args.markdown:
            md_report = analyzer.generate_markdown_report(report)
            with open(args.markdown, 'w') as f:
                f.write(md_report)
            print(f"Markdown report saved to {args.markdown}")

        # Print to console
        print("\n" + analyzer.generate_markdown_report(report))

    # Directory or domain analysis
    elif args.directory or args.domain:
        processor = BatchProcessor(analyzer)

        # Determine directory
        if args.domain:
            # Assume domain files are in domains/{domain}/
            directory = f"domains/{args.domain}"
            if not os.path.exists(directory):
                # Try mainKnowledgeGraph/pages/domains/{domain}
                directory = f"mainKnowledgeGraph/pages/domains/{args.domain}"
        else:
            directory = args.directory

        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}", file=sys.stderr)
            sys.exit(1)

        # Process all files
        print(f"Processing files in {directory}...")
        reports = processor.process_directory(directory)

        if not reports:
            print("No files found to analyze.", file=sys.stderr)
            sys.exit(1)

        # Filter by min score if specified
        if args.min_score is not None:
            reports = [r for r in reports if r.overall_score < args.min_score]
            print(f"Filtered to {len(reports)} files with score < {args.min_score}")

        # Generate summary
        summary = processor.generate_summary(reports)

        # Output
        if args.output:
            output_data = {
                'summary': summary,
                'reports': [r.to_dict() for r in reports]
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"JSON report saved to {args.output}")

        if args.csv_output:
            processor.export_csv(reports, args.csv_output)
            print(f"CSV export saved to {args.csv_output}")

        if args.visualize:
            create_visualizations(reports, args.viz_output)

        # Print summary
        print("\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total files analyzed: {summary['total_files']}")
        print(f"Average score: {summary['average_score']:.1f}/100")
        print(f"Median score: {summary['median_score']:.1f}/100")
        print(f"Score range: {summary['min_score']:.1f} - {summary['max_score']:.1f}")
        print(f"\nGrade Distribution:")
        for grade, count in sorted(summary['grade_distribution'].items()):
            print(f"  {grade}: {count} files")

        if summary['files_needing_improvement']:
            print(f"\nFiles needing improvement (score < 70): {len(summary['files_needing_improvement'])}")
            for file in summary['files_needing_improvement'][:5]:
                print(f"  - {file}")
            if len(summary['files_needing_improvement']) > 5:
                print(f"  ... and {len(summary['files_needing_improvement']) - 5} more")

        print(f"\nMost common issues:")
        for issue_type, count in list(summary['most_common_issues'].items())[:5]:
            print(f"  - {issue_type}: {count} occurrences")


if __name__ == '__main__':
    main()
