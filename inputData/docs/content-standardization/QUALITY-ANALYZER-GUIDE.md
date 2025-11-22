# Content Quality Analyzer Guide

## Overview

The Content Quality Analyzer is a comprehensive Python tool designed to assess and score the quality of content in Logseq knowledge base files. It evaluates content across five key dimensions and provides detailed reports with actionable recommendations.

## Quality Scoring System

The analyzer uses a 100-point scale divided into five categories:

### 1. Completeness (30 points)

**Required Sections** (10 points each):
- Technical Overview
- Detailed Explanation

**Optional Sections** (10 points for 3+):
- UK Context
- Historical Background
- Applications and Use Cases
- Technical Details
- Best Practices
- Common Pitfalls
- Related Concepts
- Further Reading
- Examples
- Implementation

**Scoring**:
- Has Technical Overview: +10 points
- Has Detailed Explanation: +10 points
- Has 3+ optional sections: +10 points
- Has 2 optional sections: +7 points
- Has 1 optional section: +4 points

### 2. Depth (25 points)

**Detailed Explanation Word Count** (10 points):
- 500+ words: 10 points
- 350-499 words: 7 points
- 200-349 words: 4 points
- <200 words: 0 points

**Technical Depth** (10 points):
- 2+ code blocks AND 5+ technical terms: 10 points
- 1 code block OR 3+ technical terms: 6 points
- Low technical depth: 0 points

**Examples Provided** (5 points):
- Has Examples section OR code blocks: 5 points
- No examples: 0 points

### 3. Formatting (20 points)

**Proper Logseq Markdown** (5 points):
- No formatting issues: 5 points

**Hyphen-Defined Blocks** (5 points):
- Uses `- ` for bullet points: 5 points

**Proper Heading Hierarchy** (5 points):
- Starts with `##` headings: 5 points
- Has headings but incorrect hierarchy: 3 points

**Code Blocks Formatted Correctly** (5 points):
- All code blocks have language specifiers: 5 points
- Some missing language specifiers: 3 points
- Malformed code blocks: 0 points

### 4. UK English (10 points)

**Scoring**:
- Starts at 10 points
- Deduct 1 point per US spelling found
- Minimum: 0 points

**Common US → UK conversions**:
- analyze → analyse
- organize → organise
- color → colour
- center → centre
- optimize → optimise
- realize → realise
- recognize → recognise
- customize → customise

### 5. Wiki Linking (15 points)

**Scoring based on `[[WikiLink]]` count**:
- 5+ links: 15 points
- 4 links: 12 points
- 3 links: 9 points
- 2 links: 6 points
- 1 link: 3 points
- 0 links: 0 points

## Grading Scale

- **A+**: 90-100 points (Excellent quality)
- **A**: 85-89 points (Very good quality)
- **B+**: 80-84 points (Good quality)
- **B**: 75-79 points (Above average)
- **C+**: 70-74 points (Acceptable)
- **C**: 65-69 points (Needs improvement)
- **D**: 60-64 points (Poor quality)
- **F**: <60 points (Unacceptable quality)

## Installation

### Requirements

```bash
# Core dependencies (built-in Python libraries)
# - re, json, argparse, os, sys, pathlib, typing, dataclasses, collections, csv

# Optional dependencies for visualizations
pip install matplotlib numpy
```

### Setup

```bash
# Make the script executable
chmod +x /home/user/logseq/scripts/content-standardization/analyze_content_quality.py

# Add to PATH (optional)
export PATH=$PATH:/home/user/logseq/scripts/content-standardization
```

## Usage Examples

### Single File Analysis

Analyze a single markdown file and display results in the terminal:

```bash
python analyze_content_quality.py --file mainKnowledgeGraph/pages/Machine_Learning.md
```

Save JSON report:

```bash
python analyze_content_quality.py \
  --file mainKnowledgeGraph/pages/Machine_Learning.md \
  --output reports/ml_quality.json
```

Save markdown report:

```bash
python analyze_content_quality.py \
  --file mainKnowledgeGraph/pages/Machine_Learning.md \
  --markdown reports/ml_quality.md
```

### Directory Analysis

Analyze all markdown files in a directory:

```bash
python analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --output reports/all_pages_quality.json
```

Analyze with CSV export:

```bash
python analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --csv-output reports/quality_report.csv
```

### Domain-Specific Analysis

Analyze files from a specific domain:

```bash
python analyze_content_quality.py \
  --domain ai \
  --output reports/ai_domain_quality.json
```

### Filter by Score

Show only files below a certain score:

```bash
python analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --min-score 70 \
  --csv-output reports/needs_improvement.csv
```

### Generate Visualizations

Create charts and graphs:

```bash
python analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --visualize \
  --viz-output charts/
```

This generates three charts:
- `score_distribution.png`: Histogram of quality scores
- `component_breakdown.png`: Average scores by component
- `issue_frequency.png`: Most common quality issues

### Verbose Mode

Get detailed progress information:

```bash
python analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --verbose
```

## Command-Line Options

```
usage: analyze_content_quality.py [-h] [--file FILE] [--directory DIRECTORY]
                                   [--domain DOMAIN] [--output OUTPUT]
                                   [--markdown MARKDOWN] [--csv-output CSV_OUTPUT]
                                   [--visualize] [--viz-output VIZ_OUTPUT]
                                   [--verbose] [--min-score MIN_SCORE]

Analyze content quality for Logseq knowledge base

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Path to a single file to analyze
  --directory DIRECTORY
                        Path to directory to analyze
  --domain DOMAIN       Analyze files from specific domain
  --output OUTPUT       Output file for JSON report
  --markdown MARKDOWN   Output file for markdown report
  --csv-output CSV_OUTPUT
                        Output CSV file for batch analysis
  --visualize           Generate visualization charts
  --viz-output VIZ_OUTPUT
                        Output directory for visualizations
  --verbose, -v         Verbose output
  --min-score MIN_SCORE
                        Only show files below this score
```

## Output Formats

### JSON Report

```json
{
  "file": "path/to/file.md",
  "overall_score": 75.0,
  "grade": "B",
  "scores": {
    "completeness": "25/30",
    "depth": "20/25",
    "formatting": "18/20",
    "uk_english": "8/10",
    "wiki_linking": "10/15"
  },
  "issues": [
    {
      "type": "us_english",
      "description": "US English spelling detected",
      "line": 45,
      "word": "organize",
      "suggestion": "organise",
      "severity": "low"
    }
  ],
  "recommendations": [
    "Add UK Context section",
    "Convert US English to UK English (3 instances)",
    "Add more wiki links to related concepts"
  ],
  "metadata": {
    "file_size_bytes": 12345,
    "line_count": 234,
    "word_count": 1567,
    "section_count": 8,
    "wiki_link_count": 5,
    "code_block_count": 3,
    "sections": ["Technical Overview", "Detailed Explanation", ...]
  }
}
```

### Markdown Report

```markdown
# Quality Report: Machine_Learning.md

**Overall Score**: 75.0/100 (Grade: B)

## Score Breakdown

- **Completeness**: 25/30
- **Depth**: 20/25
- **Formatting**: 18/20
- **UK English**: 8/10
- **Wiki Linking**: 10/15

## Issues Found

### High Priority

- **missing_section**: Missing required section: UK Context

### Low Priority

- **us_english**: US English spelling detected (line 45) → suggest: 'organise'

## Recommendations

1. Add UK Context section
2. Convert US English to UK English (3 instances)
3. Add more wiki links to related concepts

## Metadata

- **File Size**: 12,345 bytes
- **Lines**: 234
- **Words**: 1,567
- **Sections**: 8
- **Wiki Links**: 5
- **Code Blocks**: 3
```

### CSV Report

For batch analysis, the CSV format provides a spreadsheet-friendly view:

| File | Overall Score | Grade | Completeness | Depth | Formatting | UK English | Wiki Linking | Issue Count | Word Count | Section Count | Wiki Link Count |
|------|---------------|-------|--------------|-------|------------|------------|--------------|-------------|------------|---------------|-----------------|
| file1.md | 85.0 | A | 28.0 | 22.0 | 19.0 | 9.0 | 12.0 | 3 | 1200 | 7 | 6 |
| file2.md | 65.0 | C | 20.0 | 15.0 | 16.0 | 7.0 | 9.0 | 8 | 600 | 4 | 3 |

## Batch Analysis Features

When analyzing multiple files, the tool generates:

### Summary Statistics

- Total files analyzed
- Average, median, min, and max scores
- Grade distribution
- Files needing improvement (score < 70)
- Most common issues across all files

### Example Output

```
============================================================
BATCH ANALYSIS SUMMARY
============================================================
Total files analyzed: 150
Average score: 72.5/100
Median score: 75.0/100
Score range: 45.0 - 95.0

Grade Distribution:
  A+: 12 files
  A: 18 files
  B+: 25 files
  B: 30 files
  C+: 28 files
  C: 20 files
  D: 10 files
  F: 7 files

Files needing improvement (score < 70): 37
  - path/to/file1.md
  - path/to/file2.md
  ...

Most common issues:
  - low_wiki_linking: 89 occurrences
  - us_english: 67 occurrences
  - missing_optional_sections: 45 occurrences
  - shallow_explanation: 34 occurrences
  - missing_examples: 28 occurrences
```

## Integration with Workflow

### Pre-Commit Hook

Create a pre-commit hook to check content quality:

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Get list of modified .md files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.md$')

if [ -z "$FILES" ]; then
    exit 0
fi

# Check quality of modified files
for FILE in $FILES; do
    SCORE=$(python scripts/content-standardization/analyze_content_quality.py \
            --file "$FILE" \
            --output /tmp/quality.json 2>/dev/null | \
            jq '.overall_score')

    if [ $(echo "$SCORE < 60" | bc) -eq 1 ]; then
        echo "ERROR: $FILE has quality score $SCORE (minimum: 60)"
        exit 1
    fi
done

exit 0
```

### CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
name: Content Quality Check

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Analyze content quality
        run: |
          python scripts/content-standardization/analyze_content_quality.py \
            --directory mainKnowledgeGraph/pages \
            --min-score 60 \
            --csv-output quality_report.csv
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: quality-report
          path: quality_report.csv
```

### Periodic Quality Audits

Schedule weekly quality audits:

```bash
#!/bin/bash
# scripts/weekly_quality_audit.sh

DATE=$(date +%Y-%m-%d)
OUTPUT_DIR="reports/quality_audits/$DATE"

mkdir -p "$OUTPUT_DIR"

python scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --output "$OUTPUT_DIR/full_report.json" \
  --csv-output "$OUTPUT_DIR/quality_data.csv" \
  --visualize \
  --viz-output "$OUTPUT_DIR/charts/"

# Send email notification or post to Slack
# ...
```

## Interpreting Results

### High-Quality Content (90-100)

Content with A+ grades demonstrates:
- Comprehensive coverage with all required and 3+ optional sections
- In-depth Detailed Explanation section (500+ words)
- Rich technical content with examples and code
- Proper UK English throughout
- Well-connected to knowledge graph (5+ wiki links)
- Excellent formatting

**Action**: Use as templates for new content

### Good Quality Content (75-89)

Content with A or B+ grades is solid but has room for improvement:
- May be missing optional sections
- Could use more depth or examples
- Minor spelling or formatting issues

**Action**: Minor improvements, not urgent

### Needs Improvement (60-74)

Content with B, C+, or C grades requires attention:
- Missing required content
- Shallow explanations
- Poor formatting or linking

**Action**: Schedule for revision

### Poor Quality (<60)

Content with D or F grades is unacceptable:
- Incomplete or minimal content
- No proper structure
- Multiple critical issues

**Action**: Prioritise for immediate rewrite

## Common Issues and Solutions

### Issue: Low Wiki Linking Score

**Problem**: Content has fewer than 5 wiki links

**Solution**:
1. Identify related concepts mentioned in the text
2. Check if pages exist for these concepts
3. Add `[[WikiLink]]` syntax around relevant terms
4. Focus on: technical terms, related concepts, prerequisite knowledge

### Issue: US English Detected

**Problem**: Content uses American spelling

**Solution**:
1. Review the specific words flagged in the report
2. Replace with UK equivalents (e.g., analyze → analyse)
3. Use editor's find/replace for common patterns
4. Consider using a UK English spell checker

### Issue: Shallow Detailed Explanation

**Problem**: Detailed Explanation section < 500 words

**Solution**:
1. Expand on how the concept works
2. Add more context and background
3. Include technical details and mechanisms
4. Explain why it's important and when to use it
5. Add examples and scenarios

### Issue: Missing Optional Sections

**Problem**: Fewer than 3 optional sections present

**Solution**:
1. Add "UK Context" if relevant (always recommended)
2. Include "Applications and Use Cases" for practical relevance
3. Add "Best Practices" or "Common Pitfalls" for actionable advice
4. Include "Related Concepts" for knowledge graph connectivity

### Issue: Poor Formatting

**Problem**: Incorrect Logseq markdown structure

**Solution**:
1. Use `## Heading` for top-level sections (not `#` or `###`)
2. Use `- ` for bullet points (not `*` or numbered lists)
3. Add language specifiers to code blocks: ` ```python ` not ` ``` `
4. Ensure proper blank lines between sections

## Advanced Usage

### Custom Scoring Thresholds

Modify the scoring system by editing the script:

```python
# Adjust weights
COMPLETENESS_WEIGHT = 30  # Default
DEPTH_WEIGHT = 25         # Default
FORMATTING_WEIGHT = 20    # Default
UK_ENGLISH_WEIGHT = 10    # Default
WIKI_LINKING_WEIGHT = 15  # Default
```

### Add Custom Detection Rules

Extend the US English dictionary:

```python
US_TO_UK_SPELLING.update({
    'program': 'programme',  # Add custom mappings
    'dialog': 'dialogue'
})
```

### Integrate with Other Tools

Pipe output to other analysis tools:

```bash
# Find files with most US English errors
python analyze_content_quality.py \
  --directory pages \
  --output /tmp/quality.json | \
  jq '.reports[] | select(.issues[] | select(.type == "us_english")) | .file'

# Calculate domain-specific statistics
python analyze_content_quality.py \
  --domain ai --output /tmp/ai.json
python analyze_content_quality.py \
  --domain business --output /tmp/business.json
jq -s '.[0].summary.average_score, .[1].summary.average_score' \
  /tmp/ai.json /tmp/business.json
```

## Troubleshooting

### Error: "No files found to analyze"

**Cause**: Directory doesn't contain .md files or path is incorrect

**Solution**: Verify the directory path and ensure it contains markdown files

### Error: "matplotlib not installed"

**Cause**: Visualization dependencies missing

**Solution**:
```bash
pip install matplotlib numpy
```

### Warning: "Malformed code blocks detected"

**Cause**: Code blocks not properly closed with ` ``` `

**Solution**: Ensure all code blocks have matching opening and closing backticks

### Low Scores Despite Good Content

**Cause**: Content doesn't follow expected structure

**Solution**:
- Ensure section headings exactly match expected names
- Use `##` for section headings (not `#` or `###`)
- Check that Detailed Explanation section can be identified

## Best Practices

1. **Run regularly**: Analyze content quality at least weekly
2. **Set standards**: Define minimum acceptable scores for your team
3. **Track progress**: Keep historical reports to monitor improvement
4. **Automate**: Integrate into CI/CD pipeline
5. **Focus on high-impact**: Prioritise improving frequently-accessed pages
6. **Use as teaching tool**: Share high-quality examples with content creators
7. **Balance quantity and quality**: Better to have fewer high-quality pages than many poor ones
8. **Iterate**: Use feedback from the tool to continuously improve content

## Contributing

To extend the analyzer:

1. Add new quality dimensions in `QualityScores` dataclass
2. Implement corresponding check methods
3. Update scoring weights to total 100 points
4. Add tests for new detection logic
5. Update this documentation

## Support

For issues or questions:
- Check this guide first
- Review the script comments
- Test with sample files provided
- Consult Logseq documentation for formatting guidelines

## Version History

- **v1.0** (2025-11-21): Initial release with comprehensive quality scoring system

---

*This tool is part of the content standardisation initiative for the Logseq knowledge base.*
