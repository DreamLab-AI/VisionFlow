# Content Quality Analyzer - Usage Examples

## Quick Start

### Basic Single File Analysis

```bash
# Analyze a file and display results in terminal
python3 scripts/content-standardization/analyze_content_quality.py \
  --file tests/content-standardization/sample_high_quality.md
```

### Output Formats

```bash
# Save as JSON
python3 scripts/content-standardization/analyze_content_quality.py \
  --file mainKnowledgeGraph/pages/Machine_Learning.md \
  --output reports/ml_quality.json

# Save as Markdown
python3 scripts/content-standardization/analyze_content_quality.py \
  --file mainKnowledgeGraph/pages/Machine_Learning.md \
  --markdown reports/ml_quality.md

# Save both formats
python3 scripts/content-standardization/analyze_content_quality.py \
  --file mainKnowledgeGraph/pages/Machine_Learning.md \
  --output reports/ml_quality.json \
  --markdown reports/ml_quality.md
```

## Batch Analysis

### Analyze Entire Directory

```bash
# Analyze all markdown files in a directory
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --output reports/all_pages.json \
  --csv-output reports/all_pages.csv
```

### Filter Low-Quality Files

```bash
# Find files scoring below 70
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --min-score 70 \
  --csv-output reports/needs_improvement.csv

# Find files scoring below 60 (failing grade)
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --min-score 60 \
  --output reports/failing_files.json
```

### Domain-Specific Analysis

```bash
# Analyze AI domain files
python3 scripts/content-standardization/analyze_content_quality.py \
  --domain ai \
  --output reports/ai_quality.json

# Analyze Business domain files
python3 scripts/content-standardization/analyze_content_quality.py \
  --domain business \
  --csv-output reports/business_quality.csv
```

## Visualization

```bash
# Generate charts for quality analysis
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --visualize \
  --viz-output charts/quality_analysis/

# This creates:
# - charts/quality_analysis/score_distribution.png
# - charts/quality_analysis/component_breakdown.png
# - charts/quality_analysis/issue_frequency.png
```

## Verbose Mode

```bash
# See detailed progress information
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --verbose \
  --output reports/full_analysis.json
```

## Real-World Workflows

### Weekly Quality Audit

```bash
#!/bin/bash
# weekly_audit.sh

DATE=$(date +%Y-%m-%d)
REPORT_DIR="reports/weekly/$DATE"

mkdir -p "$REPORT_DIR"

# Run full analysis
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --output "$REPORT_DIR/full_report.json" \
  --csv-output "$REPORT_DIR/quality_data.csv" \
  --visualize \
  --viz-output "$REPORT_DIR/charts/"

# Identify files needing urgent attention
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --min-score 60 \
  --output "$REPORT_DIR/urgent_fixes.json"

echo "Quality audit complete: $REPORT_DIR"
```

### Pre-Commit Quality Check

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check only staged markdown files
for FILE in $(git diff --cached --name-only --diff-filter=ACM | grep '\.md$'); do
  RESULT=$(python3 scripts/content-standardization/analyze_content_quality.py \
    --file "$FILE" \
    --output /tmp/quality_check.json 2>&1)

  SCORE=$(jq '.overall_score' /tmp/quality_check.json 2>/dev/null)

  if [ ! -z "$SCORE" ] && [ $(echo "$SCORE < 60" | bc) -eq 1 ]; then
    echo "❌ QUALITY CHECK FAILED: $FILE (score: $SCORE/100)"
    echo "Run: python3 scripts/content-standardization/analyze_content_quality.py --file $FILE"
    exit 1
  fi
done

echo "✅ Content quality checks passed"
```

### Monthly Trend Analysis

```bash
#!/bin/bash
# monthly_trends.sh

# Analyze quality trends over time
for MONTH in 01 02 03 04 05 06; do
  BACKUP_DIR="backups/2025-$MONTH-01"

  if [ -d "$BACKUP_DIR/mainKnowledgeGraph/pages" ]; then
    python3 scripts/content-standardization/analyze_content_quality.py \
      --directory "$BACKUP_DIR/mainKnowledgeGraph/pages" \
      --output "reports/trends/2025-$MONTH.json"
  fi
done

# Compare results
jq '.summary.average_score' reports/trends/*.json
```

### Domain Comparison

```bash
#!/bin/bash
# compare_domains.sh

DOMAINS=("ai" "business" "technology" "science" "health" "engineering")

for DOMAIN in "${DOMAINS[@]}"; do
  python3 scripts/content-standardization/analyze_content_quality.py \
    --domain "$DOMAIN" \
    --output "reports/domains/${DOMAIN}_quality.json"
done

# Generate comparison report
echo "Domain Quality Comparison"
echo "========================"
for DOMAIN in "${DOMAINS[@]}"; do
  SCORE=$(jq '.summary.average_score' "reports/domains/${DOMAIN}_quality.json" 2>/dev/null)
  COUNT=$(jq '.summary.total_files' "reports/domains/${DOMAIN}_quality.json" 2>/dev/null)
  printf "%-15s: %.1f/100 (%s files)\n" "$DOMAIN" "$SCORE" "$COUNT"
done
```

### Identify Top and Bottom Performers

```bash
#!/bin/bash
# top_bottom.sh

# Run analysis
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --output /tmp/all_quality.json

# Top 10 highest quality files
echo "🏆 Top 10 Highest Quality Files"
echo "================================"
jq -r '.reports | sort_by(-.overall_score) | .[0:10] | .[] | "\(.overall_score) - \(.file)"' /tmp/all_quality.json

echo ""

# Bottom 10 lowest quality files
echo "⚠️  Top 10 Files Needing Improvement"
echo "===================================="
jq -r '.reports | sort_by(.overall_score) | .[0:10] | .[] | "\(.overall_score) - \(.file)"' /tmp/all_quality.json
```

## Integration Examples

### GitHub Actions Workflow

```yaml
# .github/workflows/content-quality.yml
name: Content Quality Check

on:
  pull_request:
    paths:
      - '**.md'

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install matplotlib numpy

      - name: Analyze changed files
        run: |
          # Get changed markdown files
          CHANGED_FILES=$(git diff --name-only origin/main HEAD | grep '\.md$' || true)

          if [ -z "$CHANGED_FILES" ]; then
            echo "No markdown files changed"
            exit 0
          fi

          # Analyze each changed file
          for FILE in $CHANGED_FILES; do
            python3 scripts/content-standardization/analyze_content_quality.py \
              --file "$FILE" \
              --output "reports/${FILE##*/}.json"

            SCORE=$(jq '.overall_score' "reports/${FILE##*/}.json")
            echo "::notice file=$FILE::Quality score: $SCORE/100"

            if [ $(echo "$SCORE < 60" | bc) -eq 1 ]; then
              echo "::error file=$FILE::Quality score too low: $SCORE/100 (minimum: 60)"
              exit 1
            fi
          done

      - name: Upload reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: quality-reports
          path: reports/*.json
```

### Makefile Integration

```makefile
# Makefile

.PHONY: quality-check quality-report quality-fix

quality-check:
	@python3 scripts/content-standardization/analyze_content_quality.py \
		--directory mainKnowledgeGraph/pages \
		--min-score 60 \
		--csv-output /tmp/quality_check.csv
	@echo "Quality check complete"

quality-report:
	@mkdir -p reports/quality
	@python3 scripts/content-standardization/analyze_content_quality.py \
		--directory mainKnowledgeGraph/pages \
		--output reports/quality/full_report.json \
		--csv-output reports/quality/quality_data.csv \
		--visualize \
		--viz-output reports/quality/charts/
	@echo "Quality report generated: reports/quality/"

quality-fix:
	@python3 scripts/content-standardization/analyze_content_quality.py \
		--directory mainKnowledgeGraph/pages \
		--min-score 70 \
		--output reports/needs_improvement.json
	@jq -r '.reports[] | .file' reports/needs_improvement.json
```

### Python Script Integration

```python
#!/usr/bin/env python3
# quality_dashboard.py

import json
import subprocess
from pathlib import Path

def analyze_directory(directory):
    """Run quality analysis on a directory."""
    result = subprocess.run([
        'python3',
        'scripts/content-standardization/analyze_content_quality.py',
        '--directory', directory,
        '--output', '/tmp/quality_analysis.json'
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    with open('/tmp/quality_analysis.json') as f:
        return json.load(f)

def generate_dashboard(data):
    """Generate HTML dashboard from quality data."""
    html = f"""
    <html>
    <head><title>Content Quality Dashboard</title></head>
    <body>
        <h1>Content Quality Dashboard</h1>
        <h2>Summary</h2>
        <ul>
            <li>Total Files: {data['summary']['total_files']}</li>
            <li>Average Score: {data['summary']['average_score']:.1f}/100</li>
            <li>Files Needing Improvement: {len(data['summary']['files_needing_improvement'])}</li>
        </ul>
        <h2>Grade Distribution</h2>
        <ul>
    """

    for grade, count in sorted(data['summary']['grade_distribution'].items()):
        html += f"<li>{grade}: {count} files</li>\n"

    html += """
        </ul>
    </body>
    </html>
    """

    return html

if __name__ == '__main__':
    data = analyze_directory('mainKnowledgeGraph/pages')
    if data:
        html = generate_dashboard(data)
        Path('reports/dashboard.html').write_text(html)
        print("Dashboard generated: reports/dashboard.html")
```

## Advanced Queries

### Using jq for Data Analysis

```bash
# Get average score by domain
for DOMAIN in ai business technology; do
  SCORE=$(python3 scripts/content-standardization/analyze_content_quality.py \
    --domain "$DOMAIN" --output /tmp/${DOMAIN}.json 2>/dev/null | \
    jq '.summary.average_score')
  echo "$DOMAIN: $SCORE"
done

# Find all files with US English issues
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --output /tmp/all.json

jq -r '.reports[] | select(.issues[] | .type == "us_english") | .file' /tmp/all.json

# Count issues by type
jq '.summary.most_common_issues | to_entries | .[] | "\(.key): \(.value)"' /tmp/all.json

# Find files with low wiki linking
jq -r '.reports[] | select(.scores.wiki_linking | tonumber < 10) | .file' /tmp/all.json
```

### Export to Excel-Compatible CSV

```bash
# Create detailed CSV for Excel analysis
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory mainKnowledgeGraph/pages \
  --csv-output reports/detailed_quality.csv

# Open in Excel or LibreOffice Calc
libreoffice --calc reports/detailed_quality.csv
```

## Testing Examples

### Test on Sample Files

```bash
# High quality (should score 90+)
python3 scripts/content-standardization/analyze_content_quality.py \
  --file tests/content-standardization/sample_high_quality.md

# Medium quality (should score 60-75)
python3 scripts/content-standardization/analyze_content_quality.py \
  --file tests/content-standardization/sample_medium_quality.md

# Low quality (should score <40)
python3 scripts/content-standardization/analyze_content_quality.py \
  --file tests/content-standardization/sample_low_quality.md
```

### Batch Test

```bash
# Test all samples at once
python3 scripts/content-standardization/analyze_content_quality.py \
  --directory tests/content-standardization \
  --csv-output /tmp/test_results.csv

# View results
cat /tmp/test_results.csv
```

## Troubleshooting

### Debug Mode

```bash
# Run with verbose output for debugging
python3 -v scripts/content-standardization/analyze_content_quality.py \
  --file problematic_file.md \
  --verbose

# Check Python version
python3 --version  # Should be 3.7+

# Test import
python3 -c "import re, json, argparse; print('OK')"
```

### Common Issues

```bash
# Issue: No output
# Solution: Check file path is correct
ls -la mainKnowledgeGraph/pages/YourFile.md

# Issue: JSON parse error
# Solution: Validate JSON output
jq '.' reports/output.json

# Issue: Permission denied
# Solution: Make script executable
chmod +x scripts/content-standardization/analyze_content_quality.py
```

## Performance Tips

```bash
# For large directories, use parallel processing
find mainKnowledgeGraph/pages -name "*.md" | \
  parallel -j 4 python3 scripts/content-standardization/analyze_content_quality.py \
    --file {} --output reports/{/.}.json

# Process only recently modified files
find mainKnowledgeGraph/pages -name "*.md" -mtime -7 | while read FILE; do
  python3 scripts/content-standardization/analyze_content_quality.py \
    --file "$FILE" --output "reports/weekly/$(basename $FILE).json"
done
```

## Best Practices

1. **Run regularly**: Weekly or after significant content changes
2. **Set thresholds**: Define minimum acceptable scores for your team
3. **Focus on trends**: Track quality over time, not just absolute scores
4. **Prioritise**: Fix failing files (F grade) before improving passing ones
5. **Automate**: Use pre-commit hooks and CI/CD integration
6. **Document**: Keep quality reports for historical reference

---

*For more information, see the [Quality Analyzer Guide](../../docs/content-standardization/QUALITY-ANALYZER-GUIDE.md)*
