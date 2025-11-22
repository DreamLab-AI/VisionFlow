# Content Enhancement Pipeline - Usage Examples

## Common Workflows

### Workflow 1: First Time Setup & Test

```bash
cd /home/user/logseq/scripts/content-standardization

# Step 1: Build corpus index
python3 corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json

# Step 2: Test on single file (preview only)
python3 enhance_content.py \
  --file "../../mainKnowledgeGraph/pages/Bitcoin.md" \
  --level 1 \
  --preview \
  --corpus-index corpus_index.json

# Step 3: Apply to single file
python3 enhance_content.py \
  --file "../../mainKnowledgeGraph/pages/Bitcoin.md" \
  --level 1 \
  --apply \
  --corpus-index corpus_index.json
```

### Workflow 2: Safe Batch Enhancement

```bash
cd /home/user/logseq/scripts/content-standardization

# Option A: Use batch script (recommended)
./batch_enhance.sh

# Option B: Manual command
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --apply \
  --corpus-index corpus_index.json \
  --report reports/enhancement_$(date +%Y%m%d).json
```

### Workflow 3: Selective Enhancement with Review

```bash
# Enhance only AI domain files
python3 enhance_content.py \
  --directory "../../mainKnowledgeGraph/pages" \
  --level 2 \
  --review \
  --corpus-index corpus_index.json

# You'll be prompted for each file:
# Apply these enhancements? (y/n):
```

### Workflow 4: Quality Analysis Only

```bash
# Analyze without making changes
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --preview \
  --report quality_analysis.json

# View summary
cat quality_analysis.json | jq '.summary'
```

### Workflow 5: Progressive Enhancement

```bash
# Level 1: Safe enhancements on all files
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --apply \
  --corpus-index corpus_index.json

# Commit changes
git add -A
git commit -m "Apply Level 1 content enhancements"

# Level 2: Moderate enhancements with review
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 2 \
  --review \
  --corpus-index corpus_index.json

# Commit approved changes
git add -A
git commit -m "Apply Level 2 content enhancements (reviewed)"
```

## Real-World Scenarios

### Scenario A: New Content Added

```bash
# Someone added new pages with US spelling
# Enhance just those files

# Find recently modified files
find ../../mainKnowledgeGraph/pages -name "*.md" -mtime -7 > recent_files.txt

# Enhance each one
while read file; do
  python3 enhance_content.py --file "$file" --level 1 --apply --corpus-index corpus_index.json
done < recent_files.txt
```

### Scenario B: Quality Audit

```bash
# Identify low-quality files
python3 analyze_content_quality.py \
  ../../mainKnowledgeGraph/pages \
  --output quality_audit.csv

# Sort by quality score
sort -t',' -k5 -n quality_audit.csv > sorted_quality.csv

# Enhance lowest quality files first
head -20 sorted_quality.csv | cut -d',' -f1 | while read file; do
  python3 enhance_content.py --file "$file" --level 2 --review --corpus-index corpus_index.json
done
```

### Scenario C: Domain-Specific Enhancement

```bash
# Enhance only Blockchain domain
python3 enhance_content.py \
  --directory "../../mainKnowledgeGraph/pages" \
  --level 2 \
  --apply \
  --corpus-index corpus_index.json \
  --report blockchain_enhancement.json

# Filter by prefix during processing
find ../../mainKnowledgeGraph/pages -name "BC-*.md" | while read file; do
  python3 enhance_content.py --file "$file" --level 2 --apply --corpus-index corpus_index.json
done
```

### Scenario D: Public Facing Content

```bash
# Enhance public content with aggressive settings
PUBLIC_FILES=(
  "Bitcoin.md"
  "Blockchain.md"
  "AI Agent System.md"
  "Money.md"
  "Privacy.md"
)

for file in "${PUBLIC_FILES[@]}"; do
  python3 enhance_content.py \
    --file "../../mainKnowledgeGraph/pages/${file}" \
    --level 3 \
    --review \
    --corpus-index corpus_index.json
done
```

### Scenario E: Rollback After Issues

```bash
# Something went wrong, need to revert

# Option 1: Git reset (if git backup was created)
git reset --hard HEAD~1

# Option 2: Revert specific files
git checkout HEAD~1 -- ../../mainKnowledgeGraph/pages/Bitcoin.md

# Option 3: Restore from specific commit
git log --oneline | grep BACKUP
git reset --hard <commit-hash>
```

## Advanced Techniques

### Custom Enhancement Rules

Edit `enhance_content.py` to add custom rules:

```python
# In ContentEnhancer class
def _custom_uk_enhancement(self, content: str):
    """Add UK-specific terminology."""
    replacements = {
        'cryptocurrency exchange': 'cryptocurrency exchange (known as a crypto trading platform in the UK)',
        'IRS': 'HMRC (UK equivalent of IRS)',
        'Federal Reserve': 'Bank of England',
    }
    # ... implementation
```

### Parallel Processing for Large Batches

```bash
# Split files into chunks and process in parallel
find ../../mainKnowledgeGraph/pages -name "*.md" | split -l 100 - chunk_

# Process chunks in parallel
for chunk in chunk_*; do
  (
    while read file; do
      python3 enhance_content.py --file "$file" --level 1 --apply --corpus-index corpus_index.json
    done < "$chunk"
  ) &
done

wait
```

### Continuous Enhancement

```bash
# Run as cron job for ongoing quality
# Add to crontab: crontab -e

# Run weekly on Sunday at 2 AM
0 2 * * 0 cd /home/user/logseq/scripts/content-standardization && ./batch_enhance.sh

# Or use git hooks
# Create .git/hooks/pre-commit:
#!/bin/bash
cd scripts/content-standardization
git diff --name-only --cached | grep "\.md$" | while read file; do
  python3 enhance_content.py --file "$file" --level 1 --apply --corpus-index corpus_index.json
done
```

## Troubleshooting Examples

### Example 1: Module Import Error

```bash
# Error: ModuleNotFoundError: No module named 'us_to_uk_dict'

# Solution: Run from correct directory
cd /home/user/logseq/scripts/content-standardization
python3 enhance_content.py --help
```

### Example 2: Git Backup Failed

```bash
# Error: Git backup failed - not a git repository

# Solution A: Initialize git
git init
git add -A
git commit -m "Initial commit"

# Solution B: Skip backup
python3 enhance_content.py --file page.md --level 1 --apply --no-backup
```

### Example 3: Low Quality Score After Enhancement

```bash
# Quality score went down after enhancement

# Debug: Check what changed
python3 enhance_content.py --file page.md --level 2 --preview > changes.diff

# Review the diff
less changes.diff

# If too aggressive, use lower level
python3 enhance_content.py --file page.md --level 1 --apply
```

### Example 4: Too Many Wiki Links

```bash
# Level 3 added too many wiki links

# Use more conservative confidence threshold
# Edit enhance_content.py temporarily:
# Change: confidence=0.6 to confidence=0.9

# Or use Level 2 instead
python3 enhance_content.py --file page.md --level 2 --apply
```

## Performance Tips

### Tip 1: Rebuild Index Regularly

```bash
# Rebuild weekly to capture new terms
python3 corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json
```

### Tip 2: Use Preview for Large Batches First

```bash
# Preview before applying
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --preview \
  --report preview_report.json

# Review report
cat preview_report.json | jq '.summary'

# Apply if satisfied
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --apply
```

### Tip 3: Process by Domain

```bash
# Process domains separately for easier tracking
DOMAINS=("AI-" "BC-" "DT-" "RB-")

for domain in "${DOMAINS[@]}"; do
  echo "Processing domain: $domain"
  find ../../mainKnowledgeGraph/pages -name "${domain}*.md" | while read file; do
    python3 enhance_content.py --file "$file" --level 1 --apply --corpus-index corpus_index.json
  done
  git add -A
  git commit -m "Enhanced ${domain} domain content"
done
```

## Monitoring & Reporting

### Generate Regular Reports

```bash
# Weekly quality report
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --preview \
  --report "reports/quality_$(date +%Y%m%d).json"

# Track quality trends
for report in reports/quality_*.json; do
  date=$(basename "$report" .json | sed 's/quality_//')
  avg=$(jq '.summary.avg_quality_improvement' "$report")
  echo "$date: $avg"
done
```

### Compare Before/After

```bash
# Before enhancement
python3 analyze_content_quality.py \
  ../../mainKnowledgeGraph/pages \
  --output quality_before.csv

# Apply enhancements
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --apply \
  --corpus-index corpus_index.json

# After enhancement
python3 analyze_content_quality.py \
  ../../mainKnowledgeGraph/pages \
  --output quality_after.csv

# Compare
join -t',' quality_before.csv quality_after.csv | \
  awk -F',' '{print $1","$5","$10","$10-$5}' > quality_comparison.csv
```

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/content-quality.yml
name: Content Quality Check

on:
  pull_request:
    paths:
      - 'mainKnowledgeGraph/pages/**/*.md'

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Build Corpus Index
        run: |
          cd scripts/content-standardization
          python3 corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json

      - name: Check Content Quality
        run: |
          cd scripts/content-standardization
          python3 enhance_content.py \
            --directory ../../mainKnowledgeGraph/pages \
            --level 1 \
            --preview \
            --report quality_report.json

      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: quality-report
          path: scripts/content-standardization/quality_report.json
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

cd "$(git rev-parse --show-toplevel)/scripts/content-standardization"

# Get staged markdown files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep "\.md$")

if [ -n "$STAGED_FILES" ]; then
    echo "Enhancing staged markdown files..."
    for file in $STAGED_FILES; do
        python3 enhance_content.py \
            --file "../../$file" \
            --level 1 \
            --apply \
            --corpus-index corpus_index.json \
            --no-backup
        git add "../../$file"
    done
fi
```

---

**Version:** 1.0.0
**Last Updated:** 2025-11-21
