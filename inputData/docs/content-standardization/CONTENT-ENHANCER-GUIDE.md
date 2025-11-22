# Content Enhancement Pipeline - User Guide

## Overview

The Content Enhancement Pipeline automatically improves content quality through intelligent transformations while preserving ontology integrity.

## Features

### 1. US → UK English Conversion
- **Comprehensive dictionary**: 100+ spelling conversions
- **Context-aware**: Preserves technical terms (e.g., "computer program")
- **Safe replacement**: Uses word boundaries to avoid partial matches
- **Multiple forms**: Handles base words, plurals, and inflected forms

**Examples:**
- `behavior` → `behaviour`
- `optimize` → `optimise`
- `center` → `centre`
- `analyze` → `analyse`

### 2. Wiki Link Enhancement
- **Corpus-based**: Uses knowledge graph index for intelligent suggestions
- **Confidence scoring**: Only links high-quality terms
- **Non-intrusive**: Links only first occurrence per section
- **Context-aware**: Skips already-linked terms

**Examples:**
- `Machine Learning` → `[[Machine Learning]]`
- `Blockchain technology` → `[[Blockchain]] technology`

### 3. Section Restructuring
- **Standard structure**: Organizes content into canonical sections
- **Missing section detection**: Identifies gaps in documentation
- **UK Context addition**: Ensures UK-specific considerations are included

**Standard Sections:**
1. Overview
2. Key Concepts
3. Technical Details
4. UK Context
5. Applications
6. Challenges
7. Future Directions
8. Related Concepts
9. Further Reading

### 4. Formatting Fixes
- **Bullet consistency**: Converts `*` to `-` for Logseq compatibility
- **Whitespace cleanup**: Removes trailing spaces and excessive blank lines
- **Heading hierarchy**: Ensures proper nesting
- **Code block formatting**: Preserves technical content

## Enhancement Levels

### Level 1: Safe (Automatic)
**Recommended for**: Production batch processing

**Includes:**
- US → UK spelling conversion
- Formatting fixes (bullets, whitespace)
- High-confidence wiki links (≥0.8)

**Safety:** 100% safe, no content restructuring

**Example:**
```bash
python enhance_content.py --file page.md --level 1 --apply
```

### Level 2: Moderate (Review Recommended)
**Recommended for**: Curated content improvement

**Includes:**
- All Level 1 enhancements
- Section restructuring
- Medium-confidence wiki links (≥0.8)
- Missing section identification

**Safety:** Safe, but review recommended

**Example:**
```bash
python enhance_content.py --directory pages/ --level 2 --review
```

### Level 3: Aggressive (Manual Review Required)
**Recommended for**: Deep content enhancement

**Includes:**
- All Level 2 enhancements
- Missing section placeholders
- Lower-confidence wiki links (≥0.6)
- Content quality improvements

**Safety:** Requires manual review before applying

**Example:**
```bash
python enhance_content.py --file page.md --level 3 --preview
```

## Safety Features

### 1. OntologyBlock Protection
**The enhancer NEVER modifies OntologyBlock content.**

The OntologyBlock is detected and preserved:
```markdown
- ### OntologyBlock
  id:: ai-agent-system-ontology
  ...
```

All enhancements apply only to body content after the OntologyBlock.

### 2. Git Backup
Before batch processing, automatic git commit is created:

```bash
# Automatic backup
git add -A
git commit -m "[BACKUP] Before content enhancement (level 1)"
```

Disable with `--no-backup` (not recommended):
```bash
python enhance_content.py --directory pages/ --level 2 --no-backup
```

### 3. Preview Mode
See all changes before applying:

```bash
python enhance_content.py --file page.md --preview
```

Shows unified diff of all modifications.

### 4. Review Mode
Review each file individually:

```bash
python enhance_content.py --directory pages/ --level 2 --review
```

Prompts for confirmation before applying each file.

### 5. Enhancement Logging
Detailed logs of all changes:

```json
{
  "file": "path/to/file.md",
  "quality_before": 65.0,
  "quality_after": 82.0,
  "improvement": +17.0,
  "enhancements_by_type": {
    "us_to_uk": 5,
    "wiki_links_added": 8,
    "formatting_fixed": 3
  }
}
```

### 6. Revert Capability
If using git backup, easy revert:

```bash
# Revert to previous state
git reset --hard HEAD~1

# Or cherry-pick specific files
git checkout HEAD~1 -- path/to/file.md
```

## Setup & Installation

### 1. Install Dependencies
```bash
# No external dependencies required!
# Uses only Python standard library
```

### 2. Build Corpus Index
Before using wiki link features, build the index:

```bash
cd /home/user/logseq/scripts/content-standardization
python corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json
```

**Output:**
```
Scanning /home/user/logseq/mainKnowledgeGraph/pages for terms...
Found 1001 markdown files

Index built:
  - 1000 preferred terms
  - 2500 alternative terms

Index saved to corpus_index.json

Index statistics:
  Total terms: 1000
  Total alternatives: 2500
  Linkable terms (quality ≥ 0.8): 850
```

### 3. Make Scripts Executable
```bash
chmod +x enhance_content.py corpus_indexer.py us_to_uk_dict.py
```

## Usage Examples

### Example 1: Single File Preview
```bash
python enhance_content.py \
  --file ../../mainKnowledgeGraph/pages/Bitcoin.md \
  --level 1 \
  --preview
```

**Output:**
```
============================================================
Processing: Bitcoin.md
============================================================

Quality Score: 68.5 → 79.2 (+10.7)
Enhancements: 12
  - us_to_uk: 5
  - wiki_links_added: 4
  - formatting_fixed: 3

Diff:
--- original
+++ enhanced
@@ -45,7 +45,7 @@
-Bitcoin provides a decentralized alternative to traditional financial systems.
+Bitcoin provides a decentralised alternative to traditional financial systems.
...
```

### Example 2: Safe Batch Enhancement
```bash
python enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --apply \
  --corpus-index corpus_index.json \
  --report enhancement_report.json
```

**Output:**
```
Found 1001 markdown files
✓ Git backup created

============================================================
Processing: AI Agent System.md
============================================================
Quality Score: 72.0 → 84.5 (+12.5)
Enhancements: 15
✓ Enhancements applied

[... processing continues ...]

============================================================
Enhancement Summary
============================================================
Files processed: 1001
Total enhancements: 8,523
Average quality improvement: +11.3

✓ Enhancement report saved to enhancement_report.json
```

### Example 3: Selective Enhancement with Review
```bash
python enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 2 \
  --review \
  --corpus-index corpus_index.json
```

**Interactive:**
```
Processing: Privacy.md
Quality Score: 65.0 → 81.0 (+16.0)
Enhancements: 23
  - us_to_uk: 8
  - wiki_links_added: 12
  - formatting_fixed: 3

Apply these enhancements? (y/n): y
✓ Enhancements applied
```

### Example 4: Generate Report Only
```bash
python enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 2 \
  --preview \
  --report analysis_report.json
```

Analyzes all files without modifying, generates comprehensive report.

## Quality Scoring

Files are scored 0-100 based on:

### 1. Length & Completeness (20 points)
- Word count vs. expected content depth
- 500+ words = full points

### 2. Section Structure (20 points)
- Number of well-organized sections
- Presence of standard sections

### 3. Wiki Link Density (20 points)
- Number of cross-references to other concepts
- Indicates integration with knowledge graph

### 4. UK Spelling Usage (20 points)
- Ratio of UK vs. US spellings
- Indicates content standardization

### 5. Formatting Quality (20 points)
- Proper bullet formatting
- No trailing whitespace
- Clean heading hierarchy

**Example Scores:**
- **65**: Adequate content, needs improvement
- **75**: Good quality, minor enhancements needed
- **85**: High quality, well-structured
- **95**: Excellent, publication-ready

## Enhancement Report Structure

```json
{
  "summary": {
    "files_processed": 100,
    "total_enhancements": 1250,
    "avg_quality_improvement": 12.5
  },
  "reports": [
    {
      "file": "path/to/file.md",
      "quality_before": 65.0,
      "quality_after": 82.0,
      "improvement": 17.0,
      "enhancements_by_type": {
        "us_to_uk": 5,
        "wiki_links_added": 8,
        "formatting_fixed": 3,
        "section_added": 1
      },
      "enhancements": [
        {
          "type": "us_to_uk",
          "description": "Converted 'behavior' to 'behaviour'",
          "old_value": "behavior",
          "new_value": "behaviour",
          "line_number": 0,
          "confidence": 1.0
        }
      ],
      "timestamp": "2025-11-21T12:00:00"
    }
  ]
}
```

## Corpus Index Structure

The corpus index enables intelligent wiki linking:

```json
{
  "terms": {
    "AI Agent System": {
      "term_id": "AI-0600",
      "file": "mainKnowledgeGraph/pages/AI Agent System.md",
      "domain": "ai",
      "status": "complete",
      "quality_score": 0.92,
      "definition": "An autonomous software entity...",
      "alt_terms": ["AI Agent", "Intelligent Agent", "Software Agent"]
    }
  },
  "alt_terms": {
    "ai agent": "AI Agent System",
    "intelligent agent": "AI Agent System",
    "software agent": "AI Agent System"
  },
  "stats": {
    "total_terms": 1000,
    "total_alt_terms": 2500
  }
}
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Ensure you're running from the correct directory:
```bash
cd /home/user/logseq/scripts/content-standardization
python enhance_content.py --help
```

### Issue: "No corpus index found"
**Solution:** Build the index first:
```bash
python corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json
```

### Issue: "Git backup failed"
**Solution:** Ensure you're in a git repository, or use `--no-backup`:
```bash
python enhance_content.py --file page.md --level 1 --apply --no-backup
```

### Issue: "Quality score decreased"
**Cause:** Usually due to removing content or aggressive restructuring.

**Solution:** Use `--preview` to review changes, consider lower enhancement level.

## Best Practices

### 1. Start Small
```bash
# Test on 5 files first
python enhance_content.py --file test1.md --level 1 --preview
python enhance_content.py --file test2.md --level 1 --preview
# ... review results ...
python enhance_content.py --file test1.md --level 1 --apply
```

### 2. Use Progressive Enhancement
```bash
# Level 1 on all files
python enhance_content.py --directory pages/ --level 1 --apply

# Then Level 2 on high-value files
python enhance_content.py --directory pages/ --level 2 --review
```

### 3. Regular Index Updates
```bash
# Rebuild index weekly
python corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json
```

### 4. Monitor Quality Trends
```bash
# Generate reports regularly
python enhance_content.py \
  --directory pages/ \
  --level 1 \
  --preview \
  --report quality_report_$(date +%Y%m%d).json
```

### 5. Version Control Integration
```bash
# Always work in a branch
git checkout -b content-enhancement

# Run enhancements
python enhance_content.py --directory pages/ --level 1 --apply

# Review changes
git diff

# Commit if satisfied
git commit -m "Apply content enhancements (level 1)"
```

## Advanced Usage

### Custom Enhancement Levels
Modify `enhance_content.py` to create custom enhancement profiles:

```python
class ContentEnhancer:
    def enhance_file(self, file_path, level=1):
        # Custom level 4: US-only for US audience
        if level == 4:
            body_content, enh = self._add_wiki_links(body_content)
            body_content, enh = self._fix_formatting(body_content)
            # Skip US→UK conversion
```

### Batch Processing with Filtering
```bash
# Only enhance AI domain files
find ../../mainKnowledgeGraph/pages -name "AI-*.md" | while read file; do
  python enhance_content.py --file "$file" --level 1 --apply
done

# Only enhance files below quality threshold
python analyze_quality.py | jq '.[] | select(.quality < 70) | .file' | while read file; do
  python enhance_content.py --file "$file" --level 2 --review
done
```

### Integration with CI/CD
```yaml
# .github/workflows/content-quality.yml
name: Content Quality Check
on: [pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build corpus index
        run: python scripts/content-standardization/corpus_indexer.py mainKnowledgeGraph/pages corpus_index.json
      - name: Analyze content quality
        run: |
          python scripts/content-standardization/enhance_content.py \
            --directory mainKnowledgeGraph/pages \
            --level 1 \
            --preview \
            --report quality_report.json
      - name: Check quality threshold
        run: |
          avg_quality=$(jq '.summary.avg_quality_improvement' quality_report.json)
          if (( $(echo "$avg_quality < 75" | bc -l) )); then
            echo "Quality below threshold: $avg_quality"
            exit 1
          fi
```

## API Reference

### ContentEnhancer Class

```python
class ContentEnhancer:
    def __init__(self, corpus_index_path: Path = None)
    def enhance_file(self, file_path: Path, level: int = 1) -> Tuple[EnhancementReport, str]
    def generate_diff(self, original: str, enhanced: str) -> str
```

### CorpusIndexer Class

```python
class CorpusIndexer:
    def __init__(self, pages_dir: Path)
    def build_index(self) -> Dict
    def find_term(self, text: str) -> Tuple[str, float]
    def get_linkable_terms(self, confidence_threshold: float = 0.8) -> List[str]
    def save_index(self, output_file: Path)
    @staticmethod
    def load_index(index_file: Path) -> 'CorpusIndexer'
```

## Support

For issues or questions:
1. Check troubleshooting section
2. Review example usage
3. Examine enhancement reports
4. Consult source code documentation

## Future Enhancements

Planned features:
- [ ] Machine learning-based quality prediction
- [ ] Automated content generation for missing sections
- [ ] Multi-language support beyond US/UK
- [ ] Integration with external style guides
- [ ] Real-time enhancement suggestions in editor
- [ ] Automated outdated content detection
- [ ] Citation and reference validation

---

**Version:** 1.0.0
**Last Updated:** 2025-11-21
**Maintainer:** Content Standardization Team
