# Content Enhancement Pipeline

Automated tools to improve content quality while preserving ontology integrity.

## Features

- **US → UK English Conversion**: 500+ automatic spelling corrections
- **Wiki Link Enhancement**: Intelligent cross-referencing using corpus index
- **Section Restructuring**: Standardized documentation structure
- **Formatting Fixes**: Bullets, whitespace, heading hierarchy
- **Quality Scoring**: Automated content quality assessment (0-100)
- **Safety Features**: OntologyBlock protection, git backup, preview mode

## Quick Start

```bash
# 1. Build corpus index
python3 corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json

# 2. Preview enhancements
python3 enhance_content.py --file <FILE> --level 1 --preview

# 3. Apply enhancements
python3 enhance_content.py --file <FILE> --level 1 --apply --corpus-index corpus_index.json
```

## Enhancement Levels

1. **Level 1** (Safe): US→UK spelling, formatting fixes
2. **Level 2** (Moderate): + wiki links, section restructuring
3. **Level 3** (Aggressive): + missing sections, aggressive linking

## Files

- `enhance_content.py` - Main enhancement engine
- `corpus_indexer.py` - Knowledge graph indexer
- `us_to_uk_dict.py` - US→UK spelling dictionary
- `QUICK-REFERENCE.md` - Command reference
- `corpus_index.json` - Generated index (run indexer first)

## Documentation

Full documentation: `/home/user/logseq/docs/content-standardization/CONTENT-ENHANCER-GUIDE.md`

## Safety

- **Never modifies OntologyBlock**
- Git backup before batch processing
- Preview/review modes available
- Revert capability via git

## Examples

```bash
# Preview single file
python3 enhance_content.py --file page.md --preview

# Batch process with review
python3 enhance_content.py --directory pages/ --level 2 --review

# Safe batch enhancement
python3 enhance_content.py \
  --directory pages/ \
  --level 1 \
  --apply \
  --corpus-index corpus_index.json \
  --report report.json
```

## Requirements

- Python 3.7+
- No external dependencies (uses standard library only)

## Support

For detailed usage, see:
- `QUICK-REFERENCE.md` - Quick command reference
- `CONTENT-ENHANCER-GUIDE.md` - Comprehensive guide (in docs/)
- `--help` flag on any script
