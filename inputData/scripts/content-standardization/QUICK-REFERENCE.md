# Content Enhancement Pipeline - Quick Reference

## Quick Start

### 1. Build Corpus Index (First Time Only)
```bash
cd /home/user/logseq/scripts/content-standardization
python3 corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json
```

### 2. Preview Single File
```bash
python3 enhance_content.py \
  --file ../../mainKnowledgeGraph/pages/Bitcoin.md \
  --level 1 \
  --preview
```

### 3. Apply Safe Enhancements to All Files
```bash
python3 enhance_content.py \
  --directory ../../mainKnowledgeGraph/pages \
  --level 1 \
  --apply \
  --corpus-index corpus_index.json \
  --report enhancement_report.json
```

## Enhancement Levels

| Level | Safety | Includes | Use Case |
|-------|--------|----------|----------|
| **1** | 100% Safe | US→UK spelling, formatting fixes | Production batch processing |
| **2** | Review recommended | + wiki links, section restructuring | Curated improvement |
| **3** | Manual review required | + missing sections, aggressive linking | Deep enhancement |

## Common Commands

### Preview Changes
```bash
python3 enhance_content.py --file <FILE> --preview
```

### Apply with Review
```bash
python3 enhance_content.py --directory <DIR> --level 2 --review
```

### Batch Process
```bash
python3 enhance_content.py --directory <DIR> --level 1 --apply --corpus-index corpus_index.json
```

### Generate Report Only
```bash
python3 enhance_content.py --directory <DIR> --level 1 --preview --report report.json
```

## Safety Features

1. **OntologyBlock Protection**: Never modified
2. **Git Backup**: Auto-created before batch processing
3. **Preview Mode**: See changes before applying
4. **Review Mode**: Confirm each file individually
5. **Enhancement Logging**: Detailed reports of all changes

## Revert Changes

```bash
# Revert all changes
git reset --hard HEAD~1

# Revert specific file
git checkout HEAD~1 -- path/to/file.md
```

## Enhancement Types

- **us_to_uk**: US → UK spelling conversion
- **wiki_links_added**: Cross-reference wiki links
- **formatting_fixed**: Bullets, whitespace, headings
- **section_added**: Missing section placeholders
- **sections_identified**: Reports missing sections

## Quality Scoring (0-100)

- **65**: Needs improvement
- **75**: Good quality
- **85**: High quality
- **95**: Excellent

## File Locations

- **Scripts**: `/home/user/logseq/scripts/content-standardization/`
- **Documentation**: `/home/user/logseq/docs/content-standardization/`
- **Knowledge Graph**: `/home/user/logseq/mainKnowledgeGraph/pages/`

## Getting Help

```bash
python3 enhance_content.py --help
python3 corpus_indexer.py --help
```

See full documentation: `/home/user/logseq/docs/content-standardization/CONTENT-ENHANCER-GUIDE.md`
