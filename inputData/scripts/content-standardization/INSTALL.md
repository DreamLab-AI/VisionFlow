# Installation & Setup Guide

## Quick Install (30 seconds)

```bash
# Navigate to scripts directory
cd /home/user/logseq/scripts/content-standardization

# Make scripts executable (already done)
chmod +x *.py *.sh

# Build corpus index (required for wiki links)
python3 corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json
```

**Done!** You can now use the enhancement tools.

## Verify Installation

```bash
# Test help output
python3 enhance_content.py --help

# Test on single file (preview only)
python3 enhance_content.py \
  --file "../../mainKnowledgeGraph/pages/Bitcoin.md" \
  --level 1 \
  --preview
```

## What Gets Installed

### Core Tools (4 Python scripts)
- `enhance_content.py` - Main enhancement engine
- `corpus_indexer.py` - Knowledge graph indexer
- `us_to_uk_dict.py` - US→UK spelling dictionary
- `analyze_content_quality.py` - Quality analyzer

### Support Tools
- `batch_enhance.sh` - Batch processing script
- `__init__.py` - Python package initialization

### Documentation (5 guides)
- `README.md` - Overview
- `QUICK-REFERENCE.md` - Command reference
- `USAGE-EXAMPLES.md` - Real-world examples
- `IMPLEMENTATION-SUMMARY.md` - Technical details
- `INSTALL.md` - This file

### Generated Data
- `corpus_index.json` - Term index (356KB)
- `reports/` - Enhancement reports directory

## System Requirements

### Required
- **Python**: 3.7 or higher
- **OS**: Linux, macOS, or Windows
- **Disk Space**: ~1 MB (plus corpus index)

### Optional
- **Git**: For backup/revert features
- **jq**: For JSON report parsing in shell scripts

### No External Python Packages Required
Uses only Python standard library:
- re, sys, json, argparse, subprocess
- pathlib, typing, datetime, dataclasses, difflib

## First Run Checklist

- [ ] Navigate to scripts directory
- [ ] Verify Python 3.7+ installed: `python3 --version`
- [ ] Make scripts executable: `chmod +x *.py *.sh`
- [ ] Build corpus index: `python3 corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json`
- [ ] Test on single file with preview mode
- [ ] Read documentation: `README.md`, `QUICK-REFERENCE.md`

## Next Steps

1. **Read Quick Reference**: `QUICK-REFERENCE.md`
2. **Try Examples**: `USAGE-EXAMPLES.md`
3. **Test on Sample Files**: Use `--preview` mode
4. **Apply Safe Enhancements**: Use `--level 1`
5. **Read Full Guide**: `/home/user/logseq/docs/content-standardization/CONTENT-ENHANCER-GUIDE.md`

## Troubleshooting

### "Command not found"
Make scripts executable: `chmod +x *.py *.sh`

### "ModuleNotFoundError"
Run from correct directory: `cd /home/user/logseq/scripts/content-standardization`

### "No corpus index found"
Build index first: `python3 corpus_indexer.py ../../mainKnowledgeGraph/pages corpus_index.json`

## Support

For detailed usage, see:
- `QUICK-REFERENCE.md` - Quick command reference
- `USAGE-EXAMPLES.md` - Real-world workflows
- `CONTENT-ENHANCER-GUIDE.md` - Comprehensive guide (in docs/)

---

**Installation Time**: ~30 seconds
**Version**: 1.0.0
**Date**: 2025-11-21
