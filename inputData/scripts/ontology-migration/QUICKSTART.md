# Quick Start Guide - Ontology Migration Pipeline

## 🚀 Ready-to-Execute Commands

### Step 1: Scan Files (Required First Step)
```bash
cd /home/user/logseq/scripts/ontology-migration
node cli.js scan
```

**What it does**: Scans 1,709 markdown files, identifies ontology blocks, classifies patterns, detects issues.

**Output**: `/home/user/logseq/docs/ontology-migration/reports/file-inventory.json`

---

### Step 2: Preview Transformations
```bash
# Preview first 10 files
node cli.js preview 10

# Or preview more
node cli.js preview 50
```

**What it does**: Shows you what the transformations will look like WITHOUT making any changes.

---

### Step 3: Test Single File
```bash
# Test robotics file (has namespace issue mv: -> rb:)
node cli.js test /home/user/logseq/mainKnowledgeGraph/pages/rb-0010-aerial-robot.md

# Test AI file
node cli.js test /home/user/logseq/mainKnowledgeGraph/pages/AI-0207-encoder-decoder-architecture.md

# Test blockchain file
node cli.js test /home/user/logseq/mainKnowledgeGraph/pages/BC-0051-consensus-mechanism.md
```

**What it does**:
- Parses the file
- Generates new canonical block
- Shows preview
- Validates and shows score

---

### Step 4: Dry-Run (SAFE - No Changes)
```bash
# Dry-run ALL files
node cli.js process

# Dry-run with validation
node cli.js process --validate

# Dry-run specific domain only
node cli.js domain robotics
```

**What it does**: Simulates the entire migration WITHOUT making any changes. Safe to run!

---

### Step 5: LIVE Update (When Ready)

**⚠️  WARNING: This WILL modify files! Backups are created automatically.**

```bash
# Full migration with backups and validation
node cli.js process --live --validate

# Or process by domain (recommended for first run)
node cli.js domain robotics --live

# Or process specific batch size
node cli.js process --live --batch=50
```

**What it does**:
- Creates timestamped backups
- Updates ontology blocks to canonical format
- Validates results
- Generates comprehensive reports

---

### Step 6: Validate Results
```bash
node cli.js validate
```

**What it does**: Validates all ontology blocks and generates validation report.

---

### Step 7: Check Statistics
```bash
node cli.js stats
```

**What it does**: Shows comprehensive statistics from all reports.

---

## 🛡️ Rollback (If Needed)

```bash
node cli.js rollback
```

**What it does**: Restores all files from backups. Use if something goes wrong.

---

## 📊 Quick Status Check

```bash
# Check what files exist
ls -l /home/user/logseq/docs/ontology-migration/reports/

# View inventory
cat /home/user/logseq/docs/ontology-migration/reports/file-inventory.json | head -50

# Check backups
ls /home/user/logseq/docs/ontology-migration/backups/ | wc -l
```

---

## 🎯 Recommended First-Time Workflow

```bash
# 1. Initial scan
node cli.js scan

# 2. Preview sample
node cli.js preview 10

# 3. Test critical files
node cli.js test /home/user/logseq/mainKnowledgeGraph/pages/rb-0010-aerial-robot.md

# 4. Dry-run full pipeline
node cli.js process

# 5. Validate current state (before changes)
node cli.js validate

# 6. Process robotics domain only (critical namespace fix)
node cli.js domain robotics --live

# 7. Check results
node cli.js stats

# 8. If satisfied, process remaining domains
node cli.js domain ai --live
node cli.js domain blockchain --live
node cli.js domain metaverse --live

# 9. Final validation
node cli.js validate
```

---

## 🔍 Key Issues Being Fixed

### 1. Robotics Namespace (CRITICAL)
**Issue**: Robotics files use `mv:` namespace instead of `rb:`
**Fix**: Automatically converts `mv:rb0010aerialrobot` → `rb:AerialRobot`
**Files Affected**: ~250 robotics files

### 2. Class Naming (HIGH PRIORITY)
**Issue**: Class names not in CamelCase (e.g., `rb0010aerialrobot`)
**Fix**: Converts to proper CamelCase (e.g., `AerialRobot`)
**Files Affected**: ~290 files

### 3. Status/Maturity Confusion (MEDIUM)
**Issue**: Inconsistent use of status vs maturity fields
**Fix**: Normalizes both fields to standard values
**Files Affected**: ~400 files

### 4. Duplicate Sections (LOW)
**Issue**: Some files have duplicate "Technical Details" sections
**Fix**: Removes duplicate sections
**Files Affected**: ~150 files

---

## 📈 Expected Results

After running the full pipeline:

- **Files Processed**: ~1,450 (files with ontology blocks)
- **Files Updated**: ~1,420 (files needing changes)
- **Files Skipped**: ~15 (files without ontology blocks)
- **Backups Created**: ~1,420
- **Validation Score**: Target 90+/100 average
- **Success Rate**: Target 98%+

---

## ⚡ Performance Notes

- **Scan**: ~2-3 minutes
- **Preview (10 files)**: ~5 seconds
- **Test (1 file)**: ~1 second
- **Dry-Run (all files)**: ~15-20 minutes
- **Live Update (all files)**: ~30-40 minutes
- **Validate**: ~15-20 minutes

**Total Time for Full Migration**: ~1 hour

---

## 🐛 Troubleshooting Quick Fixes

### "Cannot find module"
```bash
cd /home/user/logseq/scripts/ontology-migration
```

### "Permission denied"
```bash
chmod +x *.js
```

### Out of memory
```bash
node --max-old-space-size=4096 cli.js process --live --batch=50
```

### Need to start over
```bash
# Clear reports
rm /home/user/logseq/docs/ontology-migration/reports/*.json

# Re-scan
node cli.js scan
```

---

## 📞 Support

All scripts include detailed error messages and logging. Use `--verbose` or `-v` flag for detailed output:

```bash
node cli.js process --verbose
node cli.js validate -v
```

---

**Pipeline Version**: 1.0.0
**Status**: ✅ Ready for Deployment
**Last Updated**: 2025-11-21
