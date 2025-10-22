# Cargo Check Validation Logs

This directory contains the raw output from cargo check runs across different feature combinations.

## Files

- `cargo_check_default.log` - Default features (no features enabled)
- `cargo_check_gpu.log` - GPU features enabled
- `cargo_check_ontology.log` - Ontology features enabled
- `cargo_check_all_features.log` - All features enabled

## Usage

These logs are useful for:
- Detailed error analysis
- Tracking fix progress
- Before/after comparison
- Automated parsing

## Statistics

| Log File | Errors | Warnings | Size |
|----------|--------|----------|------|
| default | 353 | 194 | ~200KB |
| gpu | 353 | 194 | ~200KB |
| ontology | 353 | 194 | ~200KB |
| all_features | 361 | 193 | ~210KB |

## Parsing Examples

```bash
# Count errors by type
grep "^error\[E" cargo_check_default.log | cut -d: -f1 | sort | uniq -c

# Find all E0437 errors
grep "E0437" cargo_check_default.log

# List affected files
grep "^  --> src/" cargo_check_default.log | cut -d: -f1 | sort -u

# Count warnings
grep -c "^warning:" cargo_check_default.log
```

---

Generated: 2025-10-22
