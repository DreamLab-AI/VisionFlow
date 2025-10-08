# Legacy Documentation Archive

**Archive Date**: 2025-10-03
**Archived From**: `docs/docs/`
**Reason**: Documentation consolidation and UK English standardisation

---

## Archive Contents

This directory contains legacy documentation from the nested `docs/docs/` structure that has been consolidated into the main documentation corpus at `docs/`.

### Archived Files

#### Architecture Documentation

- **`architecture/telemetry-system-analysis.md`** - Telemetry system analysis with Mermaid diagrams
  - **Status**: Historical reference, content preserved in comprehensive telemetry guide
  - **New Location**: `docs/guides/telemetry-logging.md`

- **`architecture/API_ARCHITECTURE_ANALYSIS.md`** - API architecture migration analysis
  - **Status**: Converted to ADR format
  - **New Location**: `docs/architecture/decisions/ADR-001-unified-api-client.md`

#### API Documentation

- **`api/unified-api-client.md`** - Unified API client documentation
  - **Status**: Integrated and enhanced
  - **New Location**: `docs/reference/api/client-api.md`

#### Features Documentation

- **`features/polling-system.md`** - Agent swarm polling system
  - **Status**: Enhanced and integrated
  - **New Location**: `docs/reference/polling-system.md`

- **`features/telemetry.md`** - Client-side telemetry features
  - **Status**: Unique content extracted and integrated
  - **New Location**: `docs/guides/telemetry-logging.md` (consolidated)

#### Obsolete Content

- **`Architecture.md`** - Described wrong project (Vircadia Web standalone)
  - **Status**: Deleted as obsolete, not relevant to VisionFlow

---

## Migration Summary

### Content Integration

All relevant content from the legacy documentation has been:

1. **Reviewed for Accuracy**: Verified against current codebase implementation
2. **Enhanced with UK English**: Spelling standardised throughout
3. **Integrated into Main Corpus**: Merged into appropriate documentation sections
4. **Enhanced with Diagrams**: Added or maintained detailed Mermaid diagrams
5. **Cross-Referenced**: Updated navigation links and index entries

### Project Naming Fixes

Fixed incorrect project naming:
- **Incorrect**: "VisionFlow"
- **Correct**: "VisionFlow"

### Quality Improvements

- ✅ UK English spelling standardisation
- ✅ Enhanced Mermaid diagrams where needed
- ✅ Forward/backward navigation links
- ✅ Code examples updated to current implementation
- ✅ API references aligned with actual codebase

---

## Restoration Instructions

If you need to reference or restore any of this legacy content:

1. **Location**: All files are in this archive directory
2. **Preservation**: Content preserved exactly as it existed
3. **Integration**: See new locations listed above for updated versions
4. **Merge History**: Git history preserved for all moved files

---

## Related Documents

- [Documentation Index](../../00-INDEX.md)
- [Migration Summary](../../DOCUMENTATION-MIGRATION-COMPLETE.md)
- [Vircadia Integration Guide](../../guides/xr-quest3-setup.md)

---

**Archive Maintained By**: VisionFlow Documentation Team
**Last Updated**: 2025-10-03
