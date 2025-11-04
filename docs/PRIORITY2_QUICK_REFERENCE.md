# Priority 2 Quick Reference - All Paths at a Glance

**Generated**: 2025-11-04  
**Status**: Ready for Implementation  
**Purpose**: Quick lookup for all 27 broken links requiring correction

---

## All 27 Broken Links - Complete List

### By Severity (13 HIGH, 14 MEDIUM)

#### HIGH SEVERITY (13 links - Blocks User Navigation)

**guides/xr-setup.md** (3 links)
```
Line ~40:  ../concepts/architecture/xr-immersive-system.md
           → ../concepts/architecture/xr-immersive-system.md

Line ~45:  ../concepts/architecture/xr-immersive-system.md
           → ../concepts/architecture/xr-immersive-system.md

Line ~47:  ../concepts/architecture/vircadia-react-xr-integration.md
           → ../concepts/architecture/vircadia-react-xr-integration.md
```

**guides/ontology-storage-guide.md** (3 links)
```
Line ~35:  ../concepts/architecture/ontology-storage-architecture.md
           → ../concepts/architecture/ontology-storage-architecture.md

Line ~37:  ../concepts/architecture/ports/04-ontology-repository.md
           → ../concepts/architecture/ports/04-ontology-repository.md

Line ~95:  ../concepts/architecture/ontology-storage-architecture.md
           → ../concepts/architecture/ontology-storage-architecture.md
```

**guides/navigation-guide.md** (8 links)
```
Line ~32:  architecture/00-ARCHITECTURE-OVERVIEW.md
           → concepts/architecture/00-ARCHITECTURE-OVERVIEW.md

Line ~33:  architecture/xr-immersive-system.md
           → concepts/architecture/xr-immersive-system.md

Line ~48:  architecture/00-ARCHITECTURE-OVERVIEW.md
           → concepts/architecture/00-ARCHITECTURE-OVERVIEW.md

Line ~49:  architecture/hexagonal-cqrs-architecture.md
           → concepts/architecture/hexagonal-cqrs-architecture.md

Line ~51:  architecture/04-database-schemas.md
           → concepts/architecture/04-database-schemas.md

Line ~72:  architecture/gpu/README.md
           → concepts/architecture/gpu/README.md

Line ~74:  architecture/xr-immersive-system.md
           → concepts/architecture/xr-immersive-system.md

Line ~75:  architecture/hexagonal-cqrs-architecture.md
           → concepts/architecture/hexagonal-cqrs-architecture.md
```

---

#### MEDIUM SEVERITY (14 links - Affects Developer/Specialized Docs)

**guides/vircadia-multi-user-guide.md** (2 links)
```
Line ~25:  ../concepts/architecture/vircadia-integration-analysis.md
           → ../concepts/architecture/vircadia-integration-analysis.md

Line ~27:  ../concepts/architecture/voice-webrtc-migration-plan.md
           → ../concepts/architecture/voice-webrtc-migration-plan.md
```

**reference/api/README.md** (1 link)
```
Line ~8:   ../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md
           → ../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md
```

**reference/api/03-websocket.md** (4 links - Mixed Architecture + Reference)
```
Line ~15:  ../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md
           → ../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md

          ../reference/api/binary-protocol.md
           → ./binary-protocol.md

          ../reference/api/rest-api.md
           → ./rest-api.md

          ../reference/performance-benchmarks.md
           → ../performance-benchmarks.md
```

**reference/api/rest-api-complete.md** (1 link)
```
Line ~12:  ../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md
           → ../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md
```

**reference/api/rest-api-reference.md** (2 links)
```
Line ~18:  ../concepts/architecture/ontology-reasoning-pipeline.md
           → ../concepts/architecture/ontology-reasoning-pipeline.md

Line ~20:  ../concepts/architecture/semantic-physics-system.md
           → ../concepts/architecture/semantic-physics-system.md
```

**getting-started/01-installation.md** (1 link)
```
Line ~610: ../concepts/architecture/
           → ../concepts/architecture/
```

**guides/developer/01-development-setup.md** (1 link)
```
Line ~15:  ../../concepts/architecture/
           → ../../concepts/architecture/
```

**guides/migration/json-to-binary-protocol.md** (1 link)
```
Line ~45:  ../../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md
           → ../../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md
```

---

## By File (23 files, 27 links)

### 1. guides/xr-setup.md - 3 links (HIGH)
- `../concepts/architecture/xr-immersive-system.md` (2x)
- `../concepts/architecture/vircadia-react-xr-integration.md` (1x)

### 2. guides/ontology-storage-guide.md - 3 links (HIGH)
- `../concepts/architecture/ontology-storage-architecture.md` (2x)
- `../concepts/architecture/ports/04-ontology-repository.md` (1x)

### 3. guides/navigation-guide.md - 8 links (HIGH)
- `architecture/00-ARCHITECTURE-OVERVIEW.md` (2x)
- `architecture/xr-immersive-system.md` (2x)
- `architecture/hexagonal-cqrs-architecture.md` (2x)
- `architecture/04-database-schemas.md` (1x)
- `architecture/gpu/README.md` (1x)

### 4. guides/vircadia-multi-user-guide.md - 2 links (MEDIUM)
- `../concepts/architecture/vircadia-integration-analysis.md` (1x)
- `../concepts/architecture/voice-webrtc-migration-plan.md` (1x)

### 5. reference/api/README.md - 1 link (MEDIUM)
- `../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md` (1x)

### 6. reference/api/03-websocket.md - 4 links (MEDIUM)
- `../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md` (1x)
- `../reference/api/binary-protocol.md` (1x)
- `../reference/api/rest-api.md` (1x)
- `../reference/performance-benchmarks.md` (1x)

### 7. reference/api/rest-api-complete.md - 1 link (MEDIUM)
- `../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md` (1x)

### 8. reference/api/rest-api-reference.md - 2 links (MEDIUM)
- `../concepts/architecture/ontology-reasoning-pipeline.md` (1x)
- `../concepts/architecture/semantic-physics-system.md` (1x)

### 9. getting-started/01-installation.md - 1 link (MEDIUM)
- `../concepts/architecture/` (1x)

### 10. guides/developer/01-development-setup.md - 1 link (MEDIUM)
- `../../concepts/architecture/` (1x)

### 11. guides/migration/json-to-binary-protocol.md - 1 link (MEDIUM)
- `../../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md` (1x)

---

## Fix Patterns Summary

### Pattern 1: `../concepts/architecture/` → `../concepts/architecture/`
**Files**: 11  
**Links**: 18  
**Examples**:
- `../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md`
- `../concepts/architecture/ontology-storage-architecture.md`
- `../concepts/architecture/semantic-physics-system.md`

### Pattern 2: `architecture/` → `concepts/architecture/` (NO ../)
**Files**: 1 (guides/navigation-guide.md)  
**Links**: 8  
**Examples**:
- `architecture/00-ARCHITECTURE-OVERVIEW.md`
- `architecture/xr-immersive-system.md`

### Pattern 3: `../../concepts/architecture/` → `../../concepts/architecture/`
**Files**: 1 (guides/migration/json-to-binary-protocol.md)  
**Links**: 1  
**Examples**:
- `../../concepts/architecture/00-ARCHITECTURE-OVERVIEW.md`

### Pattern 4: Double-Reference Fixes
**File**: 1 (reference/api/03-websocket.md)  
**Links**: 4  
**Examples**:
- `../reference/api/binary-protocol.md` → `./binary-protocol.md`
- `../reference/api/rest-api.md` → `./rest-api.md`
- `../reference/performance-benchmarks.md` → `../performance-benchmarks.md`

---

## Verification Checklist

After fixes, verify:

- [ ] guides/xr-setup.md - 3 links use `../concepts/architecture/`
- [ ] guides/ontology-storage-guide.md - 3 links use `../concepts/architecture/`
- [ ] guides/navigation-guide.md - 8 links use `concepts/architecture/`
- [ ] guides/vircadia-multi-user-guide.md - 2 links use `../concepts/architecture/`
- [ ] reference/api/README.md - 1 link uses `../concepts/architecture/`
- [ ] reference/api/03-websocket.md - 1 arch link + 3 ref paths fixed
- [ ] reference/api/rest-api-complete.md - 1 link uses `../concepts/architecture/`
- [ ] reference/api/rest-api-reference.md - 2 links use `../concepts/architecture/`
- [ ] getting-started/01-installation.md - 1 link uses `../concepts/architecture/`
- [ ] guides/developer/01-development-setup.md - 1 link uses `../../concepts/architecture/`
- [ ] guides/migration/json-to-binary-protocol.md - 1 link uses `../../concepts/architecture/`
- [ ] NO `../reference/reference/` paths remain
- [ ] ALL 27 links verified in actual markdown viewers

---

## Quick Commands

### Find all broken paths
```bash
grep -r "\.\./architecture/" /home/devuser/workspace/project/docs --include="*.md" | grep -v "concepts"
```

### Find all double-reference paths
```bash
grep -r "\.\./reference/reference/" /home/devuser/workspace/project/docs --include="*.md"
```

### Count fixed architecture paths
```bash
grep -r "\.\./concepts/architecture/" /home/devuser/workspace/project/docs --include="*.md" | wc -l
# Should be 21+ after fixes
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-04  
**Status**: Ready for Reference
