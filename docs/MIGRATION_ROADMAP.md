# Documentation Migration Roadmap

This document outlines the phased approach to migrate from monolithic files to the **Diátaxis** structure.

## ✅ Phase 1: Foundation (COMPLETED)
- [x] Define Diátaxis framework structure
- [x] Create directories: `getting-started/`, `guides/`, `concepts/`, `reference/`
- [x] Create unified `README.md` as single entry point
- [x] Create `CONTRIBUTING_DOCS.md` with guidelines
- [x] Preserve legacy files for reference

## Phase 2: Content Migration (Planned)

### API.md → reference/api/ (20.5 KB)
- [ ] REST API endpoints
- [ ] WebSocket API spec
- [ ] Binary Protocol (36 bytes per node)
- [ ] API overview & index

### ARCHITECTURE.md → Multiple (31.5 KB) 
- [ ] High-level overview → concepts/architecture.md
- [ ] Hexagonal pattern → reference/architecture/hexagonal-cqrs.md
- [ ] CQRS details → reference/architecture/cqrs.md
- [ ] Actor system → reference/architecture/actor-system.md

### DATABASE.md → reference/architecture/ (13.2 KB)
- [ ] Database schema documentation
- [ ] Design decisions

### DEVELOPER_GUIDE.md → Multiple (35.2 KB - LARGEST)
- [ ] Development setup → guides/developer/
- [ ] Project structure → concepts/
- [ ] Feature development → guides/developer/
- [ ] Testing procedures → guides/developer/
- [ ] Troubleshooting → guides/developer/

## Phase 3: Archival & Cleanup (Planned)
- [ ] Archive old analysis documents
- [ ] Archive migration plans  
- [ ] Consolidate duplicate directories
- [ ] Clean up legacy files

## Priority Order
1. **DEVELOPER_GUIDE.md** (35.2 KB) - Highest impact
2. **ARCHITECTURE.md** (31.5 KB) - Most complex
3. **API.md** (20.5 KB) - Clear boundaries
4. **DATABASE.md** (13.2 KB) - Straightforward
5. **Getting Started** - Consolidation
6. **Archival** - Final cleanup

---

**Status**: Phase 1 ✅ | Phases 2-3 Pending
**Framework**: Diátaxis
**Last Updated**: 2025-10-27
