#!/usr/bin/env bash
# WS-J hygiene — DRY RUN unless APPLY=1. Executed by the coordinator after waves 1–2 report green.
set -euo pipefail
W=/home/devuser/workspace
say(){ printf '%s\n' "$*"; }
run(){ if [ "${APPLY:-0}" = 1 ]; then "$@"; else say "  would: $*"; fi; }
say "1. stale converter snapshots";            run rm -rf "$W/vault" "$W/vault-working"
say "2. logseq symlink (host bind stays until agentbox compose is edited — reported, not removed here)"; run rm -f "$W/logseq"
say "3. legacy publisher dirs";                run rm -rf "$W/logseq-publisher" "$W/logseq-publisher-rust" "$W/logseq-publisher-npm"
say "4. knowledgeGraph: archive Logseq source + pipeline with marker"
run mkdir -p "$W/knowledgeGraph/archive/logseq-era-2026-09-22"
for p in ontology/pages pipeline; do run git -C "$W/knowledgeGraph" mv "$p" "archive/logseq-era-2026-09-22/$(basename $p)"; done
run bash -c "cat > '$W/knowledgeGraph/archive/logseq-era-2026-09-22/README.md' <<'M'
# Archived 2026-09-22 — Logseq-era corpus source and pipeline
Superseded by the Obsidian-authored corpus (jjohare/visionGraph) built by \`vault build\` and published by Quartz.
See VisionFlow/docs/PRD-sovereign-corpus.md (Q3). Kept for history; never a build input.
M"
say "5. visionGraph: Python pipeline + publishing-tools (only after WS-D confirms vault build parity)"; run git -C "$W/visionGraph" rm -rq pipeline publishing-tools
say "6. VisionClaw: crates/vault-migrate (only after WS-C confirms nothing is lifted from it)"; run git -C "$W/project" rm -rq crates/vault-migrate
say "7. residue grep (must be history-only):"
grep -rIl -i logseq "$W/loom" "$W/project/src" "$W/project/crates" "$W/project/agentbox/skills" "$W/project/agentbox/mcp" "$W/VisionFlow/README.md" "$W/visionGraph/transcripts" 2>/dev/null | grep -vE "archive|legacy|/docs/adr/|node_modules|target|\.git/" || say "  none"
say "8. POST-MIGRATION ONLY (after WS-D validate is green): remove the last live key:: readers in VisionClaw —"
say "   visionclaw_domain::vault::parse (ADR-2040 D3 publish gate branch for 'public:: true'), KnowledgeGraphParser:432,"
say "   services::parsers::ontology_parser '### OntologyBlock' path; and delete tests/integration/ontology_pipeline_e2e_test.rs (no compiled target)."
say "   Then swap services::page_parser::parse_page body to vault_core::parse_page and drop the json-ld fence parser (ADR-2114 note)."
