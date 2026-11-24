---
id: VC-62
title: SQLite and Oxigraph backup at visionclaw bc6ac309e — successors to VC-22.7 and VC-22.11
area: visionclaw
governing:
  - ../project/docs/DATA-authority-erasure.md
adrs: [ADR-2017, ADR-2069, ADR-2103]
sources:
  - ../project/docs/DATA-authority-erasure.md
  - ../project/docs/adr/ADR-2103-oxigraph-point-in-time-backup.md
  - ../project/scripts/backup-sqlite.sh
  - ../project/Cargo.toml
  - ../project/src/main.rs
  - ../project/src/services/ontology_checkpoint.rs
  - ../project/docker-compose.unified.yml
verified_commit: {visionclaw: bc6ac309e0c9e6c415b24d98a9ba1b7fdfd050f8}
---

## VC-62.1 SQLite online backup (`scripts/backup-sqlite.sh`, ADR-2017) — ADR-2103 activation live
```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator/cron
    participant SH as backup-sqlite.sh
    participant CT as visionclaw_container
    participant SQL as sqlite3 CLI in container

    OP->>SH: scripts/backup-sqlite.sh (scripts/backup-sqlite.sh:1)
    SH->>SH: detect_mode docker or host (backup-sqlite.sh:138, :147)
    loop for each db in REQUIRED_DBS + OPTIONAL_DBS (backup-sqlite.sh:75-76)
        SH->>CT: docker exec test -f db (backup_one_docker, backup-sqlite.sh:173)
        alt db present
            SH->>SQL: sqlite3 db .backup tmp (ONLINE BACKUP API)
            SQL-->>SH: PRAGMA integrity_check = ok
            SH->>SH: docker cp tmp to DEST/db
            opt VERIFY_RESTORE=1 (default)
                SH->>SH: verify_restore copy to scratch, SELECT count from expected tables (backup-sqlite.sh:111-130)
            end
        else db absent
            alt is_required(db) (backup-sqlite.sh:88)
                SH->>SH: record missing_required (backup-sqlite.sh:223-224)
            else optional db missing
                SH->>SH: record missing_optional, continue (backup-sqlite.sh:226-228), logged after the loop (:248)
            end
        end
    end
    alt missing_required non-empty (backup-sqlite.sh:243-246)
        SH--xOP: die required database(s) missing, refuse to publish incomplete backup set, no manifest written
    else all required present
        SH->>SH: write MANIFEST.txt with sha256 per db
        SH-->>OP: DONE - N databases in DEST
    end
    Note over SH: ADR-2103 (Oxigraph point-in-time backup) is proposed, implementation partial, activation LIVE<br/>(docs/adr/ADR-2103-oxigraph-point-in-time-backup.md:5-7), moved to live on 2026-11-23 on deployment and runtime evidence<br/>(ADR-2103-oxigraph-point-in-time-backup.md:206, :232), RPO/RTO figures and the per-graph restore policy are still open (:233-237)
    Note over SH: Oxigraph is backed up outside this script: the visionclaw-backup crate is a workspace member on main (Cargo.toml:21)<br/>and the server checkpoints its own store on a timer (main.rs:755-759, ontology_checkpoint.rs:80-95), compose default dir<br/>and 21600 s interval (docker-compose.unified.yml:27-28), the repository schedules nothing, the host runs a nightly timer<br/>(DATA-authority-erasure.md:118-121)
```

**What it shows.** Successor to the frozen VC-22.7, re-read at visionclaw bc6ac309e (VC-22 stays at 4900ec60b). The SQLite path is unchanged; ADR-2103 is no longer drawn as proposed and partial only: its activation axis moved to live on 2026-11-23, on a running six-hourly checkpoint and the host's nightly timer and timed restore (`../project/docs/adr/ADR-2103-oxigraph-point-in-time-backup.md:218-232`), while the decision stays proposed and implementation partial.

## VC-62.2 SQLite membership contract vs KPI/enrichment sole-source risk — the Oxigraph backup text now matches main
```mermaid
flowchart TD
    Script["backup-sqlite.sh<br/>REQUIRED_DBS default: settings.sqlite3 enrichment.sqlite3 kpi.sqlite3 shares.sqlite3<br/>scripts/backup-sqlite.sh:75"] --> Req["REQUIRED - missing member fails whole run, no manifest written<br/>scripts/backup-sqlite.sh:243-246"]
    Script --> Opt["OPTIONAL_DBS default: liveness.sqlite3 - missing is logged, run still succeeds<br/>scripts/backup-sqlite.sh:76,248"]
    Req --> Verify["VERIFY_RESTORE=1 default: scratch-restore + SELECT count on expected_tables_for<br/>scripts/backup-sqlite.sh:97-106,110-131"]
    Verify --> Manifest["MANIFEST.txt sha256 per db<br/>scripts/backup-sqlite.sh:252-264"]
    Opt -.-> Missing["missing_optional logged, count still gt 0 required<br/>scripts/backup-sqlite.sh:248-249"]

    DriftNote["ADR-2069: the required/optional membership contract is the design and DATA-authority-erasure.md describes it -<br/>a missing REQUIRED_DBS member (settings, enrichment, kpi, shares, backup-sqlite.sh:75) aborts the run and publishes no manifest (:243-246),<br/>a missing OPTIONAL_DBS member (liveness, :76) is recorded (:226-228), logged and the run continues (:248).<br/>The doc now also says the Oxigraph backup tool and checkpoint task are on main since 4f02b817c<br/>(DATA-authority-erasure.md:110-119), matching the workspace member at Cargo.toml:21 and the boot spawn<br/>(main.rs:755-759, ontology_checkpoint.rs:89-95) that compose enables (docker-compose.unified.yml:27-28), see VC-62.1"]
    Req -.-> DriftNote
    Verify -.-> NoSchema["shares.sqlite3 has an expected_tables_for arm (since visionclaw 46edc196),<br/>so its restore check runs against shared_graphs and schema_migrations<br/>scripts/backup-sqlite.sh:97-106, :102, :114"]
```

**What it shows.** Successor to the frozen VC-22.11. The DRIFT it recorded is closed: VisionClaw PR #50 corrected the governing doc, which now says the `visionclaw-backup` tool and the checkpoint task are on main since `4f02b817c`, checkpointing is on by default in compose and the host runs a nightly timer (`../project/docs/DATA-authority-erasure.md:110-128`). The doc keeps the remaining gap: no cross-store consistent restore and no ratified RPO/RTO (`../project/docs/DATA-authority-erasure.md:127-128`).
