#!/usr/bin/env python3
"""
Legacy Memory Migration Script for Claude Flow V3

Migrates memory entries from legacy .swarm/memory.db files to V3 format.
Consolidates all workspace project memory into a single V3 database with
namespacing by project.
"""

import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Configuration
WORKSPACE = Path("/home/devuser/workspace")
V3_DB = WORKSPACE / "project" / ".claude" / "memory.db"
LEGACY_PATHS = [
    ".swarm/memory.db",
    ".hive-mind/memory.db",
    ".agentic-qe/memory.db",
    ".claude-flow/memory.db"
]

def get_legacy_databases():
    """Find all legacy memory databases in workspace."""
    databases = []
    for project_dir in WORKSPACE.iterdir():
        if not project_dir.is_dir():
            continue
        for legacy_path in LEGACY_PATHS:
            db_path = project_dir / legacy_path
            if db_path.exists():
                databases.append({
                    "project": project_dir.name,
                    "path": db_path,
                    "type": legacy_path.split("/")[0].replace(".", "")
                })
    return databases

def count_entries(db_path):
    """Count entries in a legacy database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM memory_entries")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except:
        return 0

def migrate_database(legacy_db, v3_conn, project_name, db_type):
    """Migrate entries from legacy database to V3."""
    import uuid
    try:
        legacy_conn = sqlite3.connect(str(legacy_db))
        legacy_conn.row_factory = sqlite3.Row

        # Check what columns exist in the legacy database
        columns_cursor = legacy_conn.execute("PRAGMA table_info(memory_entries)")
        columns = {row[1] for row in columns_cursor.fetchall()}

        # Build SELECT based on available columns
        select_cols = []
        select_cols.append("key" if "key" in columns else "'unknown' as key")
        select_cols.append("value" if "value" in columns else "content as value" if "content" in columns else "'' as value")
        select_cols.append("namespace" if "namespace" in columns else "'default' as namespace")
        select_cols.append("metadata" if "metadata" in columns else "NULL as metadata")
        select_cols.append("created_at" if "created_at" in columns else "NULL as created_at")
        select_cols.append("updated_at" if "updated_at" in columns else "NULL as updated_at")
        select_cols.append("accessed_at" if "accessed_at" in columns else "last_accessed_at as accessed_at" if "last_accessed_at" in columns else "NULL as accessed_at")
        select_cols.append("access_count" if "access_count" in columns else "0 as access_count")
        select_cols.append("ttl" if "ttl" in columns else "NULL as ttl")
        select_cols.append("expires_at" if "expires_at" in columns else "NULL as expires_at")

        # Get all entries from legacy database
        cursor = legacy_conn.execute(f"""
            SELECT {', '.join(select_cols)}
            FROM memory_entries
        """)

        migrated = 0
        skipped = 0

        for row in cursor:
            # Create namespaced key: legacy/project/type/original_namespace
            original_ns = row["namespace"] or "default"
            new_namespace = f"legacy/{project_name}/{db_type}/{original_ns}"

            try:
                # Check if entry exists
                existing = v3_conn.execute(
                    "SELECT id FROM memory_entries WHERE key = ? AND namespace = ?",
                    (row["key"], new_namespace)
                ).fetchone()

                if existing:
                    skipped += 1
                    continue

                # V3 schema columns:
                # id, key, namespace, content, type, embedding, embedding_model,
                # embedding_dimensions, tags, metadata, owner_id,
                # created_at, updated_at, expires_at, last_accessed_at, access_count, status
                #
                # V3 type constraint: 'semantic', 'episodic', 'procedural', 'working', 'pattern'

                # Convert timestamps (legacy uses seconds, V3 uses milliseconds)
                created_ms = (row["created_at"] or 0) * 1000 if row["created_at"] else int(datetime.now().timestamp() * 1000)
                updated_ms = (row["updated_at"] or 0) * 1000 if row["updated_at"] else created_ms
                accessed_ms = (row["accessed_at"] or 0) * 1000 if row["accessed_at"] else None
                expires_ms = (row["expires_at"] or 0) * 1000 if row["expires_at"] else None

                # Generate unique ID
                entry_id = str(uuid.uuid4())

                # Build metadata with migration info
                meta_obj = {}
                if row["metadata"]:
                    try:
                        meta_obj = json.loads(row["metadata"])
                    except:
                        meta_obj = {"original_metadata": row["metadata"]}
                meta_obj["migrated_from"] = f"{project_name}/{db_type}"
                meta_obj["migration_date"] = datetime.now().isoformat()

                # Insert into V3 database with correct column mapping
                v3_conn.execute("""
                    INSERT INTO memory_entries
                    (id, key, namespace, content, type, embedding, embedding_model,
                     embedding_dimensions, tags, metadata, owner_id,
                     created_at, updated_at, expires_at, last_accessed_at, access_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id,
                    row["key"],
                    new_namespace,
                    row["value"],  # content
                    "episodic",  # type - using valid V3 type for historical data
                    None,  # embedding (will be generated on access)
                    None,  # embedding_model
                    None,  # embedding_dimensions
                    json.dumps(["migrated", project_name, db_type]),  # tags
                    json.dumps(meta_obj),
                    f"migration-{project_name}",  # owner_id
                    created_ms,
                    updated_ms,
                    expires_ms,
                    accessed_ms,  # last_accessed_at
                    row["access_count"] or 0,
                    "active"  # status
                ))
                migrated += 1

            except sqlite3.IntegrityError as e:
                skipped += 1
            except Exception as e:
                # Log other errors but continue
                pass

        legacy_conn.close()
        v3_conn.commit()
        return migrated, skipped

    except Exception as e:
        print(f"  Error migrating {legacy_db}: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

def main():
    print("=" * 60)
    print("Claude Flow V3 Legacy Memory Migration")
    print("=" * 60)
    print()

    # Check V3 database exists
    if not V3_DB.exists():
        print(f"Error: V3 database not found at {V3_DB}")
        print("Run 'claude-flow memory init' first")
        sys.exit(1)

    # Find all legacy databases
    databases = get_legacy_databases()
    print(f"Found {len(databases)} legacy memory databases")
    print()

    # Show summary
    total_entries = 0
    print("Legacy databases to migrate:")
    print("-" * 60)
    for db in databases:
        count = count_entries(db["path"])
        size = db["path"].stat().st_size / 1024 / 1024  # MB
        total_entries += count
        print(f"  {db['project']}/{db['type']}: {count:,} entries ({size:.1f}MB)")
    print("-" * 60)
    print(f"Total: {total_entries:,} entries")
    print()

    # Connect to V3 database
    v3_conn = sqlite3.connect(str(V3_DB))

    # Migrate each database
    print("Migrating...")
    total_migrated = 0
    total_skipped = 0

    for db in databases:
        count = count_entries(db["path"])
        if count == 0:
            continue

        print(f"  {db['project']}/{db['type']}...", end=" ", flush=True)
        migrated, skipped = migrate_database(
            db["path"], v3_conn, db["project"], db["type"]
        )
        total_migrated += migrated
        total_skipped += skipped
        print(f"{migrated:,} migrated, {skipped:,} skipped")

    v3_conn.close()

    print()
    print("=" * 60)
    print(f"Migration Complete!")
    print(f"  Migrated: {total_migrated:,} entries")
    print(f"  Skipped:  {total_skipped:,} entries (duplicates)")
    print(f"  Target:   {V3_DB}")
    print("=" * 60)

    # Verify
    v3_conn = sqlite3.connect(str(V3_DB))
    cursor = v3_conn.execute("SELECT COUNT(*) FROM memory_entries")
    final_count = cursor.fetchone()[0]
    v3_conn.close()
    print(f"\nV3 database now has {final_count:,} total entries")

if __name__ == "__main__":
    main()
