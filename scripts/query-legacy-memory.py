#!/usr/bin/env python3
"""
Query Legacy Memory - Search migrated memory entries

Usage:
    python3 query-legacy-memory.py search "query string"
    python3 query-legacy-memory.py list [namespace] [limit]
    python3 query-legacy-memory.py namespaces
    python3 query-legacy-memory.py stats
"""

import sqlite3
import sys
import json
from pathlib import Path

DB_PATH = Path("/home/devuser/workspace/project/.claude/memory.db")

def connect():
    return sqlite3.connect(str(DB_PATH))

def search(query, limit=20):
    """Search memory entries by content."""
    conn = connect()
    cursor = conn.execute("""
        SELECT namespace, key, SUBSTR(content, 1, 200) as content_preview,
               datetime(created_at/1000, 'unixepoch') as created
        FROM memory_entries
        WHERE content LIKE ?
        LIMIT ?
    """, (f"%{query}%", limit))

    results = cursor.fetchall()
    conn.close()

    print(f"Found {len(results)} results for '{query}':")
    print("-" * 80)
    for ns, key, content, created in results:
        print(f"[{ns}] {key}")
        print(f"  Created: {created}")
        print(f"  Content: {content[:150]}...")
        print()

def list_entries(namespace=None, limit=20):
    """List memory entries."""
    conn = connect()
    if namespace:
        cursor = conn.execute("""
            SELECT namespace, key, SUBSTR(content, 1, 100) as content_preview
            FROM memory_entries
            WHERE namespace LIKE ?
            LIMIT ?
        """, (f"%{namespace}%", limit))
    else:
        cursor = conn.execute("""
            SELECT namespace, key, SUBSTR(content, 1, 100) as content_preview
            FROM memory_entries
            LIMIT ?
        """, (limit,))

    results = cursor.fetchall()
    conn.close()

    print(f"Listing {len(results)} entries:")
    print("-" * 80)
    for ns, key, content in results:
        print(f"[{ns}] {key}")
        if content:
            print(f"  {content[:80]}...")
        print()

def namespaces():
    """List all namespaces with counts."""
    conn = connect()
    cursor = conn.execute("""
        SELECT namespace, COUNT(*) as count
        FROM memory_entries
        GROUP BY namespace
        ORDER BY count DESC
    """)

    results = cursor.fetchall()
    conn.close()

    print(f"Namespaces ({len(results)} total):")
    print("-" * 60)
    for ns, count in results:
        print(f"  {count:>8}  {ns}")

def stats():
    """Show database statistics."""
    conn = connect()

    total = conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
    with_content = conn.execute("SELECT COUNT(*) FROM memory_entries WHERE content IS NOT NULL AND content != ''").fetchone()[0]

    # By project
    cursor = conn.execute("""
        SELECT
            CASE
                WHEN namespace LIKE 'legacy/%' THEN
                    SUBSTR(namespace, 8, INSTR(SUBSTR(namespace, 8), '/') - 1)
                ELSE namespace
            END as project,
            COUNT(*) as count
        FROM memory_entries
        GROUP BY project
        ORDER BY count DESC
        LIMIT 20
    """)
    projects = cursor.fetchall()

    conn.close()

    print("Memory Statistics")
    print("=" * 60)
    print(f"Total entries: {total:,}")
    print(f"With content: {with_content:,}")
    print(f"Database size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("By Project:")
    print("-" * 60)
    for proj, count in projects:
        print(f"  {count:>8}  {proj}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "search":
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        search(query, limit)
    elif cmd == "list":
        namespace = sys.argv[2] if len(sys.argv) > 2 else None
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        list_entries(namespace, limit)
    elif cmd == "namespaces":
        namespaces()
    elif cmd == "stats":
        stats()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)

if __name__ == "__main__":
    main()
