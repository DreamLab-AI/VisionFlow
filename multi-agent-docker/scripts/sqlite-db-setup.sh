#!/bin/bash
# SQLite Database Optimization Setup
# Configures SQLite databases with optimal settings for concurrent access

set -e

configure_sqlite_db() {
    local DB_PATH="$1"

    if [ ! -f "$DB_PATH" ]; then
        echo "Creating database: $DB_PATH"
        touch "$DB_PATH"
    fi

    echo "Configuring SQLite pragmas for: $DB_PATH"

    # Apply performance and concurrency optimizations
    sqlite3 "$DB_PATH" <<EOF
-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;

-- Increase busy timeout to 60 seconds (60000ms)
PRAGMA busy_timeout=60000;

-- Use NORMAL synchronous mode (balance between safety and performance)
PRAGMA synchronous=NORMAL;

-- Set cache size to 10MB (-10000 pages)
PRAGMA cache_size=-10000;

-- Enable memory-mapped I/O (100MB)
PRAGMA mmap_size=104857600;

-- Optimize page size
PRAGMA page_size=4096;

-- Enable automatic checkpointing at 1000 pages
PRAGMA wal_autocheckpoint=1000;

-- Vacuum to optimize
VACUUM;

-- Show current settings
PRAGMA journal_mode;
PRAGMA busy_timeout;
PRAGMA synchronous;
PRAGMA cache_size;
EOF

    echo "✅ Configured: $DB_PATH"
}

# Configure all swarm databases
configure_swarm_databases() {
    echo "🔧 Configuring SQLite databases for multi-agent environment..."

    # Main memory database
    if [ -f "/workspace/.swarm/memory.db" ]; then
        configure_sqlite_db "/workspace/.swarm/memory.db"
    fi

    # TCP server database
    if [ -f "/workspace/.swarm/tcp-server-instance/.swarm/memory.db" ]; then
        configure_sqlite_db "/workspace/.swarm/tcp-server-instance/.swarm/memory.db"
    fi

    # Hook database
    if [ -f "/workspace/.swarm/claude-hooks.db" ]; then
        configure_sqlite_db "/workspace/.swarm/claude-hooks.db"
    fi

    # Configure session databases
    if [ -d "/workspace/.swarm/sessions" ]; then
        find /workspace/.swarm/sessions -name "*.db" -type f | while read -r DB_FILE; do
            configure_sqlite_db "$DB_FILE"
        done
    fi

    echo "✅ All databases configured"
}

# Main
if [ "$1" = "configure" ]; then
    configure_swarm_databases
elif [ -n "$1" ]; then
    configure_sqlite_db "$1"
else
    echo "Usage: $0 configure | <db-path>"
    exit 1
fi
