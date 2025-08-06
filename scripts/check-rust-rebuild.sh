#!/bin/bash
# Check if Rust source code has changed and rebuild if necessary

set -e

# Function to log messages
log() {
    echo "[check-rust-rebuild] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check if source files are newer than the binary
check_rebuild_needed() {
    local BINARY="/app/webxr"
    local REBUILD_NEEDED=0
    
    # If binary doesn't exist, we need to build
    if [ ! -f "$BINARY" ]; then
        log "Binary not found, rebuild needed"
        return 0
    fi
    
    # Get binary modification time
    BINARY_TIME=$(stat -c %Y "$BINARY" 2>/dev/null || echo 0)
    
    # Check if any Rust source file is newer than the binary
    while IFS= read -r -d '' file; do
        FILE_TIME=$(stat -c %Y "$file" 2>/dev/null || echo 0)
        if [ "$FILE_TIME" -gt "$BINARY_TIME" ]; then
            log "Source file $file is newer than binary"
            REBUILD_NEEDED=1
            break
        fi
    done < <(find /app/src -name "*.rs" -print0)
    
    # Check if Cargo.toml or Cargo.lock is newer
    for cargo_file in /app/Cargo.toml /app/Cargo.lock; do
        if [ -f "$cargo_file" ]; then
            FILE_TIME=$(stat -c %Y "$cargo_file" 2>/dev/null || echo 0)
            if [ "$FILE_TIME" -gt "$BINARY_TIME" ]; then
                log "$cargo_file is newer than binary"
                REBUILD_NEEDED=1
                break
            fi
        fi
    done
    
    return $REBUILD_NEEDED
}

# Rebuild the Rust binary
rebuild_rust() {
    log "Starting Rust rebuild with GPU features..."
    
    cd /app
    
    # Build with GPU features
    if cargo build --features gpu; then
        cp target/debug/webxr /app/webxr
        log "Rust rebuild completed successfully"
        return 0
    else
        log "ERROR: Rust rebuild failed"
        return 1
    fi
}

# Main execution
main() {
    if check_rebuild_needed; then
        log "Rebuild is needed"
        if rebuild_rust; then
            log "Rebuild successful"
            exit 0
        else
            log "Rebuild failed"
            exit 1
        fi
    else
        log "Binary is up to date, no rebuild needed"
        exit 0
    fi
}

# Run main function
main "$@"