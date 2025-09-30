#!/bin/bash
# Mount validation script
# Ensures all required mount points exist and have correct permissions

set -e

REQUIRED_MOUNTS=(
    "/workspace:rw"
    "/workspace/ext:rw"
    "/home/dev/.claude:ro"
)

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_success() {
    echo "[SUCCESS] $1"
}

validate_mount() {
    local mount_spec="$1"
    local mount_path="${mount_spec%:*}"
    local mount_perms="${mount_spec#*:}"

    # Check if path exists
    if [ ! -e "$mount_path" ]; then
        log_error "Mount point does not exist: $mount_path"
        return 1
    fi

    # Check read permissions
    if [ ! -r "$mount_path" ]; then
        log_error "No read permission for: $mount_path"
        return 1
    fi

    # Check write permissions if required
    if [[ "$mount_perms" == "rw" ]]; then
        if [ ! -w "$mount_path" ]; then
            log_error "No write permission for: $mount_path"
            return 1
        fi
    fi

    log_success "Mount validated: $mount_path ($mount_perms)"
    return 0
}

main() {
    log_info "Validating mount points..."

    local failed=0
    for mount in "${REQUIRED_MOUNTS[@]}"; do
        if ! validate_mount "$mount"; then
            failed=$((failed + 1))
        fi
    done

    if [ $failed -eq 0 ]; then
        log_success "All mounts validated successfully"
        return 0
    else
        log_error "$failed mount validation(s) failed"
        return 1
    fi
}

main "$@"