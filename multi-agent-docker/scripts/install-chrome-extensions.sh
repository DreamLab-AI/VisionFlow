#!/bin/bash
set -e

# Chrome Extension Installer for Docker Container
# Installs Chrome extensions from the Chrome Web Store

EXTENSION_LOG="/workspace/logs/chrome-extensions.log"
mkdir -p "$(dirname "$EXTENSION_LOG")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$EXTENSION_LOG"
}

log "Starting Chrome extension installation..."

# Claude extension ID: fcoeoabgfenejglbffodgkkbkcdhcgfn
CLAUDE_EXTENSION_ID="fcoeoabgfenejglbffodgkkbkcdhcgfn"

# Get Chrome profile directory
CHROME_USER_DATA_DIR="/home/dev/.config/chromium"
mkdir -p "$CHROME_USER_DATA_DIR/Default/Extensions"

log "Chrome user data directory: $CHROME_USER_DATA_DIR"

install_extension() {
    local extension_id="$1"
    local extension_name="$2"

    log "Installing extension: $extension_name ($extension_id)"

    # Create extension directory
    local ext_dir="$CHROME_USER_DATA_DIR/Default/Extensions/$extension_id"
    mkdir -p "$ext_dir"

    # Download the CRX file from Chrome Web Store
    local crx_url="https://clients2.google.com/service/update2/crx?response=redirect&prodversion=120.0.0.0&acceptformat=crx2,crx3&x=id%3D${extension_id}%26uc"
    local crx_file="/tmp/${extension_id}.crx"

    log "Downloading extension from Chrome Web Store..."
    if curl -L -o "$crx_file" "$crx_url" 2>&1 | tee -a "$EXTENSION_LOG"; then
        log "Downloaded extension CRX to $crx_file"

        # Extract CRX (it's a ZIP file with header)
        # Skip the CRX header (first 16 bytes for CRX2, variable for CRX3)
        local temp_dir="/tmp/${extension_id}_extract"
        mkdir -p "$temp_dir"

        # Try to extract as ZIP (skip CRX header)
        log "Extracting extension..."
        if unzip -q -o "$crx_file" -d "$temp_dir" 2>/dev/null || \
           (dd if="$crx_file" bs=1 skip=16 2>/dev/null | unzip -q - -d "$temp_dir" 2>/dev/null); then

            # Find version from manifest
            if [ -f "$temp_dir/manifest.json" ]; then
                local version=$(jq -r '.version' "$temp_dir/manifest.json" 2>/dev/null || echo "1.0.0")
                local version_dir="$ext_dir/$version"

                mkdir -p "$version_dir"
                cp -r "$temp_dir"/* "$version_dir/"

                log "Installed extension $extension_name version $version"
            else
                log "WARNING: manifest.json not found for $extension_name"
            fi
        else
            log "WARNING: Failed to extract extension $extension_name"
        fi

        # Cleanup
        rm -rf "$temp_dir" "$crx_file"
    else
        log "WARNING: Failed to download extension $extension_name"
    fi
}

# Install Claude extension
install_extension "$CLAUDE_EXTENSION_ID" "Claude"

# Create preferences file to enable extensions
log "Configuring Chrome preferences..."
PREFS_FILE="$CHROME_USER_DATA_DIR/Default/Preferences"
if [ ! -f "$PREFS_FILE" ]; then
    cat > "$PREFS_FILE" <<EOF
{
   "extensions": {
      "settings": {
         "$CLAUDE_EXTENSION_ID": {
            "state": 1,
            "location": 1,
            "manifest": {},
            "path": "$CLAUDE_EXTENSION_ID"
         }
      }
   }
}
EOF
    log "Created Chrome preferences file"
else
    log "Chrome preferences already exist"
fi

# Set permissions
chown -R dev:dev "$CHROME_USER_DATA_DIR"
chmod -R 755 "$CHROME_USER_DATA_DIR"

log "Chrome extension installation complete"
log "Extensions will be available after Chromium restart"
