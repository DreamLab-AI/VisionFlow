#!/bin/bash
# Script to update Claude Code authentication credentials
# Can be run inside the container to refresh credentials

set -e

# Check if running as root, switch to dev user if needed
if [ "$(id -u)" = "0" ]; then
    exec su - dev -c "$0 $@"
fi

# Function to update credentials
update_credentials() {
    local access_token="$1"
    local refresh_token="$2"
    
    if [ -z "$access_token" ] || [ -z "$refresh_token" ]; then
        echo "Usage: $0 <access_token> <refresh_token>"
        echo "Or set CLAUDE_CODE_ACCESS and CLAUDE_CODE_REFRESH environment variables"
        exit 1
    fi
    
    # Calculate expiry time (30 days from now)
    local expires_at=$(($(date +%s) * 1000 + 2592000000))
    
    # Update credentials for both possible locations
    for user_home in /home/dev /home/ubuntu; do
        if [ -d "$user_home" ]; then
            mkdir -p "$user_home/.claude"
            
            # Create new credentials file
            cat > "$user_home/.claude/.credentials.json" << EOF
{
  "claudeAiOauth": {
    "accessToken": "$access_token",
    "refreshToken": "$refresh_token",
    "expiresAt": $expires_at,
    "scopes": ["user:inference", "user:profile"],
    "subscriptionType": "max"
  }
}
EOF
            
            # Set proper permissions
            chmod 600 "$user_home/.claude/.credentials.json"
            echo "✅ Updated credentials in $user_home/.claude/"
        fi
    done
    
    # Test if claude is authenticated
    if command -v claude >/dev/null 2>&1; then
        echo ""
        echo "Testing Claude authentication..."
        if claude --version >/dev/null 2>&1; then
            echo "✅ Claude is authenticated and working!"
        else
            echo "⚠️  Claude command exists but authentication may have issues"
        fi
    else
        echo "⚠️  Claude command not found. You may need to install it."
    fi
}

# Main execution
if [ $# -eq 2 ]; then
    # Arguments provided
    update_credentials "$1" "$2"
elif [ -n "$CLAUDE_CODE_ACCESS" ] && [ -n "$CLAUDE_CODE_REFRESH" ]; then
    # Environment variables provided
    update_credentials "$CLAUDE_CODE_ACCESS" "$CLAUDE_CODE_REFRESH"
else
    # Interactive mode
    echo "Claude Code Authentication Updater"
    echo "================================="
    echo ""
    echo "Enter your Claude Code credentials:"
    echo "(You can find these in ~/.claude/.credentials.json on your host)"
    echo ""
    
    read -p "Access Token (sk-ant-oat01-...): " access_token
    read -p "Refresh Token (sk-ant-ort01-...): " refresh_token
    
    update_credentials "$access_token" "$refresh_token"
fi