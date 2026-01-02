#!/bin/bash
# Sync JavaScriptSolidServer from upstream
# Usage: ./scripts/sync-jss-upstream.sh

set -e

cd "$(dirname "$0")/.."

echo "🔄 Syncing JavaScriptSolidServer from upstream..."

# Check if remote exists
if ! git remote | grep -q "jss-upstream"; then
    echo "Adding jss-upstream remote..."
    git remote add jss-upstream git@github.com:JavaScriptSolidServer/JavaScriptSolidServer.git
fi

# Fetch latest
git fetch jss-upstream

# Show what would be updated
echo "📋 Changes available from upstream:"
git log HEAD..jss-upstream/main --oneline 2>/dev/null || echo "No new commits or branch not found"

# Prompt for confirmation
read -p "Proceed with subtree pull? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git subtree pull --prefix=JavaScriptSolidServer jss-upstream main --squash \
        -m "chore: sync JavaScriptSolidServer from upstream"
    echo "✅ Sync complete"
else
    echo "⏭️ Sync skipped"
fi
