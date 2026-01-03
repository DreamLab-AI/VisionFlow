#!/bin/bash
# sync-jss-upstream.sh - Sync JavaScriptSolidServer from upstream
#
# Usage:
#   ./scripts/sync-jss-upstream.sh           # Check for updates
#   ./scripts/sync-jss-upstream.sh v0.0.46   # Sync to specific version
#   ./scripts/sync-jss-upstream.sh --pull    # Pull latest main branch
#
# AGPL-3.0: This script helps maintain upstream sync for JavaScriptSolidServer
# See: https://github.com/JavaScriptSolidServer/JavaScriptSolidServer

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
JSS_DIR="$PROJECT_ROOT/JavaScriptSolidServer"
REMOTE="jss-upstream"
LOCAL_CHANGES="$JSS_DIR/LOCAL_CHANGES.md"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  JavaScriptSolidServer Upstream Sync Tool${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Check if remote exists
if ! git remote | grep -q "^${REMOTE}$"; then
    echo -e "${YELLOW}Adding jss-upstream remote...${NC}"
    git remote add "$REMOTE" git@github.com:JavaScriptSolidServer/JavaScriptSolidServer.git
fi

# Fetch upstream
echo -e "${BLUE}Fetching from upstream...${NC}"
git fetch "$REMOTE" --tags 2>/dev/null || {
    echo -e "${RED}Failed to fetch from upstream. Check network/SSH keys.${NC}"
    exit 1
}

# Get current vendored version
CURRENT_VERSION=$(grep -oP 'Base Version:\*\* v?\K[0-9.]+' "$LOCAL_CHANGES" 2>/dev/null || echo "unknown")
echo -e "${GREEN}Current vendored version:${NC} v$CURRENT_VERSION"

# Get latest upstream version
LATEST_TAG=$(git tag -l 'v*' --sort=-v:refname | head -1)
LATEST_VERSION=${LATEST_TAG#v}
echo -e "${GREEN}Latest upstream version:${NC} $LATEST_TAG"

# Compare versions
if [ "$CURRENT_VERSION" = "$LATEST_VERSION" ]; then
    echo -e "${GREEN}✓ Already at latest version${NC}"
else
    echo -e "${YELLOW}⚠ Update available: v$CURRENT_VERSION → $LATEST_TAG${NC}"
fi

# Show what's new
echo ""
echo -e "${BLUE}Recent upstream commits:${NC}"
git log --oneline "$REMOTE/main" -10 2>/dev/null || \
git log --oneline "$REMOTE/master" -10 2>/dev/null || \
echo -e "${YELLOW}  (Unable to show commits - try running 'git fetch $REMOTE' first)${NC}"

# Handle arguments
case "${1:-}" in
    --pull)
        echo ""
        echo -e "${YELLOW}Pulling latest from upstream main branch...${NC}"
        echo -e "${RED}WARNING: This will use git subtree pull. Review changes carefully!${NC}"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git subtree pull --prefix=JavaScriptSolidServer "$REMOTE" main --squash \
                -m "chore: sync JavaScriptSolidServer from upstream"
            echo -e "${GREEN}✓ Subtree pull complete${NC}"
            echo -e "${YELLOW}Remember to update LOCAL_CHANGES.md with new version!${NC}"
        else
            echo "Aborted."
        fi
        ;;
    v*)
        TARGET_VERSION="$1"
        echo ""
        echo -e "${YELLOW}Syncing to specific version: $TARGET_VERSION${NC}"

        # Verify tag exists
        if ! git tag -l | grep -q "^${TARGET_VERSION}$"; then
            echo -e "${RED}Error: Tag $TARGET_VERSION not found upstream${NC}"
            echo "Available versions:"
            git tag -l 'v*' --sort=-v:refname | head -10
            exit 1
        fi

        echo -e "${RED}WARNING: This will checkout JSS at $TARGET_VERSION${NC}"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Create a temporary branch from the tag
            git subtree pull --prefix=JavaScriptSolidServer "$REMOTE" "$TARGET_VERSION" --squash \
                -m "chore: sync JavaScriptSolidServer to $TARGET_VERSION"
            echo -e "${GREEN}✓ Synced to $TARGET_VERSION${NC}"
            echo -e "${YELLOW}Remember to update LOCAL_CHANGES.md!${NC}"
        else
            echo "Aborted."
        fi
        ;;
    "")
        # Default: just show status (already done above)
        echo ""
        echo -e "${BLUE}Usage:${NC}"
        echo "  $0           Check for updates (this output)"
        echo "  $0 --pull    Pull latest from main branch"
        echo "  $0 v0.0.46   Sync to specific version tag"
        echo ""
        echo -e "${YELLOW}Note:${NC} After syncing, review JavaScriptSolidServer/LOCAL_CHANGES.md"
        echo "and reapply any local modifications if needed."
        ;;
    *)
        echo -e "${RED}Unknown argument: $1${NC}"
        echo "Usage: $0 [--pull|vX.X.X]"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
