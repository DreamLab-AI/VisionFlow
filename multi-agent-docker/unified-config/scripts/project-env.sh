#!/bin/bash
# Source this script to set project environment variables
# Usage: source ./project-env.sh /path/to/project

PROJECT_PATH="$1"

if [ -z "$PROJECT_PATH" ]; then
    return 1
fi

GIT_REMOTE=$(git -C "$PROJECT_PATH" config --get remote.origin.url 2>/dev/null || echo "")

if [ -n "$RUVECTOR_PG_CONNINFO" ] && [ -n "$GIT_REMOTE" ]; then
    ID=$(psql "$RUVECTOR_PG_CONNINFO" -t -c "SELECT id FROM projects WHERE git_remote = '$GIT_REMOTE' LIMIT 1;" | xargs)
    if [ -n "$ID" ]; then
        export RUVECTOR_PROJECT_ID="$ID"
        echo "Project ID: $ID"
    fi
fi
