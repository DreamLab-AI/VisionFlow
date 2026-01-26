#!/bin/bash
# Identify and register project in RuVector
# Usage: ./project-identify.sh /path/to/project [identify|register]

PROJECT_PATH="$1"
ACTION="${2:-identify}"

if [ -z "$PROJECT_PATH" ]; then
    echo "Usage: $0 <path> [identify|register]"
    exit 1
fi

# Get Git remote
GIT_REMOTE=$(git -C "$PROJECT_PATH" config --get remote.origin.url 2>/dev/null || echo "")

# Get Package Name
PKG_NAME=$(jq -r .name "$PROJECT_PATH/package.json" 2>/dev/null || echo "")

echo "Project Info:"
echo "  Path: $PROJECT_PATH"
echo "  Git:  $GIT_REMOTE"
echo "  Pkg:  $PKG_NAME"

if [ "$ACTION" == "register" ]; then
    echo "Registering project in RuVector..."
    # SQL to insert/update
    if [ -n "$RUVECTOR_PG_CONNINFO" ]; then
        psql "$RUVECTOR_PG_CONNINFO" -c "INSERT INTO projects (git_remote, pkg_name, path) VALUES ('$GIT_REMOTE', '$PKG_NAME', '$PROJECT_PATH') ON CONFLICT DO NOTHING;"
    else
        echo "Warning: RUVECTOR_PG_CONNINFO not set, skipping DB registration"
    fi
fi
