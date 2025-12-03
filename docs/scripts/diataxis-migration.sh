#!/bin/bash

echo "=== Diátaxis Documentation Restructuring ==="
echo "Phase 1 & 2: Sanitation and Structural Reorganization"
echo ""

# Track operations
MOVED=0
RENAMED=0
CREATED=0

# Helper function for safe moves
safe_move() {
    local src="$1"
    local dest="$2"

    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dest")"
        mv "$src" "$dest"
        echo "✓ Moved: $src -> $dest"
        MOVED=$((MOVED + 1))
    else
        echo "✗ Not found: $src"
    fi
}

# Helper for directory creation
ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "✓ Created directory: $1"
        CREATED=$((CREATED + 1))
    fi
}

echo "Step 1: Creating target directory structure..."
echo "-------------------------------------------"

# Create main structure
ensure_dir "docs/tutorials"
ensure_dir "docs/guides/developer"
ensure_dir "docs/guides/operations"
ensure_dir "docs/guides/infrastructure"
ensure_dir "docs/guides/features"
ensure_dir "docs/explanations/architecture"
ensure_dir "docs/explanations/ontology"
ensure_dir "docs/explanations/physics"
ensure_dir "docs/reference/api"
ensure_dir "docs/reference/database"
ensure_dir "docs/reference/protocols"
ensure_dir "docs/archive/sprint-logs"
ensure_dir "docs/archive/implementation-logs"
ensure_dir "scripts/tests"

echo ""
echo "Step 2: Phase 1 - Cleanup Artifacts..."
echo "-------------------------------------------"

# Move test scripts
safe_move "docs/implementation/test-p1-1.sh" "scripts/tests/test-p1-1.sh"

# Move implementation logs to archive
for file in docs/implementation/*.txt; do
    [ -f "$file" ] && safe_move "$file" "docs/archive/implementation-logs/$(basename "$file")"
done

# Move sprint logs
for file in docs/implementation/p*.md; do
    [ -f "$file" ] && safe_move "$file" "docs/archive/sprint-logs/$(basename "$file")"
done

# Move stress-majorization (implementation detail)
safe_move "docs/implementation/stress-majorization-implementation.md" "docs/archive/implementation-logs/stress-majorization-implementation.md"

echo ""
echo "Step 3: Phase 2 - Tutorials Migration..."
echo "-------------------------------------------"

# Move tutorials
safe_move "docs/getting-started/01-installation.md" "docs/tutorials/01-installation.md"
safe_move "docs/getting-started/02-first-graph-and-agents.md" "docs/tutorials/02-first-graph.md"

# Check if neo4j-quick-start exists in reference/api
if [ -f "docs/reference/api/neo4j-quick-start.md" ]; then
    safe_move "docs/reference/api/neo4j-quick-start.md" "docs/tutorials/neo4j-quick-start.md"
fi

echo ""
echo "Step 4: Phase 2 - Guides Migration..."
echo "-------------------------------------------"

# Move existing guides (they're already in the right place, just verify structure)
if [ -d "docs/guides/developer" ] && [ "$(ls -A docs/guides/developer 2>/dev/null)" ]; then
    echo "✓ Developer guides already in place"
fi

if [ -d "docs/guides/operations" ] && [ "$(ls -A docs/guides/operations 2>/dev/null)" ]; then
    echo "✓ Operations guides already in place"
fi

# Move features to guides/features with renaming
safe_move "docs/features/client-side-filtering.md" "docs/guides/features/filtering-nodes.md"
safe_move "docs/features/DEEPSEEK_API_VERIFIED.md" "docs/guides/features/deepseek-verification.md"
safe_move "docs/features/DEEPSEEK_SKILL_DEPLOYMENT.md" "docs/guides/features/deepseek-deployment.md"
safe_move "docs/features/PAGINATION_BUG_FIX.md" "docs/guides/features/github-pagination-fix.md"
safe_move "docs/features/LOCAL_FILE_SYNC_STRATEGY.md" "docs/guides/features/local-file-sync-strategy.md"
safe_move "docs/features/intelligent-pathfinding.md" "docs/guides/features/intelligent-pathfinding.md"
safe_move "docs/features/natural-language-queries.md" "docs/guides/features/natural-language-queries.md"
safe_move "docs/features/semantic-forces.md" "docs/guides/features/semantic-forces.md"

# Move multi-agent-docker docs to infrastructure guides
if [ -d "docs/multi-agent-docker" ]; then
    for file in docs/multi-agent-docker/*.md; do
        if [ -f "$file" ]; then
            basename=$(basename "$file")
            safe_move "$file" "docs/guides/infrastructure/$basename"
        fi
    done
fi

echo ""
echo "Step 5: Phase 2 - Explanations Migration..."
echo "-------------------------------------------"

# Merge architecture folders (rename SCREAMING_SNAKE_CASE files)
safe_move "docs/architecture/ONTOLOGY_ARCHITECTURE_ANALYSIS.md" "docs/explanations/architecture/ontology-analysis.md"
safe_move "docs/architecture/ontology-forces.md" "docs/explanations/physics/semantic-forces.md"

# Move other architecture files
for file in docs/architecture/*.md; do
    if [ -f "$file" ]; then
        basename=$(basename "$file")
        safe_move "$file" "docs/explanations/architecture/$basename"
    fi
done

# Preserve ADRs in architecture/decisions
if [ -d "docs/architecture/decisions" ]; then
    ensure_dir "docs/explanations/architecture/decisions"
    for file in docs/architecture/decisions/*.md; do
        if [ -f "$file" ]; then
            # Rename ADR-001 format to 0001-* format
            basename=$(basename "$file")
            if [[ $basename =~ ^ADR-([0-9]+)-(.+)\.md$ ]]; then
                num=$(printf "%04d" ${BASH_REMATCH[1]})
                name="${BASH_REMATCH[2]}"
                newname="${num}-${name}.md"
                safe_move "$file" "docs/explanations/architecture/decisions/$newname"
            else
                safe_move "$file" "docs/explanations/architecture/decisions/$basename"
            fi
        fi
    done
fi

# Move concepts/architecture files
if [ -d "docs/concepts/architecture" ]; then
    for file in docs/concepts/architecture/*.md; do
        if [ -f "$file" ]; then
            # Special handling for 00-architecture-overview
            if [[ $(basename "$file") == "00-architecture-overview.md" ]]; then
                safe_move "$file" "docs/explanations/system-overview.md"
            elif [[ $(basename "$file") == "hexagonal-cqrs-architecture.md" ]]; then
                safe_move "$file" "docs/explanations/architecture/hexagonal-cqrs.md"
            elif [[ $(basename "$file") == "04-database-schemas.md" ]]; then
                safe_move "$file" "docs/reference/database/schemas.md"
            else
                basename=$(basename "$file")
                safe_move "$file" "docs/explanations/architecture/$basename"
            fi
        fi
    done

    # Move subdirectories
    for subdir in components core gpu ports; do
        if [ -d "docs/concepts/architecture/$subdir" ]; then
            ensure_dir "docs/explanations/architecture/$subdir"
            for file in docs/concepts/architecture/$subdir/*.md; do
                [ -f "$file" ] && safe_move "$file" "docs/explanations/architecture/$subdir/$(basename "$file")"
            done
        fi
    done
fi

# Move other concept files to ontology explanations
for file in docs/concepts/*.md; do
    if [ -f "$file" ]; then
        if [[ $(basename "$file") == "ontology-reasoning.md" ]]; then
            safe_move "$file" "docs/explanations/ontology/reasoning-engine.md"
        else
            safe_move "$file" "docs/explanations/ontology/$(basename "$file")"
        fi
    fi
done

echo ""
echo "Step 6: Phase 2 - Reference Migration..."
echo "-------------------------------------------"

# Move API reference files
if [ -d "docs/api" ]; then
    for file in docs/api/*.md; do
        [ -f "$file" ] && safe_move "$file" "docs/reference/api/$(basename "$file")"
    done
fi

# Move existing reference files
safe_move "docs/reference/binary-protocol-specification.md" "docs/reference/protocols/binary-websocket.md"
safe_move "docs/reference/semantic-physics-implementation.md" "docs/reference/physics-implementation.md"

# Keep error-codes.md in reference root
if [ ! -f "docs/reference/error-codes.md" ]; then
    echo "✗ error-codes.md not found in expected location"
fi

echo ""
echo "Step 7: Cleanup Empty Directories..."
echo "-------------------------------------------"

# Remove empty directories
for dir in docs/getting-started docs/concepts docs/features docs/multi-agent-docker docs/implementation docs/architecture; do
    if [ -d "$dir" ]; then
        # Only remove if empty
        if [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
            rmdir "$dir"
            echo "✓ Removed empty directory: $dir"
        else
            echo "⚠ Directory not empty, keeping: $dir"
        fi
    fi
done

echo ""
echo "=== Migration Summary ==="
echo "Directories created: $CREATED"
echo "Files moved: $MOVED"
echo ""
echo "✓ Phase 1 & 2 Complete"
echo ""
echo "Next steps:"
echo "  - Review the new structure"
echo "  - Run Phase 3 script for frontmatter and link fixing"
echo "  - Create the golden index (Phase 4)"
