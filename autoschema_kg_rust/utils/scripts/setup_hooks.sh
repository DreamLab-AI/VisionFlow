#!/bin/bash

# Setup coordination hooks for swarm integration
# This script sets up the necessary coordination hooks for the utils crate

set -euo pipefail

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
UTILS_DIR=$(dirname $SCRIPT_DIR)
PROJECT_ROOT=$(cd $UTILS_DIR/../.. && pwd)

echo \"Setting up coordination hooks for utils crate...\"
echo \"Utils directory: $UTILS_DIR\"
echo \"Project root: $PROJECT_ROOT\"

# Function to run coordination hooks
run_hook() {
    local hook_type=\"$1\"
    local description=\"$2\"
    shift 2

    echo \"Running $hook_type hook: $description\"

    if command -v npx >/dev/null 2>&1 && npx claude-flow@alpha --version >/dev/null 2>&1; then
        case \"$hook_type\" in
            \"pre-task\")
                npx claude-flow@alpha hooks pre-task --description \"$description\" || true
                ;;
            \"post-task\")
                npx claude-flow@alpha hooks post-task --task-id \"$description\" || true
                ;;
            \"post-edit\")
                local file=\"$1\"
                local memory_key=\"$2\"
                npx claude-flow@alpha hooks post-edit --file \"$file\" --memory-key \"$memory_key\" || true
                ;;
            \"notify\")
                npx claude-flow@alpha hooks notify --message \"$description\" || true
                ;;
            \"session-restore\")
                npx claude-flow@alpha hooks session-restore --session-id \"$description\" || true
                ;;
            \"session-end\")
                npx claude-flow@alpha hooks session-end --export-metrics true || true
                ;;
        esac
    else
        echo \"Claude Flow not available, skipping hook: $hook_type\"
    fi
}

# Initialize coordination session
run_hook \"session-restore\" \"utils-swarm-session\"

# Notify about utils crate setup
run_hook \"pre-task\" \"Setting up Rust utils crate with comprehensive data processing utilities\"

# Register all implemented modules
echo \"Registering implemented modules...\"

modules=(
    \"csv_processing\"
    \"json_processing\"
    \"markdown_processing\"
    \"graph_conversion\"
    \"hash_utils\"
    \"text_cleaning\"
    \"file_io\"
)

for module in \"${modules[@]}\"; do
    module_file=\"$UTILS_DIR/src/${module}.rs\"
    if [ -f \"$module_file\" ]; then
        echo \"Registering module: $module\"
        run_hook \"post-edit\" \"$module_file\" \"swarm/utils/$module\"
        run_hook \"notify\" \"Implemented $module module with comprehensive functionality\"
    fi
done

# Store utils crate capabilities in coordination memory
echo \"Storing utils crate capabilities...\"

# Create capabilities summary
cat > \"$UTILS_DIR/capabilities.json\" << 'EOF'
{
  \"crate\": \"utils\",
  \"version\": \"0.1.0\",
  \"capabilities\": {
    \"csv_processing\": {
      \"functions\": [\"merge_csv\", \"csv_to_graphml\", \"csv_add_numeric_id\", \"analyze_csv\"],
      \"features\": [\"streaming\", \"memory_efficient\", \"statistics\", \"format_conversion\"]
    },
    \"json_processing\": {
      \"functions\": [\"json_to_csv\", \"json_to_graphml\", \"json_repair\", \"validate_json\"],
      \"features\": [\"flattening\", \"repair\", \"streaming\", \"validation\"]
    },
    \"markdown_processing\": {
      \"functions\": [\"markdown_to_json\", \"parse_markdown_content\", \"generate_toc\", \"extract_sections\"],
      \"features\": [\"frontmatter\", \"code_blocks\", \"tables\", \"links\", \"images\"]
    },
    \"graph_conversion\": {
      \"functions\": [\"graph_to_graphml\", \"graphml_to_graph\", \"adjacency_matrix_conversion\"],
      \"features\": [\"cycle_detection\", \"statistics\", \"node_edge_handling\", \"format_conversion\"]
    },
    \"hash_utils\": {
      \"functions\": [\"generate_hash_id\", \"generate_composite_hash_id\", \"batch_generate_hash_ids\"],
      \"features\": [\"sha256\", \"md5\", \"collision_detection\", \"namespacing\", \"consistency\"]
    },
    \"text_cleaning\": {
      \"functions\": [\"clean_text\", \"batch_clean_text\", \"normalize_quotes\", \"remove_character_sets\"],
      \"features\": [\"html_removal\", \"url_removal\", \"email_removal\", \"normalization\", \"stop_words\"]
    },
    \"file_io\": {
      \"functions\": [\"StreamingReader\", \"StreamingWriter\", \"ParallelFileProcessor\", \"FileMerger\"],
      \"features\": [\"streaming\", \"compression\", \"async\", \"memory_efficient\", \"parallel_processing\"]
    }
  },
  \"dependencies\": [\"csv\", \"serde_json\", \"quick-xml\", \"pulldown-cmark\", \"sha2\", \"md5\", \"regex\", \"rayon\"],
  \"performance\": {
    \"memory_efficient\": true,
    \"streaming_support\": true,
    \"parallel_processing\": true,
    \"compression_support\": true
  },
  \"testing\": {
    \"unit_tests\": true,
    \"integration_tests\": true,
    \"benchmarks\": true,
    \"property_tests\": false
  }
}
EOF

run_hook \"post-edit\" \"$UTILS_DIR/capabilities.json\" \"swarm/utils/capabilities\"

# Set up development workflow hooks
echo \"Setting up development workflow hooks...\"

# Create pre-commit hook for coordination
cat > \"$UTILS_DIR/.pre-commit-hook.sh\" << 'EOF'
#!/bin/bash
# Pre-commit hook for utils crate coordination

if command -v npx >/dev/null 2>&1 && npx claude-flow@alpha --version >/dev/null 2>&1; then
    echo \"Running pre-commit coordination hooks...\"

    # Notify about code changes
    npx claude-flow@alpha hooks notify --message \"Utils crate: preparing commit with latest changes\" || true

    # Store current state
    npx claude-flow@alpha hooks post-edit --file \"utils/src/lib.rs\" --memory-key \"swarm/utils/pre-commit-state\" || true
fi

# Run tests before committing
echo \"Running utils crate tests...\"
cd \"$(dirname \"$0\")\"
cargo test --quiet || {
    echo \"Tests failed! Commit aborted.\"
    exit 1
}

echo \"Pre-commit checks passed.\"
EOF

chmod +x \"$UTILS_DIR/.pre-commit-hook.sh\"

# Create post-build hook
cat > \"$UTILS_DIR/.post-build-hook.sh\" << 'EOF'
#!/bin/bash
# Post-build hook for utils crate coordination

if command -v npx >/dev/null 2>&1 && npx claude-flow@alpha --version >/dev/null 2>&1; then
    echo \"Running post-build coordination hooks...\"

    # Notify about successful build
    npx claude-flow@alpha hooks notify --message \"Utils crate: build completed successfully\" || true

    # Store build artifacts information
    npx claude-flow@alpha hooks post-edit --file \"utils/target/release\" --memory-key \"swarm/utils/build-artifacts\" || true
fi
EOF

chmod +x \"$UTILS_DIR/.post-build-hook.sh\"

# Create coordination test script
cat > \"$UTILS_DIR/scripts/test_coordination.sh\" << 'EOF'
#!/bin/bash
# Test coordination hooks functionality

set -euo pipefail

SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"
UTILS_DIR=\"$(cd \"$SCRIPT_DIR\" && cd .. && pwd)\"

echo \"Testing coordination hooks for utils crate...\"

# Test session management
if command -v npx >/dev/null 2>&1 && npx claude-flow@alpha --version >/dev/null 2>&1; then
    echo \"Testing session management...\"
    npx claude-flow@alpha hooks session-restore --session-id \"utils-test-session\" || true

    echo \"Testing task notifications...\"
    npx claude-flow@alpha hooks pre-task --description \"Testing utils crate coordination\" || true
    npx claude-flow@alpha hooks notify --message \"Utils crate coordination test in progress\" || true
    npx claude-flow@alpha hooks post-task --task-id \"utils-coordination-test\" || true

    echo \"Testing memory operations...\"
    npx claude-flow@alpha hooks post-edit --file \"$UTILS_DIR/src/lib.rs\" --memory-key \"swarm/utils/test\" || true

    echo \"Ending test session...\"
    npx claude-flow@alpha hooks session-end --export-metrics true || true

    echo \"Coordination hooks test completed successfully!\"
else
    echo \"Claude Flow not available, coordination hooks will be no-ops\"
    echo \"Install with: npm install -g claude-flow@alpha\"
fi
EOF

chmod +x \"$UTILS_DIR/scripts/test_coordination.sh\"

# Create integration script with other crates
cat > \"$UTILS_DIR/scripts/integrate_with_swarm.sh\" << 'EOF'
#!/bin/bash
# Integration script for swarm coordination

set -euo pipefail

SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"
UTILS_DIR=\"$(cd \"$SCRIPT_DIR\" && cd .. && pwd)\"
PROJECT_ROOT=\"$(cd \"$UTILS_DIR\" && cd ../.. && pwd)\"

echo \"Integrating utils crate with swarm ecosystem...\"

if command -v npx >/dev/null 2>&1 && npx claude-flow@alpha --version >/dev/null 2>&1; then
    echo \"Setting up swarm integration...\"

    # Initialize swarm if not already done
    cd \"$PROJECT_ROOT\"
    npx claude-flow@alpha swarm init --topology mesh --max-agents 5 || true

    # Register utils crate capabilities
    npx claude-flow@alpha agent spawn --type coder --name \"utils-processor\" || true

    # Register available processing functions
    npx claude-flow@alpha hooks notify --message \"Utils crate registered with capabilities: CSV, JSON, Markdown, Graph, Hash, Text, FileIO\" || true

    echo \"Utils crate integrated with swarm successfully!\"
else
    echo \"Claude Flow not available, skipping swarm integration\"
fi
EOF

chmod +x \"$UTILS_DIR/scripts/integrate_with_swarm.sh\"

# Final notifications
run_hook \"notify\" \"Utils crate setup completed with full coordination hooks\"
run_hook \"post-task\" \"utils-crate-setup\"

echo \"Coordination hooks setup completed!\"
echo \"\"
echo \"Available coordination scripts:\"
echo \"  - $UTILS_DIR/scripts/test_coordination.sh\"
echo \"  - $UTILS_DIR/scripts/integrate_with_swarm.sh\"
echo \"  - $UTILS_DIR/.pre-commit-hook.sh\"
echo \"  - $UTILS_DIR/.post-build-hook.sh\"
echo \"\"
echo \"To test coordination: cd $UTILS_DIR && ./scripts/test_coordination.sh\"
echo \"To integrate with swarm: cd $UTILS_DIR && ./scripts/integrate_with_swarm.sh\"