#!/bin/bash
# update-all-references.sh
# Automated reference update script for filename standardization
# Usage: ./update-all-references.sh {phase1|phase3|phase4|all}

set -e

DOCS_ROOT="/home/devuser/workspace/project/docs"
DRY_RUN=${DRY_RUN:-false}

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Phase 1 reference updates
declare -A phase1_renames=(
  ["guides/developer/development-setup.md"]="guides/developer/01-development-setup.md"
  ["guides/developer/adding-a-feature.md"]="guides/developer/04-adding-features.md"
  ["guides/developer/testing-guide.md"]="guides/developer/05-testing-guide.md"
  ["guides/testing-guide.md"]="guides/developer/05-testing-guide.md"
  ["guides/developer/05-testing.md"]="guides/developer/05-testing-guide.md"
)

# Phase 3 reference updates (SCREAMING_SNAKE_CASE)
declare -A phase3_renames=(
  ["ALIGNMENT_REPORT.md"]="reports/audits/alignment-report-2025-11-04.md"
  ["DEPRECATION_STRATEGY_INDEX.md"]="reports/deprecation-strategy-index.md"
  ["DOCUMENTATION_AUDIT_COMPLETION_REPORT.md"]="reports/audits/documentation-audit-completion-2025-11-04.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_ANALYSIS.md"]="reports/deprecation/graphserviceactor-deprecation-analysis.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_DELIVERY.md"]="reports/deprecation/graphserviceactor-deprecation-delivery.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_RESEARCH.md"]="reports/deprecation/graphserviceactor-deprecation-research.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_SUMMARY.md"]="reports/deprecation/graphserviceactor-deprecation-summary.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_TEMPLATES.md"]="reports/deprecation/graphserviceactor-deprecation-templates.md"
  ["GRAPHSERVICEACTOR_IMPLEMENTATION_PLAN.md"]="reports/deprecation/graphserviceactor-implementation-plan.md"
  ["GRAPHSERVICEACTOR_SEARCH_INDEX.md"]="reports/deprecation/graphserviceactor-search-index.md"
  ["LINK_VALIDATION_REPORT.md"]="reports/audits/link-validation-report-2025-11-04.md"
  ["NEO4J_SETTINGS_MIGRATION_DOCUMENTATION_REPORT.md"]="guides/migration/neo4j-settings-migration.md"
  ["concepts/architecture/00-ARCHITECTURE-OVERVIEW.md"]="concepts/architecture/00-architecture-overview.md"
  ["concepts/architecture/CQRS_DIRECTIVE_TEMPLATE.md"]="concepts/architecture/cqrs-directive-template.md"
  ["concepts/architecture/PIPELINE_INTEGRATION.md"]="concepts/architecture/pipeline-integration.md"
  ["concepts/architecture/PIPELINE_SEQUENCE_DIAGRAMS.md"]="concepts/architecture/pipeline-sequence-diagrams.md"
  ["concepts/architecture/QUICK_REFERENCE.md"]="concepts/architecture/quick-reference.md"
  ["guides/operations/PIPELINE_OPERATOR_RUNBOOK.md"]="guides/operations/pipeline-operator-runbook.md"
  ["implementation/STRESS_MAJORIZATION_IMPLEMENTATION.md"]="implementation/stress-majorization-implementation.md"
  ["multi-agent-docker/ARCHITECTURE.md"]="multi-agent-docker/architecture.md"
  ["multi-agent-docker/DOCKER-ENVIRONMENT.md"]="multi-agent-docker/docker-environment.md"
  ["multi-agent-docker/GOALIE-INTEGRATION.md"]="multi-agent-docker/goalie-integration.md"
  ["multi-agent-docker/PORT-CONFIGURATION.md"]="multi-agent-docker/port-configuration.md"
  ["multi-agent-docker/TOOLS.md"]="multi-agent-docker/tools.md"
  ["multi-agent-docker/TROUBLESHOOTING.md"]="multi-agent-docker/troubleshooting.md"
)

# Phase 4 reference updates (disambiguation)
declare -A phase4_renames=(
  ["concepts/architecture/semantic-physics.md"]="concepts/architecture/semantic-physics-overview.md"
  ["concepts/architecture/semantic-physics-system.md"]="concepts/architecture/semantic-physics-architecture.md"
  ["reference/semantic-physics-implementation.md"]="reference/semantic-physics-api-reference.md"
  ["reference/api/rest-api-reference.md"]="reference/api/02-rest-api.md"
  ["reference/api/rest-api-complete.md"]="reference/api/rest-api-detailed-spec.md"
  ["concepts/architecture/reasoning-tests-summary.md"]="concepts/architecture/reasoning-test-results.md"
  ["concepts/ontology-reasoning.md"]="concepts/ontology-reasoning-concepts.md"
)

update_references() {
  local phase=$1
  declare -n renames=$2
  local updated_count=0

  log_info "=== Updating references for $phase ==="

  for old_path in "${!renames[@]}"; do
    new_path="${renames[$old_path]}"
    log_info "Processing: $old_path → $new_path"

    # Count references before update
    local ref_count=$(grep -r "$old_path" "$DOCS_ROOT" --include="*.md" 2>/dev/null | wc -l)
    log_info "  Found $ref_count references"

    if [ "$DRY_RUN" = true ]; then
      log_warn "  DRY RUN: Would update references"
      continue
    fi

    # Update markdown links [text](path)
    find "$DOCS_ROOT" -type f -name "*.md" -exec \
      sed -i "s|]($old_path)|]($new_path)|g" {} + 2>/dev/null || true

    # Update markdown links with /docs/ prefix
    find "$DOCS_ROOT" -type f -name "*.md" -exec \
      sed -i "s|](/docs/$old_path)|](/docs/$new_path)|g" {} + 2>/dev/null || true

    # Update relative links (../)
    find "$DOCS_ROOT" -type f -name "*.md" -exec \
      sed -i "s|](\.\./$old_path)|](../$new_path)|g" {} + 2>/dev/null || true

    # Update relative links (./)
    find "$DOCS_ROOT" -type f -name "*.md" -exec \
      sed -i "s|](\.\/$old_path)|](./$new_path)|g" {} + 2>/dev/null || true

    # Verify update
    local new_ref_count=$(grep -r "$new_path" "$DOCS_ROOT" --include="*.md" 2>/dev/null | wc -l)
    log_info "  Updated: $new_ref_count references to new path"

    ((updated_count++))
  done

  log_info "$phase complete: $updated_count files processed"
}

# Validation function
validate_links() {
  log_info "=== Validating links after updates ==="

  local broken_count=0
  local total_checked=0

  while IFS= read -r file; do
    # Extract markdown links
    while IFS= read -r link; do
      # Skip external links
      [[ "$link" =~ ^https?:// ]] && continue
      [[ "$link" =~ ^#.* ]] && continue # Skip anchors

      ((total_checked++))

      # Resolve relative path
      local dir=$(dirname "$file")
      local target="$dir/$link"

      # Check if file exists
      if [ ! -f "$target" ] && [ ! -d "$target" ]; then
        log_error "BROKEN: $file -> $link"
        ((broken_count++))
      fi
    done < <(grep -oP '\]\(\K[^)]+' "$file" 2>/dev/null || true)
  done < <(find "$DOCS_ROOT" -name "*.md" -type f)

  log_info "Validation complete: $total_checked links checked"

  if [ $broken_count -eq 0 ]; then
    log_info "✓ All links valid!"
    return 0
  else
    log_error "✗ $broken_count broken links found"
    return 1
  fi
}

# Backup function
create_backup() {
  local backup_dir="/home/devuser/workspace/project/docs-backups"
  mkdir -p "$backup_dir"

  local timestamp=$(date +%Y%m%d-%H%M%S)
  local backup_file="$backup_dir/docs-backup-$timestamp.tar.gz"

  log_info "Creating backup: $backup_file"
  tar -czf "$backup_file" -C /home/devuser/workspace/project docs/

  log_info "Backup created successfully"
  echo "$backup_file"
}

# Main execution
main() {
  local phase="${1:-all}"

  log_info "Starting reference updates (phase: $phase, dry-run: $DRY_RUN)"

  # Create backup unless in dry-run mode
  if [ "$DRY_RUN" = false ]; then
    create_backup
  fi

  case "$phase" in
    phase1)
      update_references "Phase 1 (Duplicates)" phase1_renames
      ;;
    phase3)
      update_references "Phase 3 (Case Normalization)" phase3_renames
      ;;
    phase4)
      update_references "Phase 4 (Disambiguation)" phase4_renames
      ;;
    all)
      update_references "Phase 1 (Duplicates)" phase1_renames
      update_references "Phase 3 (Case Normalization)" phase3_renames
      update_references "Phase 4 (Disambiguation)" phase4_renames
      ;;
    validate)
      validate_links
      exit $?
      ;;
    *)
      log_error "Usage: $0 {phase1|phase3|phase4|all|validate}"
      log_info "Set DRY_RUN=true for testing without making changes"
      exit 1
      ;;
  esac

  log_info "Reference updates complete!"

  # Run validation
  if [ "$DRY_RUN" = false ]; then
    log_info "Running link validation..."
    validate_links
  fi
}

main "$@"
