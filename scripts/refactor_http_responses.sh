#!/bin/bash
# HTTP Response Standardization Script
# Task 1.4 - Phase 1: API Specialist Agent
# Replaces direct HttpResponse construction with response macros

set -e

HANDLERS_DIR="/home/devuser/workspace/project/src/handlers"
LOG_FILE="/tmp/http_response_refactor.log"

echo "Starting HTTP Response Standardization..." | tee "$LOG_FILE"
echo "Target directory: $HANDLERS_DIR" | tee -a "$LOG_FILE"

# Counter for tracking changes
TOTAL_REPLACEMENTS=0

# Function to add imports to a file if not present
add_imports() {
    local file="$1"

    # Check if imports already exist
    if ! grep -q "use crate::ok_json;" "$file" 2>/dev/null; then
        # Find the last 'use' statement
        local last_use_line=$(grep -n "^use " "$file" | tail -1 | cut -d: -f1)

        if [ -n "$last_use_line" ]; then
            # Add imports after the last use statement
            sed -i "${last_use_line}a\\
\\
// Response macros - Task 1.4 HTTP Standardization\\
use crate::{ok_json, error_json, bad_request, not_found, created_json};\\
use crate::utils::handler_commons::HandlerResponse;" "$file"
            echo "  Added imports to: $file" | tee -a "$LOG_FILE"
        fi
    fi
}

# Pattern 1: HttpResponse::Ok().json(...) → ok_json!(...)
refactor_ok_responses() {
    local file="$1"
    local count=0

    # Simple case: HttpResponse::Ok().json(data)
    if grep -q "HttpResponse::Ok()\.json(" "$file" 2>/dev/null; then
        # Use perl for complex regex replacement
        perl -i -pe 's/HttpResponse::Ok\(\)\.json\((.*?)\)/ok_json!($1)/g' "$file"
        count=$(grep -c "ok_json!" "$file" || echo 0)
        echo "  [OK] Replaced $count Ok responses in: $(basename $file)" | tee -a "$LOG_FILE"
    fi

    echo $count
}

# Pattern 2: HttpResponse::InternalServerError().json(...) → error_json!(...)
refactor_error_responses() {
    local file="$1"
    local count=0

    if grep -q "HttpResponse::InternalServerError()\.json(" "$file" 2>/dev/null; then
        # For error responses, we need to extract the message
        # This handles: HttpResponse::InternalServerError().json(serde_json::json!({"error": "msg"}))

        # First, handle simple error messages
        perl -i -pe 's/HttpResponse::InternalServerError\(\)\.json\(serde_json::json!\(\{[^}]*"error":\s*"([^"]+)"[^}]*\}\)\)/error_json!("$1")/g' "$file"

        # Handle with message field
        perl -i -pe 's/HttpResponse::InternalServerError\(\)\.json\(serde_json::json!\(\{[^}]*"message":\s*([^,}]+)[^}]*\}\)\)/error_json!($1)/g' "$file"

        count=$(grep -c "error_json!" "$file" || echo 0)
        echo "  [ERROR] Replaced $count error responses in: $(basename $file)" | tee -a "$LOG_FILE"
    fi

    echo $count
}

# Pattern 3: HttpResponse::BadRequest().json(...) → bad_request!(...)
refactor_bad_request_responses() {
    local file="$1"
    local count=0

    if grep -q "HttpResponse::BadRequest()\.json(" "$file" 2>/dev/null; then
        # Handle bad request error messages
        perl -i -pe 's/HttpResponse::BadRequest\(\)\.json\(serde_json::json!\(\{[^}]*"error":\s*"([^"]+)"[^}]*\}\)\)/bad_request!("$1")/g' "$file"
        perl -i -pe 's/HttpResponse::BadRequest\(\)\.json\(serde_json::json!\(\{[^}]*"message":\s*([^,}]+)[^}]*\}\)\)/bad_request!($1)/g' "$file"

        count=$(grep -c "bad_request!" "$file" || echo 0)
        echo "  [BAD_REQUEST] Replaced $count bad request responses in: $(basename $file)" | tee -a "$LOG_FILE"
    fi

    echo $count
}

# Pattern 4: HttpResponse::NotFound().json(...) → not_found!(...)
refactor_not_found_responses() {
    local file="$1"
    local count=0

    if grep -q "HttpResponse::NotFound()\.json(" "$file" 2>/dev/null; then
        perl -i -pe 's/HttpResponse::NotFound\(\)\.json\(serde_json::json!\(\{[^}]*"error":\s*"([^"]+)"[^}]*\}\)\)/not_found!("$1")/g' "$file"

        count=$(grep -c "not_found!" "$file" || echo 0)
        echo "  [NOT_FOUND] Replaced $count not found responses in: $(basename $file)" | tee -a "$LOG_FILE"
    fi

    echo $count
}

# Pattern 5: HttpResponse::Created().json(...) → created_json!(...)
refactor_created_responses() {
    local file="$1"
    local count=0

    if grep -q "HttpResponse::Created()\.json(" "$file" 2>/dev/null; then
        perl -i -pe 's/HttpResponse::Created\(\)\.json\((.*?)\)/created_json!($1)/g' "$file"

        count=$(grep -c "created_json!" "$file" || echo 0)
        echo "  [CREATED] Replaced $count created responses in: $(basename $file)" | tee -a "$LOG_FILE"
    fi

    echo $count
}

# Process all Rust files in handlers directory
echo -e "\n=== Processing handler files ===" | tee -a "$LOG_FILE"

find "$HANDLERS_DIR" -type f -name "*.rs" | while read -r file; do
    echo -e "\nProcessing: $file" | tee -a "$LOG_FILE"

    # Skip if file doesn't contain HttpResponse
    if ! grep -q "HttpResponse::" "$file" 2>/dev/null; then
        echo "  Skipping (no HttpResponse found)" | tee -a "$LOG_FILE"
        continue
    fi

    # Add necessary imports
    add_imports "$file"

    # Apply refactoring patterns
    file_total=0
    file_total=$((file_total + $(refactor_ok_responses "$file")))
    file_total=$((file_total + $(refactor_error_responses "$file")))
    file_total=$((file_total + $(refactor_bad_request_responses "$file")))
    file_total=$((file_total + $(refactor_not_found_responses "$file")))
    file_total=$((file_total + $(refactor_created_responses "$file")))

    TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + file_total))

    if [ $file_total -gt 0 ]; then
        echo "  ✓ Total replacements in file: $file_total" | tee -a "$LOG_FILE"
    fi
done

echo -e "\n=== Refactoring Summary ===" | tee -a "$LOG_FILE"
echo "Total HTTP responses standardized: $TOTAL_REPLACEMENTS" | tee -a "$LOG_FILE"

# Verify remaining direct HttpResponse usage
echo -e "\n=== Verification ===" | tee -a "$LOG_FILE"
REMAINING=$(grep -r "HttpResponse::" "$HANDLERS_DIR" --include="*.rs" | grep -v "use actix" | grep -v "response_macros" | grep -v "handler_commons" | wc -l)
echo "Remaining direct HttpResponse usages: $REMAINING" | tee -a "$LOG_FILE"

if [ $REMAINING -eq 0 ]; then
    echo "✓ SUCCESS: All HTTP responses standardized!" | tee -a "$LOG_FILE"
else
    echo "⚠ WARNING: $REMAINING direct HttpResponse usages still remain" | tee -a "$LOG_FILE"
    echo "Manual review may be required for complex cases" | tee -a "$LOG_FILE"
fi

echo -e "\nLog file: $LOG_FILE"
