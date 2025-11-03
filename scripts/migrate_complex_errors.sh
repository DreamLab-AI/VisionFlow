#!/bin/bash

# Migration script for complex multi-field error responses
# These patterns need to stay as HttpResponse until we extend the macros

echo "Analyzing remaining HttpResponse patterns..."
echo "=========================================="

# Find all remaining HttpResponse error patterns
echo ""
echo "Files with remaining patterns:"
grep -r "HttpResponse::InternalServerError\|HttpResponse::BadRequest\|HttpResponse::NotFound" \
    src/handlers/ --include="*.rs" -l | while read file; do
    count=$(grep -c "HttpResponse::InternalServerError\|HttpResponse::BadRequest\|HttpResponse::NotFound" "$file" 2>/dev/null || echo "0")
    if [ "$count" -gt 0 ]; then
        echo "  $file: $count patterns"
    fi
done

echo ""
echo "Pattern analysis:"
echo "=========================================="

# Count different pattern types
total=$(grep -r "HttpResponse::" src/handlers/ --include="*.rs" | grep -v "use actix_web" | grep -v "//" | wc -l)
internal=$(grep -r "HttpResponse::InternalServerError" src/handlers/ --include="*.rs" | grep -v "use " | wc -l)
badreq=$(grep -r "HttpResponse::BadRequest" src/handlers/ --include="*.rs" | grep -v "use " | wc -l)
notfound=$(grep -r "HttpResponse::NotFound" src/handlers/ --include="*.rs" | grep -v "use " | wc -l)
ok_stream=$(grep -r "HttpResponse::Ok()" src/handlers/ --include="*.rs" | grep -v ".json" | wc -l)

echo "Total HttpResponse calls: $total"
echo "InternalServerError: $internal"
echo "BadRequest: $badreq"
echo "NotFound: $notfound"
echo "Ok (streaming/SSE): $ok_stream"
echo ""

# These are complex error patterns that contain both "error" and "message" fields
# We need to either:
# 1. Extend the macros to support this pattern
# 2. Create a new macro for complex errors
# 3. Document these as intentional exceptions

echo "Recommendation: These complex error responses should either:"
echo "1. Be simplified to use error_json! with just the error message"
echo "2. Have a new macro created: complex_error!(error_msg, details)"
echo "3. Be documented as intentional exceptions for rich error responses"
