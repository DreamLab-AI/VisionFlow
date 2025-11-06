#!/bin/bash
set -e

echo "🔄 Triggering GitHub Sync from multi-ontology branch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Environment:"
echo "  GITHUB_BRANCH: ${GITHUB_BRANCH}"
echo "  GITHUB_OWNER: ${GITHUB_OWNER}"
echo "  GITHUB_REPO: ${GITHUB_REPO}"
echo "  FORCE_FULL_SYNC: ${FORCE_FULL_SYNC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Try calling the sync endpoint
echo "Calling sync endpoint..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:4000/api/admin/sync \
  -H "Content-Type: application/json" \
  -H "X-Nostr-Pubkey: ${POWER_USER_PUBKEYS}" 2>&1)

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

echo "HTTP Status: $HTTP_CODE"
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Sync triggered successfully!"
    echo "$BODY" | jq . 2>/dev/null || echo "$BODY"
else
    echo "❌ Sync request failed"
    echo "$BODY"
fi
