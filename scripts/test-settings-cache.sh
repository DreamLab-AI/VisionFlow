#!/bin/bash

API_URL="http://172.18.0.10:4000/api/settings"
TEST_PUBKEY="test-user-$(date +%s)"
TEST_TOKEN="token-$(date +%s)"
TEST_VALUE=$(echo "scale=4; $RANDOM / 10000" | bc)

echo "Testing Settings Cache Issue"
echo "============================"
echo "Test User: $TEST_PUBKEY"
echo "Test Value: $TEST_VALUE"
echo ""

echo "1. Getting current value WITHOUT auth:"
CURRENT=$(curl -s $API_URL | jq '.visualisation.graphs.logseq.nodes.nodeSize')
echo "   Current nodeSize: $CURRENT"
echo ""

echo "2. Updating value WITH auth headers:"
curl -s -X PUT $API_URL \
  -H "Content-Type: application/json" \
  -H "X-Nostr-Pubkey: $TEST_PUBKEY" \
  -H "X-Nostr-Token: $TEST_TOKEN" \
  -d "{\"path\": \"visualisation.graphs.logseq.nodes.nodeSize\", \"value\": $TEST_VALUE}" \
  > /dev/null
echo "   Updated to: $TEST_VALUE"
echo ""

echo "3. Getting value immediately WITH same auth (should be $TEST_VALUE):"
AUTHED=$(curl -s \
  -H "X-Nostr-Pubkey: $TEST_PUBKEY" \
  -H "X-Nostr-Token: $TEST_TOKEN" \
  $API_URL | jq '.visualisation.graphs.logseq.nodes.nodeSize')
echo "   Got: $AUTHED"
echo ""

echo "4. Getting value WITHOUT auth (should be default):"
UNAUTHED=$(curl -s $API_URL | jq '.visualisation.graphs.logseq.nodes.nodeSize')
echo "   Got: $UNAUTHED"
echo ""

echo "5. Getting value with DIFFERENT auth:"
DIFFERENT=$(curl -s \
  -H "X-Nostr-Pubkey: different-user" \
  -H "X-Nostr-Token: different-token" \
  $API_URL | jq '.visualisation.graphs.logseq.nodes.nodeSize')
echo "   Got: $DIFFERENT"
echo ""

echo "Results:"
echo "--------"
if [ "$AUTHED" == "$TEST_VALUE" ]; then
  echo "❌ CACHE BUG: Authenticated user got STALE value ($AUTHED) instead of their value ($TEST_VALUE)"
else
  echo "✅ Authenticated user got their personalized value"
fi

if [ "$UNAUTHED" == "$AUTHED" ]; then
  echo "❌ AUTH BUG: Unauthenticated request got same value as authenticated"
else
  echo "✅ Different values for authenticated vs unauthenticated"
fi