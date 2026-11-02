#!/usr/bin/env bash
# Cross-topic consistency: for each confirmed finding, list every topic that
# mentions the mechanism, and whether it also names the confirmed fact. A topic
# that mentions the mechanism but not the fact is a candidate stale claim.
# Usage: cross-topic.sh
W=/home/devuser/workspace/wt/VisionFlow-audit-wave3/docs/diagrams
check() {  # label, mechanism-regex, fact-regex
  echo "== $1"
  grep -l -E -i "$2" -r "$W" --include='*.md' | grep -v -E '/(README|COVERAGE|REGISTER|VERIFICATION|DECISIONS-TIMELINE)\.md$' | while read -r f; do
    if grep -q -E -i "$3" "$f"; then tag=names-fact; else tag=MISSING-FACT; fi
    echo "   $tag  ${f#$W/}"
  done
}
check "W2-01 agent-event envelope"   'agent-event-ws-subscriber|createMcpNotification|notifications/agent_action' 'Malformed|bare event|W2-01|unwrapped|DRIFT'
check "ADR-2142 roster routing"      'bots/data|GetBotsGraphData|bots_graph_data'                                 'AgentBeamActor|GraphServiceSupervisor'
check "relay consumer empty allowlist" 'AGENTBOX_RELAY_ALLOWED_PUBKEYS|relay-consumer|RelayConsumer'             'empty|every (signed )?author|admit|Tension'
check "LWS CID iat window"            'lws_cid|LwsCidVerifier'                                                   'iat|&&|never (fires|rejects)'
check "dream-engine/headroom licence" 'dream-engine|headroom-napi'                                               'AGPL'
check "API key read/write split"      'api-keys|ApiKeys|ProtectedSettingsActor'                                  'NostrService|never written|empty|split|two (different )?stores'
