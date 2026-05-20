#!/bin/bash
# Generate art-upgraded diagram images using Gemini image generation API
# Uses gemini-2.0-flash-exp with image generation response modality

API_KEY="${GOOGLE_GEMINI_API_KEY}"
OUTPUT_DIR="/home/devuser/workspace/VisionFlow/assets/generated"
mkdir -p "$OUTPUT_DIR"

generate_image() {
    local name="$1"
    local prompt="$2"
    local output="$OUTPUT_DIR/${name}.png"

    echo "Generating: $name..."

    response=$(curl -s "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key=${API_KEY}" \
        -H 'Content-Type: application/json' \
        -d "{
            \"contents\": [{
                \"parts\": [{
                    \"text\": \"$prompt\"
                }]
            }],
            \"generationConfig\": {
                \"responseModalities\": [\"TEXT\", \"IMAGE\"]
            }
        }")

    # Extract base64 image data from response
    image_data=$(echo "$response" | python3 -c "
import json, sys, base64
data = json.load(sys.stdin)
try:
    parts = data['candidates'][0]['content']['parts']
    for part in parts:
        if 'inlineData' in part:
            print(part['inlineData']['data'])
            break
    else:
        print('NO_IMAGE', file=sys.stderr)
        # Print text response for debugging
        for part in parts:
            if 'text' in part:
                print(part['text'][:200], file=sys.stderr)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print(json.dumps(data)[:500], file=sys.stderr)
" 2>/tmp/gemini_debug_${name}.txt)

    if [ -n "$image_data" ] && [ "$image_data" != "" ]; then
        echo "$image_data" | base64 -d > "$output"
        echo "  Saved: $output ($(du -h "$output" | cut -f1))"
        return 0
    else
        echo "  FAILED - see /tmp/gemini_debug_${name}.txt"
        cat /tmp/gemini_debug_${name}.txt
        return 1
    fi
}

# 1. Evolution Line — LLM → Coordination Harness
generate_image "evolution-line" \
"Create a professional, dark-themed technology evolution diagram showing the progression from left to right: LLMs → Chatbots → Reasoning Systems → AI Agents → Agentic Systems → External Harnesses → Coordination Harness. Use a sleek horizontal flow with glowing nodes connected by arrows. Each stage should have a small icon. The final stage 'Coordination Harness' should be highlighted with a golden glow and labeled 'VisionFlow'. Use deep navy blue background with cyan and gold accents. Clean, modern, enterprise-quality design. 2048x1024 landscape format."

# 2. Five Substrates Architecture
generate_image "five-substrates" \
"Create a professional hexagonal architecture diagram on dark navy background showing five interconnected systems around a central coordination layer. Center: 'VisionFlow Coordination Layer' in red. Five surrounding hexagons: 'VisionClaw - Knowledge Engineering' (cyan), 'Agentbox - Harness Engineering' (purple), 'solid-pod-rs - Protocol Foundation' (green), 'nostr-rust-forum - Forum Kit' (green), 'DreamLab Edge - Deployment' (amber). Connect all to center with glowing lines. Each hexagon has 3-4 capability bullet points. Modern, clean enterprise architecture diagram style. 2048x2048 square."

# 3. Judgment Broker / Agent Control Surface
generate_image "judgment-broker" \
"Create a professional sequence diagram style illustration showing an AI governance workflow on dark background. Three vertical swim lanes: 'AI Agent' (purple), 'Judgment Broker' (red/gold), 'Human Operator' (cyan). Flow: Agent publishes a Panel Definition → Broker receives and triages → Broker presents to Human → Human approves/rejects → Broker routes response back to Agent. Use Nostr event kind labels (31400, 31402, 31403). Glowing connection lines, enterprise quality. Show cryptographic signatures as small lock icons. 2048x1536."

# 4. Coordination Topology — Real-time mesh
generate_image "coordination-topology" \
"Create a professional network topology diagram on dark background showing three deployment zones connected by message passing. Zone 1: 'Cloudflare Edge' (amber) with Relay Worker, Auth Worker, Pod Worker. Zone 2: 'GPU Host' (cyan) with BrokerActor, ServerNostrActor, ForceComputeActor. Zone 3: 'Agent Container' (purple) with Nostr Bridge, Solid Pod Server, Management API. A green 'Cloudflare Tunnel' connects zones. WebSocket and NIP-42 AUTH labels on connections. Modern enterprise infrastructure diagram. 2048x1536."

# 5. DID:Nostr Identity Spine
generate_image "identity-spine" \
"Create a professional identity architecture diagram on dark background. Central element: a large golden key icon representing 'secp256k1 keypair' with 'did:nostr' label. Five radiating connections to: NIP-42 Relay Auth (shield icon), NIP-98 HTTP Auth (lock icon), WAC Access Control (document icon), Provenance Beads (chain icon), DID Document (fingerprint icon). Emphasise that one cryptographic primitive provides all identity functions. Clean, modern, security-focused design. Cyan and gold on navy. 2048x1536."

echo ""
echo "Generation complete. Files in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/" 2>/dev/null
