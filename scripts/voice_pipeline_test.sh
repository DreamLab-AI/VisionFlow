#!/bin/bash

# Complete Voice Pipeline Test Script
# Tests: Kokoro TTS → WAV file → Whisper STT

echo "=== Voice Pipeline Test ==="
echo "Kokoro TTS: 172.18.0.9:8880"
echo "Whisper STT: 172.18.0.5:8000"
echo ""

# Test phrases
PHRASES=(
    "Hello world"
    "This is a test of the voice system"
    "The quick brown fox jumps over the lazy dog"
)

for i in "${!PHRASES[@]}"; do
    PHRASE="${PHRASES[$i]}"
    echo "Test $((i+1)): \"$PHRASE\""
    echo "------------------------"
    
    # Step 1: Generate audio with Kokoro
    echo "1. Generating audio with Kokoro TTS..."
    HTTP_STATUS=$(curl -X POST "http://172.18.0.9:8880/v1/audio/speech" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"kokoro\",
            \"input\": \"$PHRASE\",
            \"voice\": \"af_bella\",
            \"response_format\": \"wav\",
            \"speed\": 1.0,
            \"stream\": false
        }" \
        --output "/tmp/voice_test_$i.wav" \
        -w "%{http_code}" \
        -s)
    
    if [ "$HTTP_STATUS" = "200" ]; then
        SIZE=$(stat -c%s "/tmp/voice_test_$i.wav" 2>/dev/null || echo 0)
        echo "   ✓ Audio generated: $SIZE bytes"
        
        # Step 2: Send to Whisper
        echo "2. Sending to Whisper STT..."
        RESPONSE=$(curl -X POST "http://172.18.0.5:8000/transcription/" \
            -F "file=@/tmp/voice_test_$i.wav" \
            -F "model_size=base" \
            -F "lang=en" \
            -s)
        
        TASK_ID=$(echo "$RESPONSE" | grep -o '"identifier":"[^"]*"' | cut -d'"' -f4)
        
        if [ -n "$TASK_ID" ]; then
            echo "   Task ID: $TASK_ID"
            
            # Step 3: Poll for result
            echo "3. Waiting for transcription..."
            for j in {1..20}; do
                sleep 0.5
                TASK_RESPONSE=$(curl -s "http://172.18.0.5:8000/task/$TASK_ID")
                STATUS=$(echo "$TASK_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
                
                if [ "$STATUS" = "completed" ]; then
                    # Extract text from result
                    TEXT=$(echo "$TASK_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'result' in data and data['result']:
        result = data['result']
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get('text', '').strip()
            print(text if text else '[empty]')
        else:
            print('[no result]')
except:
    print('[parse error]')
" 2>/dev/null)
                    
                    echo "   ✓ Transcription complete"
                    echo ""
                    echo "   Original:    \"$PHRASE\""
                    echo "   Transcribed: \"$TEXT\""
                    break
                elif [ "$STATUS" = "failed" ]; then
                    echo "   ✗ Transcription failed"
                    break
                fi
            done
        else
            echo "   ✗ Failed to submit to Whisper"
        fi
    else
        echo "   ✗ Kokoro TTS failed (HTTP $HTTP_STATUS)"
    fi
    
    echo ""
done

echo "=== Test Complete ==="
echo ""
echo "To run this test from your host system:"
echo "  bash /workspace/ext/scripts/voice_pipeline_test.sh"
echo ""
echo "Configuration is stored in:"
echo "  /workspace/ext/data/settings.yaml"