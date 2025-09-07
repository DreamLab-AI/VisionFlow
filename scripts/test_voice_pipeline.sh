#!/bin/bash

echo "=== Testing Voice Pipeline: Kokoro TTS → Whisper STT ==="

# Test 1: Try with container name (if on network)
echo -e "\n1. Testing Kokoro TTS with container name..."
curl -X POST "http://friendly_dewdney:8880/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello world, this is a test of the text to speech and speech to text pipeline.",
    "voice": "af_bella",
    "response_format": "wav",
    "speed": 1.0,
    "stream": false
  }' \
  --output /tmp/kokoro_speech.wav \
  -w "\nHTTP Status: %{http_code}\nDownload Size: %{size_download} bytes\n" \
  2>/dev/null

if [ -f /tmp/kokoro_speech.wav ] && [ $(stat -c%s /tmp/kokoro_speech.wav) -gt 1000 ]; then
    echo "✓ Kokoro TTS successful!"
    ls -lh /tmp/kokoro_speech.wav
    
    echo -e "\n2. Sending audio to Whisper STT..."
    RESPONSE=$(curl -X POST "http://whisper-webui-backend:8000/transcription/" \
      -F "file=@/tmp/kokoro_speech.wav" \
      -F "model_size=base" \
      -F "lang=en" \
      -F "vad_filter=false" \
      -H "Accept: application/json" \
      -w "\nHTTP_STATUS:%{http_code}" \
      2>/dev/null)
    
    # Extract HTTP status and body
    HTTP_STATUS=$(echo "$RESPONSE" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
    BODY=$(echo "$RESPONSE" | sed 's/HTTP_STATUS:[0-9]*//')
    
    echo "Response: $BODY"
    echo "HTTP Status: $HTTP_STATUS"
    
    # Extract task ID if successful
    if [ "$HTTP_STATUS" = "200" ] || [ "$HTTP_STATUS" = "201" ]; then
        TASK_ID=$(echo "$BODY" | grep -o '"identifier":"[^"]*"' | cut -d'"' -f4)
        
        if [ -n "$TASK_ID" ]; then
            echo -e "\nTask ID: $TASK_ID"
            echo "Polling for completion..."
            
            # Poll for task completion
            for i in {1..30}; do
                sleep 0.5
                
                TASK_RESPONSE=$(curl -s "http://whisper-webui-backend:8000/task/$TASK_ID" \
                  -H "Accept: application/json")
                
                STATUS=$(echo "$TASK_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
                
                if [ "$STATUS" = "completed" ]; then
                    echo -e "\n✓ Transcription completed!"
                    
                    # Extract transcribed text - handle both array and single result formats
                    TEXT=$(echo "$TASK_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'result' in data:
        result = data['result']
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'text' in result[0]:
                print(result[0]['text'] or 'No text found')
            else:
                print('No text in result')
        elif isinstance(result, dict) and 'text' in result:
            print(result['text'] or 'No text found')
        else:
            print('Unknown result format')
    else:
        print('No result field')
except:
    print('Failed to parse')
" 2>/dev/null)
                    
                    echo -e "\n===================="
                    echo "Original text: Hello world, this is a test of the text to speech and speech to text pipeline."
                    echo "Transcribed text: $TEXT"
                    echo "===================="
                    
                    if [ "$TEXT" != "No text found" ] && [ "$TEXT" != "Failed to parse" ]; then
                        echo -e "\n✓✓✓ Voice pipeline is working! ✓✓✓"
                    else
                        echo -e "\n⚠ Pipeline completed but transcription was empty"
                        echo "Full response for debugging:"
                        echo "$TASK_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TASK_RESPONSE"
                    fi
                    break
                elif [ "$STATUS" = "failed" ]; then
                    echo -e "\n✗ Transcription failed!"
                    echo "$TASK_RESPONSE"
                    break
                fi
                
                # Show progress
                if [ $((i % 5)) -eq 0 ]; then
                    echo "Still polling... (attempt $i/30, status: $STATUS)"
                fi
            done
        else
            echo "✗ Failed to get task ID from response"
        fi
    else
        echo "✗ Failed to submit audio to Whisper (HTTP $HTTP_STATUS)"
        echo "Error details: $BODY"
    fi
else
    echo "✗ Failed to generate audio with Kokoro"
    echo "File size: $(stat -c%s /tmp/kokoro_speech.wav 2>/dev/null || echo 0) bytes"
    
    # Try to see what's in the file
    if [ -f /tmp/kokoro_speech.wav ]; then
        echo "File content preview:"
        head -c 100 /tmp/kokoro_speech.wav
    fi
fi