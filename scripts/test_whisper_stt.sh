#!/bin/bash

# Test Whisper STT API with Kokoro-generated audio
echo "Testing Whisper STT at whisper-webui-backend:8000"

# First, generate audio with Kokoro if needed
if [ ! -f /tmp/kokoro_test.wav ]; then
    echo "Generating test audio with Kokoro first..."
    bash /workspace/ext/scripts/test_kokoro_tts.sh
fi

if [ -f /tmp/kokoro_test.wav ]; then
    echo -e "\nSubmitting audio to Whisper for transcription..."
    
    # Submit audio for transcription
    RESPONSE=$(curl -X POST "http://whisper-webui-backend:8000/transcription/" \
      -F "file=@/tmp/kokoro_test.wav" \
      -F "model_size=base" \
      -F "lang=en" \
      -F "vad_filter=true" \
      -H "Accept: application/json" \
      -w "\nHTTP_STATUS:%{http_code}" \
      2>/dev/null)
    
    # Extract HTTP status
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
                echo "Attempt $i: Status = $STATUS"
                
                if [ "$STATUS" = "completed" ]; then
                    echo -e "\n✓ Transcription completed!"
                    echo "Full response:"
                    echo "$TASK_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TASK_RESPONSE"
                    
                    # Extract transcribed text
                    TEXT=$(echo "$TASK_RESPONSE" | grep -o '"text":"[^"]*"' | head -1 | cut -d'"' -f4)
                    echo -e "\nTranscribed text: $TEXT"
                    break
                elif [ "$STATUS" = "failed" ]; then
                    echo -e "\n✗ Transcription failed!"
                    echo "$TASK_RESPONSE"
                    break
                fi
            done
        else
            echo "✗ Failed to get task ID from response"
        fi
    else
        echo "✗ Failed to submit audio (HTTP $HTTP_STATUS)"
        echo "Error details: $BODY"
    fi
else
    echo "✗ No audio file to test with"
fi