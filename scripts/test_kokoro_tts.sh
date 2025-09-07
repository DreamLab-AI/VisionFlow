#!/bin/bash

# Test Kokoro TTS API
echo "Testing Kokoro TTS at pedantic_morse:8880"

# Create test TTS request
echo -e "\nGenerating test audio with Kokoro..."
curl -X POST "http://pedantic_morse:8880/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello, this is a test of the text to speech system.",
    "voice": "af_heart",
    "response_format": "wav",
    "speed": 1.0,
    "stream": false
  }' \
  --output /tmp/kokoro_test.wav \
  -w "\nHTTP Status: %{http_code}\nDownload Size: %{size_download} bytes\n" \
  2>/dev/null

echo -e "\nChecking generated file..."
if [ -f /tmp/kokoro_test.wav ]; then
    echo "✓ Audio file created successfully"
    ls -lh /tmp/kokoro_test.wav
    file /tmp/kokoro_test.wav
else
    echo "✗ Failed to create audio file"
fi