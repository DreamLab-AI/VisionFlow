# Web Summary MCP - YouTube Transcript Fix

## Problem
Google Gemini was returning **hallucinated** summaries for YouTube URLs instead of actual video content. Testing revealed:
- Video `WPlUSnLTmfI` ("AI Animation Tutorial") → Gemini returned generic "time management" summary
- Video `X-zB4-gX3eA` ("AnimateDiff Tutorial") → Gemini returned generic "digital photos" summary

## Root Cause
Google Gemini **cannot reliably access YouTube video content** directly via URL. It hallucinated plausible-sounding summaries based on URL patterns.

## Solution
Implemented **YouTube Transcript API** integration:
1. Detects YouTube URLs (`youtube.com` or `youtu.be`)
2. Fetches actual video transcripts via `youtube-transcript-api` library
3. Passes transcript text to Gemini for summarization
4. Falls back to URL-based summarization if transcript unavailable

## Dependencies Added
- **Package**: `youtube-transcript-api==1.2.2`
- **Installation**: Must be installed in `/opt/venv312` for MCP server

## Installation Command
```bash
sudo /opt/venv312/bin/pip install youtube-transcript-api
```

## Code Changes
### New Imports
```python
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
```

### New Methods
- `extract_youtube_id(url)` - Parse video ID from YouTube URL
- `get_youtube_transcript(url)` - Fetch transcript using API

### Modified Method
- `summarize_url()` - Now detects YouTube and fetches transcript before summarizing

## Test Results
**Before Fix:**
- `WPlUSnLTmfI` → "Discusses strategies for effective time management..." ❌ WRONG

**After Fix:**
- `WPlUSnLTmfI` → "The video is the first in a series about mastering animation in Stable Diffusion..." ✅ CORRECT

## Build Integration
Updated files:
- `/opt/web-summary-mcp-server.py` (runtime)
- `/workspace/ext/multi-agent-docker/gui-tools-assets/web-summary-mcp-server.py` (build)
- `/workspace/ext/multi-agent-docker/core-assets/scripts/web-summary-mcp-server.py` (build)

## Next Steps
Add `youtube-transcript-api` to Dockerfile pip install list to persist across container rebuilds.
