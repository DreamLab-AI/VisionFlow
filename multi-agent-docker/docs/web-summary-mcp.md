# Web Summary MCP Server

## Overview
Fetches and summarizes web content using Google Gemini 2.0 Flash, with Claude CLI-based selective topic matching for Logseq note-taking.

## Configuration
- **Location**: `/app/core-assets/scripts/web-summary-mcp-server.py`
- **MCP Config**: `core-assets/mcp.json`
- **Model**: `gemini-2.0-flash-exp`
- **API Key**: Requires `GOOGLE_API_KEY` environment variable

## Features
- Automatic web content fetching via Google Gemini
- UK English summaries with proper spelling
- **Selective topic linking**: Uses Claude CLI to match only strongly relevant topics from `/app/core-assets/topics.json`
- Maximum 5-8 topic links per summary (prevents over-linking)
- Logseq markdown formatting with wiki-style `[[links]]`
- Proper line endings (`\r\n`) for compatibility

## Usage

### Via MCP Tool
```javascript
mcp__web-summary__summarize_url({
  url: "https://example.com",
  prompt_override: "Optional custom instructions"
})
```

### Response Format
```json
{
  "success": true,
  "url": "https://example.com",
  "summary": "- First point with [[concepts]]\r\n- Second point",
  "raw_summary": "Unformatted text"
}
```

## Build Integration

### Dockerfile Steps
1. `COPY gui-tools-assets/web-summary-mcp-server.py /opt/web-summary-mcp-server.py`
2. Sets executable permissions and ownership
3. Python package `google-generativeai` installed via requirements.txt

### Requirements
- Python 3.13+
- google-generativeai >= 0.8.5
- Valid Google API key

## Troubleshooting

### Model Version
- ✅ Use: `gemini-2.0-flash-exp`
- ❌ Avoid: `gemini-2.5-flash` (not yet available)

### Tools Syntax
- ✅ Correct: `tools='google_search_retrieval'`
- ❌ Wrong: `tools=[{'url_context': {}}]` (invalid syntax)

### Testing
```bash
echo '{"jsonrpc":"2.0","id":"test","method":"tools/list","params":{}}' | \
  python3 /app/core-assets/scripts/web-summary-mcp-server.py
```

## Topic Matching System

### How It Works
1. Gemini generates plain text summary without any links
2. Claude CLI analyzes summary against permitted topics list
3. Only **5-8 strongest matches** are selected
4. Topics are linked **once per bullet point** (first occurrence only)

### Topic Configuration
Edit `/app/core-assets/topics.json` to customize permitted topics:
```json
{
  "topics": [
    "artificial intelligence",
    "machine learning",
    "ethics",
    "governance",
    ...
  ]
}
```

### Dependencies
- `google-generativeai` (Gemini API)
- `claude` CLI (for topic matching)
- `GOOGLE_API_KEY` environment variable

## Example Output
**Input URL**: Research paper on AI governance

**Matched Topics**: `["artificial intelligence", "ethics", "governance", "policy", "regulation"]`

**Summary**:
```markdown
- The paper examines [[artificial intelligence]] deployment in organisations.
- Key focus on [[ethics]] frameworks for responsible AI.
- [[Governance]] structures needed to manage risks.
```

Note: Only matched topics get `[[links]]`, everything else is plain text.
