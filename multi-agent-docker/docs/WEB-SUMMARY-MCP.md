# Web Summary MCP Server

## Overview

The Web Summary MCP Server uses Google AI Studio's Vertex API with the URL Context Tool to retrieve and summarise web content in UK English, formatted specifically for Logseq markdown with wiki-style links and proper line endings.

## Features

- **Google AI Studio URL Context Tool** - Leverages Gemini 2.5 Flash with automatic URL fetching
- **UK English Spelling** - All summaries use British spelling (summarise, colour, organise, etc.)
- **Logseq Formatting** - Dash-space bullet points with `\r\n` line endings
- **Wiki-Style Links** - Automatic `[[concept]]` link formatting for key topics
- **Support for Multiple Content Types** - Text, HTML, PDF, images (up to 34MB per URL)

## Configuration

### Environment Variables

Add to `.env`:
```bash
GOOGLE_API_KEY=your-google-ai-studio-api-key
```

### Port Configuration

- **Default Port**: 9880
- **Host**: 0.0.0.0 (accessible within container)

## Usage

### Via MCP Protocol (stdio)

```bash
# Test via Claude CLI with MCP configuration
claude --mcp-config /app/core-assets/mcp.json
```

### Direct Testing

```bash
# Test the server directly
docker exec -u dev multi-agent-container bash -c '
echo "{\"method\":\"tools/call\",\"params\":{\"name\":\"summarize_url\",\"arguments\":{\"url\":\"https://example.com\"}}}" | \
/opt/venv312/bin/python3 /opt/web-summary-mcp-server.py
'
```

### Example Request

```json
{
  "method": "tools/call",
  "params": {
    "name": "summarize_url",
    "arguments": {
      "url": "https://en.wikipedia.org/wiki/Machine_learning",
      "prompt_override": "Summarise this page focusing on recent developments"
    }
  }
}
```

### Example Response

```json
{
  "content": [
    {
      "type": "text",
      "text": {
        "success": true,
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "summary": "- [[Machine learning]] is a subset of [[artificial intelligence]] that focuses on algorithms learning from data\r\n- Key approaches include [[supervised learning]], [[unsupervised learning]], and [[reinforcement learning]]\r\n- Recent advances in [[deep learning]] have revolutionised computer vision and natural language processing\r\n- Applications span healthcare, finance, autonomous vehicles, and scientific research\r\n",
        "raw_summary": "Machine learning is a subset of artificial intelligence...",
        "url_metadata": "..."
      }
    }
  ]
}
```

## URL Context Tool Capabilities

### Supported Content Types

- **Text**: HTML, JSON, plain text, XML, CSS, JavaScript, CSV, RTF
- **Images**: PNG, JPEG, BMP, WebP
- **Documents**: PDF (up to 34MB)

### Limitations

- Maximum 20 URLs per request
- Maximum 34MB per URL
- Paywalled content not supported
- YouTube videos not supported (use video understanding instead)
- Google Workspace files not supported
- Video/audio files not supported

### URL Metadata

The server returns metadata about URL retrieval:
```json
{
  "url_metadata": {
    "retrieved_url": "https://example.com",
    "url_retrieval_status": "URL_RETRIEVAL_STATUS_SUCCESS"
  }
}
```

**Status Values**:
- `URL_RETRIEVAL_STATUS_SUCCESS` - Content retrieved successfully
- `URL_RETRIEVAL_STATUS_UNSAFE` - URL failed safety check
- Other status codes indicate retrieval failures

## Logseq Formatting

### Output Format

The server formats summaries with:
- Dash-space prefix: `- `
- Wiki-style links: `[[concept]]`
- Both line terminators: `\r\n` (Logseq style)

### Example Output

```markdown
- [[Machine Learning]] enables computers to learn from data without explicit programming\r\n
- Key techniques include [[neural networks]], [[decision trees]], and [[support vector machines]]\r\n
- Applications range from [[image recognition]] to [[natural language processing]]\r\n
```

## Integration with Claude

### MCP Configuration

The server is automatically configured in `/app/core-assets/mcp.json`:

```json
{
  "mcpServers": {
    "web-summary": {
      "command": "/opt/venv312/bin/python3",
      "args": ["-u", "/opt/web-summary-mcp-server.py"],
      "type": "stdio",
      "env": {
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}"
      }
    }
  }
}
```

### Usage in Claude

```
summarize_url(url="https://example.com")
```

## Supervisor Configuration

The server runs as a supervised service:

```ini
[program:web-summary-mcp-server]
command=/opt/venv312/bin/python3 /opt/web-summary-mcp-server.py
directory=/workspace
autorestart=true
priority=20
user=dev
environment=GOOGLE_API_KEY="%(ENV_GOOGLE_API_KEY)s"
```

### Service Management

```bash
# Check status
docker exec multi-agent-container supervisorctl status web-summary-mcp-server

# Restart service
docker exec multi-agent-container supervisorctl restart web-summary-mcp-server

# View logs
docker exec multi-agent-container supervisorctl tail -f web-summary-mcp-server
```

## Troubleshooting

### Missing API Key

**Error**: `GOOGLE_API_KEY not set`

**Solution**: Add your Google AI Studio API key to `.env`:
```bash
GOOGLE_API_KEY=your-api-key-here
```

### URL Retrieval Failed

**Error**: `URL_RETRIEVAL_STATUS_UNSAFE` or retrieval errors

**Causes**:
- Paywalled content
- Content failed safety check
- URL requires authentication
- Content type not supported

**Solution**: Verify URL is publicly accessible and contains supported content types

### Token Limit Exceeded

**Error**: Token count too high

**Solution**: Content from URLs counts as input tokens. Large pages may exceed limits. Consider:
- Providing more specific URLs
- Using custom prompts to focus on specific sections
- Breaking large content across multiple requests

## API Reference

### Tool: `summarize_url`

**Description**: Fetch a web URL and generate a UK English summary formatted for Logseq markdown

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "url": {
      "type": "string",
      "description": "The URL to fetch and summarise"
    },
    "prompt_override": {
      "type": "string",
      "description": "Optional custom prompt for summarisation"
    }
  },
  "required": ["url"]
}
```

**Output**:
```json
{
  "success": boolean,
  "url": string,
  "summary": string,      // Logseq-formatted with \r\n
  "raw_summary": string,  // Unformatted summary
  "url_metadata": string, // URL retrieval metadata (optional)
  "error": string         // Error message (if success=false)
}
```

## Performance Notes

- **Token Usage**: URL content counts as input tokens
- **Rate Limits**: Based on Gemini 2.5 Flash model limits
- **Cost**: See [Google AI pricing](https://ai.google.dev/pricing)
- **Speed**: URL Context Tool uses cached index where possible, falls back to live fetch

## Security

- API key stored in environment variables
- No local storage of fetched content
- All URLs subject to Google's content moderation
- Unsafe content automatically blocked

## References

- [Google AI Studio URL Context Tool](https://ai.google.dev/gemini-api/docs/url-context)
- [Gemini 2.5 Flash Documentation](https://ai.google.dev/gemini-api/docs/models/gemini-v2)
- [Logseq Markdown Format](https://docs.logseq.com/)
