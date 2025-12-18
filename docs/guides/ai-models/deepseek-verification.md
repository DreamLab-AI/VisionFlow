---
title: DeepSeek API Verification - Complete
description: **URL:** `https://api.deepseek.com` **Status:** ✅ Working
category: howto
tags:
  - guide
  - docker
  - rust
updated-date: 2025-12-18
difficulty-level: advanced
---


# DeepSeek API Verification - Complete

## API Status: ✅ FULLY OPERATIONAL

### Tested Endpoints

#### Standard Endpoint (Recommended)
**URL:** `https://api.deepseek.com`
**Status:** ✅ Working

**Available Models:**
1. **deepseek-chat** (Standard)
   - Fast general-purpose chat model
   - ✅ Tested with curl
   - ✅ Tested with Node.js
   - ✅ Tested with custom CLI tool

2. **deepseek-reasoner** (R1)
   - Advanced reasoning model with thinking process
   - Includes `reasoning_content` in responses
   - ✅ Tested successfully

#### Special Endpoint
**URL:** `https://api.deepseek.com/v3.2_speciale_expires_on_20251215`
**Status:** ⚠️ Requires reasoning mode (use deepseek-reasoner model)

### Test Results

#### Test 1: Direct curl (deepseek-chat)
```bash
curl -X POST https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer sk-[your deepseek api key]" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Say hello"}]}'
```
**Result:** ✅ Success
```json
{
  "choices": [{"message": {"content": "Hi"}}],
  "usage": {"total_tokens": 10}
}
```

#### Test 2: Node.js Integration
```javascript
const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer sk-[your deepseek api key]',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'deepseek-chat',
    messages: [{role: 'user', content: 'Write hello world in Python'}]
  })
});
```
**Result:** ✅ Success - Generated Python code

#### Test 3: Custom CLI Tool (deepseek-chat)
```bash
deepseek-chat "Write a fibonacci function in Python"
```
**Result:** ✅ Success
- Tokens used: 109 (17 prompt + 92 completion)
- Generated complete Python function
- Response time: <2 seconds

#### Test 4: Reasoning Model (deepseek-reasoner)
```bash
curl -X POST https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer sk-[your deepseek api key]" \
  -d '{"model":"deepseek-reasoner","messages":[{"role":"user","content":"What is 2+2?"}]}'
```
**Result:** ✅ Success
```json
{
  "choices": [{
    "message": {
      "content": "The result of 2 + 2 is 4.",
      "reasoning_content": "We are asked: \"What is 2+2?\" This is a simple arithmetic question..."
    }
  }],
  "usage": {
    "total_tokens": 76,
    "completion_tokens_details": {"reasoning_tokens": 52}
  }
}
```

### Configuration

#### Current Setup
- **User:** deepseek-user (UID 1004)
- **API Key:** `sk-[your deepseek api key]`
- **Base URL:** `https://api.deepseek.com`
- **Default Model:** `deepseek-chat`

#### Config Location
`/home/deepseek-user/.config/deepseek/config.json`
```json
{
  "apiKey": "sk-[your deepseek api key]",
  "baseUrl": "https://api.deepseek.com",
  "models": {
    "chat": "deepseek-chat",
    "reasoner": "deepseek-reasoner"
  },
  "defaultModel": "deepseek-chat",
  "maxTokens": 4096,
  "temperature": 0.7
}
```

### Custom CLI Tool

**Installation:** `/usr/local/bin/deepseek-chat` (executable by all users)

**Usage:**
```bash
# Single prompt
deepseek-chat "your question here"

# Interactive mode
deepseek-chat --interactive

# Help
deepseek-chat --help
```

**Features:**
- ✅ Direct DeepSeek API integration
- ✅ Token usage reporting
- ✅ Interactive chat mode
- ✅ Environment variable support
- ✅ Error handling

**Example:**
```bash
$ deepseek-chat "Write hello world in Rust"

🤖 DeepSeek thinking...

fn main() {
    println!("Hello, world!");
}

📊 Tokens: 25 (prompt: 8, completion: 17)
⏱️  Model: deepseek-chat
```

### Integration Status

#### ✅ Working
- Direct API calls (curl)
- Node.js integration
- Custom CLI tool
- Both models (chat + reasoner)
- User isolation (deepseek-user)
- Configuration management

#### ⚠️ Limitations
- **agentic-flow:** Does not natively support DeepSeek
  - Requires modification to add DeepSeek provider
  - Alternative: Use custom `deepseek-chat` CLI tool
  - Can use OpenAI-compatible mode with base URL override

### Performance Metrics

**Model:** deepseek-chat
- Average response time: <2 seconds
- Token efficiency: Good (109 tokens for fibonacci function)
- Code generation quality: Excellent
- Cost: Significantly lower than GPT-4

**Model:** deepseek-reasoner
- Includes reasoning process in response
- Higher token usage (includes reasoning tokens)
- Better for complex problem-solving
- Transparent thinking process

### Recommendations

1. **For Simple Tasks:** Use `deepseek-chat`
   - Faster responses
   - Lower token usage
   - Good code generation

2. **For Complex Reasoning:** Use `deepseek-reasoner`
   - Includes thinking process
   - Better for math, logic, analysis
   - More transparent reasoning

3. **CLI Tool:** Use `deepseek-chat` command for quick queries
   - Fast and simple
   - No framework overhead
   - Direct API access

4. **agentic-flow Integration:** Currently not supported
   - Use custom CLI tool instead
   - Or modify agentic-flow to add DeepSeek provider
   - Or use as OpenAI-compatible endpoint

### Environment Variables

```bash
# For deepseek-chat CLI tool
export DEEPSEEK_API_KEY="sk-[your deepseek api key]"
export DEEPSEEK_MODEL="deepseek-chat"  # or "deepseek-reasoner"
export DEEPSEEK_TEMPERATURE="0.7"

# For agentic-flow .env
DEEPSEEK_API_KEY=sk-[your deepseek api key]
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### Verification Commands

```bash
# Test as deepseek-user
docker exec -u deepseek-user agentic-workstation deepseek-chat "Hello"

# Test with curl
docker exec agentic-workstation curl -X POST https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer sk-[your deepseek api key]" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hi"}]}'

# List available models
docker exec agentic-workstation curl https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer sk-[your deepseek api key]"

# Interactive chat
docker exec -u deepseek-user -it agentic-workstation deepseek-chat --interactive
```

### Summary

✅ DeepSeek API fully operational
✅ Both models tested and working
✅ Custom CLI tool installed and functional
✅ User isolation configured
✅ Configuration files in place
✅ Ready for production use

**Note:** While agentic-flow doesn't natively support DeepSeek, the custom `deepseek-chat` CLI provides direct API access with equivalent functionality.

---
**Verified:** December 2, 2025
**Container:** agentic-workstation
**User:** deepseek-user (UID 1004)
