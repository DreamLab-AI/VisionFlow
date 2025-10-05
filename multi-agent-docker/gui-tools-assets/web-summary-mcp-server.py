#!/usr/bin/env python3
"""
Web Summary MCP Server - Google AI Studio with URL Context Tool
Retrieves web content using URL Context Tool and generates UK English summaries formatted for Logseq
"""

import json
import sys
import os
import asyncio
from typing import Any, Dict, List
import google.generativeai as genai
import re

# Configure Vertex AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

class WebSummaryMCPServer:
    """MCP Server for web URL summarization with Logseq formatting"""

    def __init__(self):
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            tools=[{'url_context': {}}]  # Enable URL Context Tool
        )

    def format_logseq_markdown(self, text: str) -> str:
        """Convert summary to Logseq markdown format with proper line endings"""
        lines = []

        # Split by sentences or paragraphs
        blocks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Convert **bold** to wiki-style links where appropriate
            # Convert URLs to inline format
            block = re.sub(r'\*\*([^*]+)\*\*', r'[[\1]]', block)

            # Add dash-space prefix and proper line endings (both \r and \n)
            lines.append(f"- {block}\r\n")

        return ''.join(lines)

    async def summarize_url(self, url: str, prompt_override: str = None) -> Dict[str, Any]:
        """Fetch URL using URL Context Tool and generate UK English summary formatted for Logseq"""
        try:
            # Build prompt - URL Context Tool will automatically fetch content
            base_prompt = f"""Summarise the content from {url} in UK English spelling.
Format the summary as bullet points suitable for Logseq markdown:
- Use proper UK English spelling (summarise, colour, organise, etc.)
- Create concise, informative bullet points
- Use wiki-style [[links]] for key concepts and topics
- Each point should be a complete thought
- Focus on main ideas, key facts, and actionable insights

Summary:"""

            prompt = prompt_override if prompt_override else base_prompt

            # Generate summary - URL Context Tool automatically handles fetching
            response = self.model.generate_content(prompt)
            summary_text = response.text

            # Extract URL metadata if available
            url_metadata = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'url_context_metadata'):
                    url_metadata = candidate.url_context_metadata

            # Format for Logseq
            formatted = self.format_logseq_markdown(summary_text)

            result = {
                "success": True,
                "url": url,
                "summary": formatted,
                "raw_summary": summary_text
            }

            if url_metadata:
                result["url_metadata"] = str(url_metadata)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "summarize_url",
                        "description": "Fetch a web URL and generate a UK English summary formatted for Logseq markdown with wiki-style links and proper line endings",
                        "inputSchema": {
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
                    }
                ]
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})

            if tool_name == "summarize_url":
                result = await self.summarize_url(
                    url=args.get("url"),
                    prompt_override=args.get("prompt_override")
                )
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

        return {"error": "Unknown method"}

    async def run(self):
        """Run the MCP server loop"""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line)
                response = await self.handle_request(request)

                print(json.dumps(response), flush=True)

            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

async def main():
    server = WebSummaryMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
