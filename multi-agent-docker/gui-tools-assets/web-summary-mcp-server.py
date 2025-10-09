#!/usr/bin/env python3
"""
Web Summary MCP Server - Google AI Studio with Topic Matching
Retrieves web content and generates UK English summaries formatted for Logseq
Uses Z.AI GLM-4.6 to match content to predefined topics
"""

import json
import sys
import os
import asyncio
import subprocess
from typing import Any, Dict, List, Set
import google.generativeai as genai
import re
import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Configure Z.AI
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
ZAI_API_URL = "https://api.z.ai/v1/chat/completions"

# Load permitted topics
TOPICS_FILE = "/app/core-assets/topics.json"
PERMITTED_TOPICS = []
try:
    with open(TOPICS_FILE, 'r') as f:
        topics_data = json.load(f)
        PERMITTED_TOPICS = topics_data.get("topics", [])
except Exception as e:
    print(f"Warning: Could not load topics from {TOPICS_FILE}: {e}", file=sys.stderr)

class WebSummaryMCPServer:
    """MCP Server for web URL summarization with Logseq formatting"""

    def __init__(self):
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp'
            # Note: Google Search grounding requires Vertex AI setup
        )

    async def add_topic_links_with_zai(self, summary_text: str, topics: List[str]) -> tuple[str, List[str]]:
        """Use Z.AI GLM-4.6 to add [[topic links]] to summary based on semantic matching"""
        if not topics or not ZAI_API_KEY:
            return summary_text, []

        try:
            prompt = f"""You are formatting a summary for Logseq. Add wiki-style [[topic links]] to the text.

CRITICAL RULES - FOLLOW EXACTLY:
1. ONLY link topics that appear EXACTLY in the provided list below
2. DO NOT create new topics or variations - you must use the EXACT text from the list
3. Use semantic matching to find concepts in the summary that match list topics
4. Maximum 5-8 total links across the entire summary
5. Link each topic at most ONCE (first strong occurrence only)
6. If a concept has NO match in the list, DO NOT link it
7. Return the formatted summary with [[links]] inserted
8. Also return a JSON list of which topics you linked (must be exact matches from the list)

PERMITTED TOPICS (use EXACT text from this list):
{json.dumps(topics)}

Summary to format:
{summary_text}

EXAMPLES OF CORRECT LINKING:
- "AI systems" → [[artificial intelligence]] (if in list)
- "ML models" → [[machine learning]] (if in list)
- "workflow automation" → [[automation]] (if in list)
- "animation" → NO LINK (if "animation" not in list)
- "models" → NO LINK (if "models" not in list)

Return your response in this EXACT format:
FORMATTED_SUMMARY:
[the summary with [[links]] added - ONLY using topics from the permitted list]

MATCHED_TOPICS:
["exact topic from list 1", "exact topic from list 2"]

Response:"""

            # Call Z.AI API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    ZAI_API_URL,
                    headers={
                        "Authorization": f"Bearer {ZAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "glm-4.6",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful AI assistant that formats text with wiki-style topic links."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data['choices'][0]['message']['content'].strip()

                    # Extract formatted summary
                    summary_match = re.search(r'FORMATTED_SUMMARY:\s*(.+?)(?=MATCHED_TOPICS:|$)', response_text, re.DOTALL)
                    # Extract matched topics JSON
                    topics_match = re.search(r'MATCHED_TOPICS:\s*(\[.*?\])', response_text, re.DOTALL)

                    if summary_match:
                        formatted_summary = summary_match.group(1).strip()
                        matched_topics = []

                        if topics_match:
                            try:
                                matched_topics = json.loads(topics_match.group(1))
                            except:
                                pass

                        return formatted_summary, matched_topics
                    else:
                        print(f"Warning: Could not parse Z.AI response", file=sys.stderr)
                        return summary_text, []
                else:
                    print(f"Warning: Z.AI API failed with status {response.status_code}: {response.text}", file=sys.stderr)
                    return summary_text, []
        except Exception as e:
            print(f"Warning: Topic linking failed: {e}", file=sys.stderr)
            return summary_text, []

    def format_logseq_markdown(self, text: str) -> str:
        """Convert summary to Logseq markdown format with proper line endings"""
        lines = []

        # Split by sentences or paragraphs
        blocks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Add dash-space prefix and proper line endings
            lines.append(f"- {block}\r\n")

        return ''.join(lines)

    def remove_unauthorized_topics(self, content: str, permitted_topics: List[str]) -> str:
        """Remove all [[topic]] links not in the permitted list"""
        # Extract all [[topic]] links from content
        all_links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        unique_links = set(all_links)

        # Find unauthorized topics (case-insensitive comparison)
        permitted_lower = [t.lower() for t in permitted_topics]
        unauthorized = [link for link in unique_links if link.lower() not in permitted_lower]

        if not unauthorized:
            return content

        # Use Python regex to remove unauthorized topics
        cleaned_content = content
        for topic in unauthorized:
            # Escape special regex characters
            escaped_topic = re.escape(topic)

            # Remove [[topic]] or [[topic|alias]] - replace with just the topic text (no brackets)
            pattern = rf'\[\[{escaped_topic}(?:\|[^\]]+)?\]\]'
            cleaned_content = re.sub(pattern, topic, cleaned_content)

        print(f"Removed {len(unauthorized)} unauthorized topics: {', '.join(sorted(unauthorized))}", file=sys.stderr)
        return cleaned_content

    async def expand_markdown_links(self, file_path: str) -> Dict[str, Any]:
        """Expand both bare [[links]] and URL links with descriptions"""
        try:
            # Read the markdown file
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "file_path": file_path
                }

            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Extract bare [[links]] (lines that only contain [[Link Name]])
            import re
            bare_link_pattern = r'^\s*-\s*\[\[([^\]]+)\]\]\s*$'
            bare_links = re.findall(bare_link_pattern, original_content, re.MULTILINE)

            # Extract URL links [text](url)
            url_link_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
            url_links = re.findall(url_link_pattern, original_content)

            if not bare_links and not url_links:
                return {
                    "success": True,
                    "file_path": file_path,
                    "message": "No links found to expand",
                    "links_expanded": 0
                }

            # Fetch summaries for each URL
            url_summaries = {}
            url_age_emojis = {}
            for link_text, url in url_links:
                print(f"Fetching summary for: {url}", file=sys.stderr)
                summary_result = await self.summarize_url(url)
                if summary_result.get("success"):
                    # Get first bullet point from summary (most concise)
                    summary_lines = summary_result.get("summary", "").split("\r\n")
                    first_line = summary_lines[0].strip() if summary_lines else ""
                    # Remove leading "- " if present
                    first_line = first_line.lstrip("- ").strip()
                    url_summaries[url] = first_line

                    # Store age emoji if available
                    if summary_result.get("age_emoji"):
                        url_age_emojis[url] = summary_result.get("age_emoji")
                else:
                    url_summaries[url] = ""

            # Use Claude to expand the links
            # Prepare age emoji information
            age_emoji_info = ""
            if url_age_emojis:
                age_emoji_info = f"""

Age emojis for URLs (add these at the START of the summary, before the text):
{json.dumps(url_age_emojis, indent=2)}

Traffic light system:
🟢 = 1-7 days (fresh)
🟡 = 1-4 weeks
🟠 = 1-6 months
🟢 = 6-12 months
⚫ = 1+ years (archived)
"""

            prompt = f"""You are updating a Logseq markdown file by adding summaries to URL links and descriptions to bare [[links]].

Original file content:
{original_content}

Bare [[links]] to expand (write concise 1-2 sentence descriptions for these):
{json.dumps(bare_links)}

URL summaries to INSERT (these are COMPLETE - just insert them EXACTLY as provided):
{json.dumps(url_summaries, indent=2)}{age_emoji_info}

Available topics for bare link descriptions ONLY:
{json.dumps(PERMITTED_TOPICS)}

CRITICAL RULES:
1. **For URL links** `[text](url)`: INSERT the provided summary EXACTLY as given, no modifications
   - Add " - " before the summary
   - If an age emoji is provided, add it at the START: " - 🟠 Summary text..."
   - If no age emoji, just: " - Summary text..."
   - DO NOT rewrite, expand, or modify the provided summaries
   - DO NOT add additional [[topic links]] to URL summaries (they already have them)

2. **For bare [[links]]** ONLY: Write a concise 1-2 sentence description
   - You MAY add [[topic links]] from the approved list for bare links only (max 2-3 links)
   - Use UK English spelling
   - Integrate the [[link]] naturally into the description

3. Keep ALL existing content structure EXACTLY as is (indentation, formatting, line breaks)

EXAMPLE FOR URL LINKS (EXACT INSERTION):
Input summary for https://example.com: "Explains [[machine learning]] basics"
Before:
```
	- [Tutorial](https://example.com)
```
After:
```
	- [Tutorial](https://example.com) - Explains [[machine learning]] basics
```

EXAMPLE WITH AGE EMOJI (EXACT INSERTION):
Input summary for https://example.com: "Guide to [[neural networks]]"
Age emoji: 🟠
Before:
```
	- [Tutorial](https://example.com)
```
After:
```
	- [Tutorial](https://example.com) - 🟠 Guide to [[neural networks]]
```

EXAMPLE FOR BARE LINKS (WRITE NEW DESCRIPTION):
Before:
```
	- [[AI Video]]
```
After:
```
	- [[AI Video]] encompasses techniques for generating video content using [[artificial intelligence]]
```

Return ONLY the complete updated markdown file, no explanations.

Response:"""

            # Call Z.AI API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    ZAI_API_URL,
                    headers={
                        "Authorization": f"Bearer {ZAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "glm-4.6",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful AI assistant that formats Logseq markdown files."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },
                    timeout=60.0
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Z.AI API failed with status {response.status_code}: {response.text}",
                        "file_path": file_path
                    }

                response_data = response.json()
                expanded_content = response_data['choices'][0]['message']['content'].strip()

            # Remove markdown code fences if Z.AI added them
            if expanded_content.startswith('```markdown'):
                expanded_content = expanded_content.replace('```markdown\n', '', 1)
                expanded_content = expanded_content.rsplit('```', 1)[0]
            elif expanded_content.startswith('```'):
                expanded_content = expanded_content.replace('```\n', '', 1)
                expanded_content = expanded_content.rsplit('```', 1)[0]

            expanded_content = expanded_content.strip()

            # CRITICAL: Remove all unauthorized topic links using sed
            # This ensures only topics from topics.json remain
            expanded_content = self.remove_unauthorized_topics(expanded_content, PERMITTED_TOPICS)

            # Create backup
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # Write expanded content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(expanded_content)

            total_expanded = len(bare_links) + len(url_links)
            return {
                "success": True,
                "file_path": file_path,
                "backup_path": backup_path,
                "links_expanded": total_expanded,
                "bare_links_expanded": bare_links,
                "url_links_expanded": [url for _, url in url_links],
                "message": f"Successfully expanded {len(bare_links)} bare links and {len(url_links)} URL links"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def extract_youtube_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        parsed = urlparse(url)
        if parsed.hostname in ['www.youtube.com', 'youtube.com']:
            if parsed.path == '/watch':
                return parse_qs(parsed.query).get('v', [None])[0]
        elif parsed.hostname == 'youtu.be':
            return parsed.path[1:]
        return None

    async def get_youtube_metadata(self, video_id: str) -> Dict[str, str]:
        """Fetch YouTube video metadata using YouTube Data API"""
        if not GOOGLE_API_KEY:
            return {}

        try:
            youtube = build('youtube', 'v3', developerKey=GOOGLE_API_KEY)
            request = youtube.videos().list(
                part='snippet,contentDetails',
                id=video_id
            )
            response = request.execute()

            if not response.get('items'):
                return {}

            item = response['items'][0]
            snippet = item['snippet']

            metadata = {
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'published_at': snippet.get('publishedAt', ''),
                'channel_title': snippet.get('channelTitle', ''),
                'tags': snippet.get('tags', [])
            }

            return metadata
        except HttpError as e:
            print(f"Warning: YouTube API error: {e}", file=sys.stderr)
            return {}
        except Exception as e:
            print(f"Warning: Could not fetch YouTube metadata: {e}", file=sys.stderr)
            return {}

    async def get_youtube_transcript(self, url: str) -> tuple[str, Dict[str, str]]:
        """Fetch YouTube transcript if available, fallback to metadata"""
        video_id = self.extract_youtube_id(url)
        if not video_id:
            return None, {}

        # Try to get transcript first
        try:
            transcript_api = YouTubeTranscriptApi()
            result = transcript_api.fetch(video_id)
            full_text = ' '.join([snippet.text for snippet in result.snippets])
            return full_text, {}
        except Exception as e:
            print(f"Warning: Could not fetch YouTube transcript: {e}", file=sys.stderr)
            print(f"Falling back to YouTube Data API for metadata...", file=sys.stderr)

            # Fallback to YouTube Data API
            metadata = await self.get_youtube_metadata(video_id)
            return None, metadata

    def get_age_emoji(self, date_str: str) -> str:
        """Convert date to age emoji based on traffic light system"""
        try:
            from datetime import datetime, timezone

            # Try to parse various date formats
            for fmt in [
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%Y/%m/%d',
            ]:
                try:
                    parsed_date = datetime.strptime(date_str.strip(), fmt)
                    if parsed_date.tzinfo is None:
                        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                    break
                except:
                    continue
            else:
                return ""  # Could not parse date

            # Calculate age
            now = datetime.now(timezone.utc)
            age_days = (now - parsed_date).days

            # Traffic light system
            if age_days <= 7:
                return "🟢"  # 1-7 days: Green (fresh)
            elif age_days <= 28:
                return "🟡"  # 1-4 weeks: Yellow
            elif age_days <= 180:
                return "🟠"  # 1-6 months: Orange
            elif age_days <= 365:
                return "🔴"  # 6-12 months: Red
            else:
                return "⚫"  # 1+ years: Black

        except Exception as e:
            print(f"Warning: Could not calculate age emoji: {e}", file=sys.stderr)
            return ""

    def extract_page_metadata(self, response) -> Dict[str, str]:
        """Extract metadata from Gemini response if available"""
        metadata = {}

        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                # Check for grounding metadata
                if hasattr(candidate, 'grounding_metadata'):
                    grounding = candidate.grounding_metadata

                    # Extract publish date if available
                    if hasattr(grounding, 'web_search_queries'):
                        for query in grounding.web_search_queries:
                            if hasattr(query, 'publish_date'):
                                metadata['publish_date'] = str(query.publish_date)

                    # Extract from grounding chunks
                    if hasattr(grounding, 'grounding_chunks'):
                        for chunk in grounding.grounding_chunks:
                            if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                                if hasattr(chunk.web, 'title'):
                                    metadata['title'] = chunk.web.title

                # Check candidate content for metadata
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            # Look for date patterns in response
                            import re
                            date_patterns = [
                                r'published[:\s]+(\d{4}-\d{2}-\d{2})',
                                r'updated[:\s]+(\d{4}-\d{2}-\d{2})',
                                r'date[:\s]+(\d{4}-\d{2}-\d{2})',
                            ]
                            for pattern in date_patterns:
                                match = re.search(pattern, part.text, re.IGNORECASE)
                                if match and 'publish_date' not in metadata:
                                    metadata['publish_date'] = match.group(1)
                                    break
        except Exception as e:
            print(f"Warning: Could not extract metadata: {e}", file=sys.stderr)

        return metadata

    async def summarize_url(self, url: str, prompt_override: str = None) -> Dict[str, Any]:
        """Fetch URL and generate UK English summary with topic matching"""
        try:
            # Check if YouTube and try to get transcript
            is_youtube = 'youtube.com' in url or 'youtu.be' in url
            transcript_text = None
            youtube_metadata = {}

            if is_youtube:
                transcript_text, youtube_metadata = await self.get_youtube_transcript(url)

            # Build prompt based on what data we have
            if transcript_text:
                # Have transcript - use it
                base_prompt = f"""Summarise the following YouTube video transcript in UK English spelling.
The video is from: {url}

TRANSCRIPT:
{transcript_text[:4000]}  # Limit to first 4000 chars to avoid token limits

Format the summary as clear bullet points:
- Use proper UK English spelling (summarise, colour, organise, etc.)
- Create concise, informative bullet points
- Each point should be a complete thought
- Focus on main ideas, key facts, and actionable insights
- Do NOT add any [[wiki links]] or **bold** formatting - just plain text

Summary:"""
            elif is_youtube and not transcript_text:
                # YouTube video without transcript or metadata - skip to avoid hallucinations
                return {
                    "success": False,
                    "error": "YouTube transcript not available and API disabled",
                    "url": url,
                    "message": "Cannot summarize YouTube video without transcript access"
                }
            elif youtube_metadata and youtube_metadata.get('title'):
                # Have metadata from YouTube API - use it instead of hallucinating
                title = youtube_metadata.get('title', '')
                description = youtube_metadata.get('description', '')[:500]  # Limit description
                channel = youtube_metadata.get('channel_title', '')

                base_prompt = f"""Summarise the following YouTube video based on its metadata in UK English spelling.

VIDEO TITLE: {title}
CHANNEL: {channel}
DESCRIPTION: {description}

Format the summary as clear bullet points:
- Use proper UK English spelling (summarise, colour, organise, etc.)
- Create concise, informative bullet points based on the title and description
- Focus on what the video is about based on the metadata provided
- Do NOT make up content - only summarize what's in the title and description
- Do NOT add any [[wiki links]] or **bold** formatting - just plain text

Summary:"""
            else:
                # Generic URL - let Gemini try
                base_prompt = f"""Summarise the content from {url} in UK English spelling.
Format the summary as clear bullet points:
- Use proper UK English spelling (summarise, colour, organise, etc.)
- Create concise, informative bullet points
- Each point should be a complete thought
- Focus on main ideas, key facts, and actionable insights
- Do NOT add any [[wiki links]] or **bold** formatting - just plain text

Summary:"""

            prompt = prompt_override if prompt_override else base_prompt

            # Generate summary
            response = self.model.generate_content(prompt)
            summary_text = response.text

            # Extract metadata and age emoji
            metadata = self.extract_page_metadata(response)

            # Use YouTube metadata if available
            if youtube_metadata and youtube_metadata.get('published_at'):
                metadata['publish_date'] = youtube_metadata['published_at']
                metadata['source'] = 'youtube_api'

            age_emoji = ""
            if 'publish_date' in metadata:
                age_emoji = self.get_age_emoji(metadata['publish_date'])

            # Use Claude to add topic links with semantic matching
            formatted_with_links, matched_topics = await self.add_topic_links_with_zai(summary_text, PERMITTED_TOPICS)

            # Add age emoji to beginning if available
            if age_emoji:
                formatted_with_links = f"{age_emoji} {formatted_with_links}"

            # Format for Logseq (add bullet points and line endings)
            formatted = self.format_logseq_markdown(formatted_with_links)

            result = {
                "success": True,
                "url": url,
                "summary": formatted,
                "raw_summary": summary_text,
                "matched_topics": list(matched_topics)
            }

            if age_emoji:
                result["age_emoji"] = age_emoji
                result["publish_date"] = metadata.get('publish_date', '')

            if metadata:
                result["metadata"] = metadata

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
                    },
                    {
                        "name": "expand_markdown_links",
                        "description": "Expand bare [[links]] in a Logseq markdown file with descriptive summaries and semantic topic matching. Overwrites the file with expanded content and creates a .backup file.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Absolute path to the markdown file to expand"
                                }
                            },
                            "required": ["file_path"]
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

            elif tool_name == "expand_markdown_links":
                result = await self.expand_markdown_links(
                    file_path=args.get("file_path")
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
    # Check for CLI mode
    if len(sys.argv) > 1:
        command = sys.argv[1]
        server = WebSummaryMCPServer()

        if command == "expand_markdown_links" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            result = await server.expand_markdown_links(file_path)
            print(json.dumps(result, indent=2))
            return
        elif command == "summarize_url" and len(sys.argv) > 2:
            url = sys.argv[2]
            result = await server.summarize_url(url)
            print(json.dumps(result, indent=2))
            return
        else:
            print("Usage: python web-summary-mcp-server.py [expand_markdown_links|summarize_url] <path|url>")
            return

    # Run as MCP server
    server = WebSummaryMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
