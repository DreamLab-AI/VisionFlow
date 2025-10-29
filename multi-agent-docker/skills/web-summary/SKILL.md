---
name: Web Summary
description: Fetch and summarize web content including YouTube videos with semantic topic links for Logseq
---

# Web Summary Skill

This skill provides comprehensive web content summarization capabilities, including YouTube transcript extraction and semantic topic linking for knowledge management systems like Logseq.

## Capabilities

- Fetch and summarize web pages
- Extract YouTube video transcripts
- Use Z.AI for cost-effective summarization
- Support for multiple output formats (markdown, plain text, JSON)
- ~~Generate semantic topic links~~ (Deprecated - migrating to new ontology system)

## When to Use This Skill

Use this skill when you need to:
- Summarize long articles or blog posts
- Extract and summarize YouTube video content
- Create knowledge base entries with topic links
- Generate research summaries from multiple sources
- Create Logseq-compatible notes with semantic links

## Instructions

### Basic Web Page Summarization

To summarize a web page:
1. Provide the URL of the web page
2. Specify desired summary length (short, medium, long)
3. Optionally request semantic topics for linking

### YouTube Video Summarization

To summarize a YouTube video:
1. Provide the YouTube URL or video ID
2. The skill will extract the transcript automatically
3. Generate a structured summary with key points
4. Create topic tags for knowledge management

### Semantic Topic Generation

For knowledge management integration:
1. Extract main topics from content
2. Generate [[topic-name]] links in Logseq format
3. Create hierarchical topic structure
4. Suggest related topics for linking

## Tool Functions

The skill provides these callable tools:

### `summarize_url`
Summarize content from any URL.

Parameters:
- `url` (required): The URL to summarize
- `length` (optional): "short" | "medium" | "long" (default: "medium")
- ~~`include_topics` (optional): boolean (default: true)~~ (Deprecated)

### `youtube_transcript`
Extract transcript from YouTube video.

Parameters:
- `video_id` (required): YouTube video ID or full URL
- `language` (optional): Language code (default: "en")

### ~~`generate_topics`~~ (DEPRECATED)
**This tool is deprecated.** The project is migrating to a new ontology system.

Previously generated semantic topic links from text using keyword extraction.

## Examples

### Example 1: Summarize a Blog Post
```
Use the Web Summary skill to summarize https://example.com/article with medium length and include semantic topics.
```

### Example 2: YouTube Video Summary
```
Extract and summarize the YouTube video https://www.youtube.com/watch?v=dQw4w9WgXcQ with topic tags for Logseq.
```

### Example 3: Batch Summarization
```
Summarize these 5 URLs and create a unified note with cross-referenced topics:
- https://example1.com
- https://example2.com
- https://youtube.com/watch?v=xyz
```

## Technical Details

- Uses Z.AI service for cost-effective summarization (invoked via internal ragflow network)
- Falls back to primary Claude API if Z.AI unavailable
- Supports rate limiting and retry logic
- Caches summaries for repeated requests
- Extracts metadata (title, author, date) when available

## Error Handling

The skill handles:
- Invalid URLs (returns error with suggestions)
- Unavailable YouTube transcripts (attempts multiple languages)
- Network timeouts (retries with exponential backoff)
- Rate limits (queues requests)

## Integration with Other Skills

Works well with:
- `filesystem` skill for saving summaries
- `git` skill for version controlling knowledge base
- `github` skill for creating issues from summaries
