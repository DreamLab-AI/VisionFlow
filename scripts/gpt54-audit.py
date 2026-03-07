#!/usr/bin/env python3
"""
VisionFlow GPT-5.4 Codebase Audit
Sends assembled blobs to GPT-5.4's 1M context window for comprehensive audit.
"""

import json
import os
import sys
import time
import requests

API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-5.4"
BLOB_DIR = "/tmp/visionflow-blobs"
OUTPUT_DIR = "/tmp/visionflow-audit"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

AUDIT_SYSTEM_PROMPT = """You are a senior staff engineer conducting a comprehensive codebase audit of VisionFlow,
a 3D knowledge graph visualization platform. You have been given the complete source code
concatenated into blob files with metadata headers for each file.

The system architecture is:
- Rust/Actix-web backend with Actor system, CQRS, Hexagonal Architecture, CUDA GPU compute
- TypeScript/React 19 frontend with Three.js/R3F, WebAssembly scene effects, Zustand stores
- Neo4j graph database, Nostr authentication, Binary WebSocket protocol
- Features: OWL ontology reasoning, semantic analysis, graph algorithms, XR/VR, AI integrations

For each audit section, provide:
1. Specific file paths and line references
2. Severity (CRITICAL / HIGH / MEDIUM / LOW / INFO)
3. Concrete recommendations with code patterns

Be thorough but structured. Focus on actionable findings."""


def read_blob(name):
    path = os.path.join(BLOB_DIR, name)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def send_audit(label, messages, max_output=16000):
    """Send audit request to GPT-5.4"""
    print(f"\n{'='*60}")
    print(f"AUDIT: {label}")
    print(f"{'='*60}")

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": max_output,
        "temperature": 0.2,
    }

    # Estimate input tokens
    total_chars = sum(len(json.dumps(m)) for m in messages)
    est_tokens = total_chars // 4
    print(f"Estimated input: ~{est_tokens:,} tokens")
    if est_tokens > 272000:
        print(f"NOTE: >272K tokens - 2x pricing applies")

    start = time.time()
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=600,  # 10 min timeout for large context
        )
        elapsed = time.time() - start
        print(f"Response time: {elapsed:.1f}s")

        data = resp.json()
        if "error" in data:
            print(f"ERROR: {data['error']}")
            return None

        usage = data.get("usage", {})
        print(f"Actual input tokens:  {usage.get('prompt_tokens', '?'):,}")
        print(f"Output tokens:        {usage.get('completion_tokens', '?'):,}")
        print(f"Total tokens:         {usage.get('total_tokens', '?'):,}")

        content = data["choices"][0]["message"]["content"]
        return content

    except requests.exceptions.Timeout:
        print(f"TIMEOUT after {time.time()-start:.0f}s")
        return None
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # Read blobs
    print("Reading blobs...")
    server_blob = read_blob("server-blob.txt")
    client_blob = read_blob("client-blob.txt")
    spec_blob = read_blob("spec-blob.txt")

    print(f"Server: {len(server_blob):,} chars (~{len(server_blob)//4:,} tokens)")
    print(f"Client: {len(client_blob):,} chars (~{len(client_blob)//4:,} tokens)")
    print(f"Spec:   {len(spec_blob):,} chars (~{len(spec_blob)//4:,} tokens)")
    total_chars = len(server_blob) + len(client_blob) + len(spec_blob)
    print(f"Total:  {total_chars:,} chars (~{total_chars//4:,} tokens)")

    # ================================================================
    # AUDIT 1: Full codebase - Architecture & Security
    # ================================================================
    audit1_prompt = """Perform a comprehensive architecture and security audit of this complete codebase.

REQUIRED SECTIONS:

## 1. Architecture Assessment
- Evaluate the hexagonal/CQRS/actor architecture implementation
- Identify architectural violations or inconsistencies
- Assess the separation of concerns between layers
- Review the CUDA/GPU integration architecture
- Evaluate the WebSocket binary protocol design

## 2. Security Audit
- Authentication: Review the Nostr NIP-98 auth implementation
- Authorization: Check for missing auth on endpoints
- Input validation: Find any unvalidated user inputs
- SQL/Cypher injection: Check Neo4j query construction
- WebSocket security: Check for message validation gaps
- CUDA safety: Check for unsafe GPU memory operations
- Secrets management: Check for hardcoded credentials or leaked secrets
- Rate limiting: Evaluate rate limit implementation

## 3. Data Flow Integrity
- Trace data from API endpoint to database and back
- Identify any data transformation inconsistencies
- Check for type mismatches between client/server
- Evaluate the binary protocol serialization safety

## 4. Diagrams vs Reality
- Compare the architecture diagrams in the spec section against actual code
- Identify any diagrams that don't match the implementation
- Note any undocumented components or data flows

## 5. Critical Bugs & Race Conditions
- Identify potential race conditions in the actor system
- Check for deadlock potential between actors
- Review async/await patterns for correctness
- Check CUDA kernel launch safety

For each finding, provide: file path, line reference (from the // FILE: headers), severity, and fix.
"""

    messages1 = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Here is the complete VisionFlow codebase for audit.

=== FUNCTIONAL SPECIFICATION & ARCHITECTURE DIAGRAMS ===
{spec_blob}

=== SERVER CODEBASE (Rust/Actix-web) ===
{server_blob}

=== CLIENT CODEBASE (TypeScript/React/Three.js) ===
{client_blob}

{audit1_prompt}"""},
    ]

    result1 = send_audit("Full Codebase - Architecture & Security", messages1, max_output=16000)
    if result1:
        out1 = os.path.join(OUTPUT_DIR, "audit-architecture-security.md")
        with open(out1, "w") as f:
            f.write(f"# VisionFlow Architecture & Security Audit\n")
            f.write(f"**Model**: {MODEL}\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M UTC')}\n\n")
            f.write(result1)
        print(f"\nSaved: {out1}")

    # ================================================================
    # AUDIT 2: Code Quality & Performance
    # ================================================================
    audit2_prompt = """Now perform a code quality and performance audit.

REQUIRED SECTIONS:

## 1. Code Quality
- Dead code: Functions/modules defined but never called
- Duplicated logic: Copy-paste patterns that should be abstracted
- Error handling: Inconsistent error handling patterns
- Naming conventions: Inconsistencies between modules
- Documentation gaps: Public APIs without documentation

## 2. Performance Analysis
- GPU utilization: Is the CUDA pipeline optimal?
- Memory management: Memory leaks or excessive allocations
- WebSocket throughput: Binary protocol efficiency
- Database queries: N+1 queries or missing indexes
- Frontend rendering: Three.js performance bottlenecks
- Bundle size concerns: Large dependencies or unnecessary imports

## 3. Client-Server Contract
- Type alignment between Rust types and TypeScript types
- WebSocket message format consistency
- API endpoint contract verification
- Settings schema synchronization

## 4. Testing Gaps
- Which critical paths lack test coverage?
- Which complex algorithms need property-based tests?
- Integration test gaps between client and server

## 5. Technical Debt Inventory
- Ranked list of technical debt items by impact
- Estimated effort for each (S/M/L/XL)
- Recommended prioritization

For each finding, provide: file path, severity, and concrete recommendation.
"""

    messages2 = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Here is the complete VisionFlow codebase for quality audit.

=== FUNCTIONAL SPECIFICATION ===
{spec_blob}

=== SERVER CODEBASE (Rust/Actix-web) ===
{server_blob}

=== CLIENT CODEBASE (TypeScript/React/Three.js) ===
{client_blob}

{audit2_prompt}"""},
    ]

    result2 = send_audit("Full Codebase - Quality & Performance", messages2, max_output=16000)
    if result2:
        out2 = os.path.join(OUTPUT_DIR, "audit-quality-performance.md")
        with open(out2, "w") as f:
            f.write(f"# VisionFlow Code Quality & Performance Audit\n")
            f.write(f"**Model**: {MODEL}\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M UTC')}\n\n")
            f.write(result2)
        print(f"\nSaved: {out2}")

    # Summary
    print(f"\n{'='*60}")
    print("AUDIT COMPLETE")
    print(f"{'='*60}")
    print(f"Results in: {OUTPUT_DIR}/")
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size:,} bytes")


if __name__ == "__main__":
    main()
