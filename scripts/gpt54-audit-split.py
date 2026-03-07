#!/usr/bin/env python3
"""
VisionFlow GPT-5.4 Codebase Audit (Split for 922K token limit)
3 passes: Server Part 1 + Spec, Server Part 2, Client
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

SYSTEM_PROMPT = """You are a senior staff engineer conducting a comprehensive codebase audit of VisionFlow,
a 3D knowledge graph visualization platform with Rust backend and TypeScript/React frontend.

Architecture: Actix-web actors + CQRS + Hexagonal + CUDA GPU, Neo4j, Nostr auth, Binary WebSocket, OWL ontology.

For each finding provide: file path (from // FILE: headers), severity (CRITICAL/HIGH/MEDIUM/LOW/INFO), and concrete fix."""


def read_file(name):
    path = os.path.join(BLOB_DIR, name)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def send_request(label, messages, max_output=16000):
    print(f"\n{'='*60}")
    print(f"PASS: {label}")
    print(f"{'='*60}")

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": max_output,
        "temperature": 0.2,
    }

    total_chars = sum(len(m.get("content", "")) for m in messages)
    est_tokens = int(total_chars / 4.46)
    print(f"Content: {total_chars:,} chars (~{est_tokens:,} tokens)")

    if est_tokens > 922000:
        print(f"WARNING: Estimated {est_tokens:,} > 922K limit. May fail.")

    start = time.time()
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=600,
        )
        elapsed = time.time() - start
        data = resp.json()

        if "error" in data:
            print(f"ERROR after {elapsed:.1f}s: {data['error'].get('message', str(data['error']))}")
            return None

        usage = data.get("usage", {})
        print(f"Time: {elapsed:.1f}s | Input: {usage.get('prompt_tokens', '?'):,} | Output: {usage.get('completion_tokens', '?'):,}")
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        print(f"TIMEOUT after {time.time()-start:.0f}s")
        return None
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return None


def save_result(filename, title, content):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(f"# {title}\n")
        f.write(f"**Model**: {MODEL} | **Date**: {time.strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(content)
    print(f"Saved: {path} ({os.path.getsize(path):,} bytes)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    server_p1 = read_file("server-blob-part1.txt")
    server_p2 = read_file("server-blob-part2.txt")
    client = read_file("client-blob.txt")
    spec = read_file("spec-blob.txt")

    print(f"Server P1: {len(server_p1):,} chars (~{int(len(server_p1)/4.46):,} tok)")
    print(f"Server P2: {len(server_p2):,} chars (~{int(len(server_p2)/4.46):,} tok)")
    print(f"Client:    {len(client):,} chars (~{int(len(client)/4.46):,} tok)")
    print(f"Spec:      {len(spec):,} chars (~{int(len(spec)/4.46):,} tok)")

    # ================================================================
    # PASS 1: Server Core (models, ports, adapters, CQRS, services, actors, handlers) + Spec
    # ================================================================
    r1 = send_request("Server Core + Architecture Spec", [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""ARCHITECTURE DIAGRAMS & SPEC:
{spec}

SERVER CODEBASE PART 1 (models, ports, adapters, CQRS, application, services, actors, handlers):
{server_p1}

Audit this server core code against the architecture spec. Cover:
1. **Architecture Compliance**: Does the code match the hexagonal/CQRS/actor diagrams?
2. **Security**: Auth bypass risks, injection vectors, unsafe GPU ops, missing validation
3. **Actor System**: Race conditions, deadlock potential, mailbox overflow risks
4. **Data Integrity**: Neo4j query safety, serialization correctness, type mismatches
5. **Diagrams vs Code**: Which documented components exist vs are phantom?
List all findings with file paths, severity, and fixes."""}
    ])
    if r1:
        save_result("01-server-core-audit.md", "Server Core & Architecture Audit", r1)

    # ================================================================
    # PASS 2: Server Infrastructure (GPU, physics, utils, config, middleware, events)
    # ================================================================
    r2 = send_request("Server Infrastructure", [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""SERVER CODEBASE PART 2 (GPU compute, physics, config, middleware, utils, events, constraints, types):
{server_p2}

Audit this server infrastructure code. Cover:
1. **CUDA/GPU Safety**: Memory management, kernel launch safety, error recovery
2. **Physics Engine**: Force computation correctness, convergence, numerical stability
3. **Middleware Stack**: Rate limiting, validation, auth middleware completeness
4. **Event System**: Event bus reliability, handler error isolation
5. **Binary Protocol**: Serialization/deserialization safety, buffer overflow risks
6. **Performance**: Allocation hotspots, unnecessary copies, optimization opportunities
7. **Dead Code**: Functions defined but never called, unreachable paths
List all findings with file paths, severity, and fixes."""}
    ])
    if r2:
        save_result("02-server-infra-audit.md", "Server Infrastructure Audit", r2)

    # ================================================================
    # PASS 3: Client Codebase
    # ================================================================
    r3 = send_request("Client Codebase", [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""CLIENT CODEBASE (TypeScript/React 19/Three.js/R3F):
{client}

Audit this complete client codebase. Cover:
1. **React Patterns**: Component lifecycle, hook dependencies, re-render optimization
2. **State Management**: Zustand store design, state normalization, subscription efficiency
3. **Three.js/WebGL**: Memory leaks (geometries, textures, materials disposal), frame budget
4. **WebSocket Client**: Reconnection logic, binary protocol handling, message queue
5. **WASM Integration**: Memory safety, bridge correctness, zero-copy pattern
6. **Type Safety**: TypeScript strictness gaps, any-casts, missing types
7. **Security**: XSS vectors, credential handling, input sanitization
8. **Performance**: Bundle size, code splitting, lazy loading gaps
9. **Accessibility**: ARIA compliance, keyboard navigation, screen reader support
List all findings with file paths, severity, and fixes."""}
    ])
    if r3:
        save_result("03-client-audit.md", "Client Codebase Audit", r3)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("ALL PASSES COMPLETE")
    print(f"{'='*60}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        p = os.path.join(OUTPUT_DIR, f)
        print(f"  {f}: {os.path.getsize(p):,} bytes")


if __name__ == "__main__":
    main()
