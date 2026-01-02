---
layout: default
title: Architecture
parent: Guides
nav_order: 31
has_children: true
description: System architecture documentation for VisionFlow
---

# Architecture Guides

Technical architecture documentation for VisionFlow system components.

## Available Guides

| Guide | Description |
|-------|-------------|
| [Actor System](actor-system.md) | Actix actor patterns, supervision, and concurrency |

## Key Concepts

### Actor System

VisionFlow uses Actix actors for:
- Concurrent graph operations
- Physics simulation orchestration
- Real-time client coordination
- Fault-tolerant message processing

## See Also

- [Main Guides](../index.md)
- [Infrastructure](../infrastructure/index.md)
