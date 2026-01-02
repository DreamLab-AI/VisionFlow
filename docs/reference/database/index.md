---
layout: default
title: Database Reference
parent: Reference
nav_order: 2
has_children: true
permalink: /reference/database/
---

# Database Reference

Database schemas, queries, and persistence patterns for VisionFlow.

## Overview

VisionFlow uses a dual-database architecture:

- **SQLite (unified.db)**: Fast local queries, physics state, primary source of truth
- **Neo4j**: Complex graph traversals, multi-hop reasoning, semantic analysis

## Contents

| Document | Description |
|----------|-------------|
| [Schemas](./schemas.md) | Complete database schema definitions |
| [Neo4j Persistence Analysis](./neo4j-persistence-analysis.md) | Neo4j persistence patterns |
| [Ontology Schema v2](./ontology-schema-v2.md) | Version 2 ontology schema |
| [SOLID Pod Schema](./solid-pod-schema.md) | SOLID Pod data schema |
| [User Settings Schema](./user-settings-schema.md) | User settings persistence |
