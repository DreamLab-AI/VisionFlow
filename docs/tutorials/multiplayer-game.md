---
title: "Tutorial: Building a Multiplayer Game Lobby with VisionFlow"
description: Create a multiplayer game lobby backed by VisionFlow's knowledge graph, with LiveKit voice chat, Vircadia spatial synchronisation, and player entity graphs.
category: tutorial
tags:
  - multiplayer
  - gaming
  - livekit
  - vircadia
  - tutorial
  - real-time
updated-date: 2026-02-12
difficulty-level: intermediate
---

# Tutorial: Building a Multiplayer Game Lobby with VisionFlow

This tutorial walks you through building an interactive game lobby where players
appear as nodes in a live 3D graph, communicate with spatial voice via LiveKit,
and synchronise state through Vircadia. By the end you will have a working
prototype that can host 50 players in a single lobby.

## Prerequisites

- VisionFlow stack running (`docker compose --profile dev up -d`).
- LiveKit server accessible (local Docker instance or LiveKit Cloud). See the
  [LiveKit quick-start](https://docs.livekit.io/realtime/quickstarts/) if you
  need to set one up.
- Familiarity with the [First Graph tutorial](first-graph.md).
- Optional: Vircadia domain server for XR support.

## What You Will Build

1. A lobby graph where each player is a node and friendships or party
   memberships are edges.
2. LiveKit-powered voice chat with spatial audio positioned according to graph
   layout.
3. Vircadia entity sync so that XR-equipped players can walk through the lobby
   graph.
4. A matchmaking trigger that groups players into teams based on graph
   clustering.

## Step 1 -- Create the Lobby Graph

Model the lobby as a VisionFlow graph. Each player joining the lobby creates a
node; party invitations create edges.

```bash
# Simulate 6 players joining
for name in Alice Bob Carol Dave Eve Frank; do
  curl -s -X POST http://localhost:3030/api/nodes \
    -H "Content-Type: application/json" \
    -d "{\"label\": \"$name\", \"type\": \"Player\", \"status\": \"idle\"}"
done

# Create party edges
curl -s -X POST http://localhost:3030/api/edges \
  -H "Content-Type: application/json" \
  -d '{"source": "Alice", "target": "Bob", "type": "PARTY_WITH"}'
curl -s -X POST http://localhost:3030/api/edges \
  -H "Content-Type: application/json" \
  -d '{"source": "Carol", "target": "Dave", "type": "PARTY_WITH"}'
```

Open **http://localhost:3030**. You will see six player nodes; Alice-Bob and
Carol-Dave are pulled together by the physics engine because their party edges
act as springs.

## Step 2 -- Integrate LiveKit Voice

VisionFlow's frontend can subscribe to a LiveKit room and map audio tracks to
player nodes. Configure the connection in the UI:

1. Open the **Voice** panel in the left sidebar.
2. Enter your LiveKit server URL and an API key with room-join permissions.
3. Click **Connect**. Each player who publishes an audio track will see their
   node pulse with a glow effect when they speak.

Because VisionFlow renders nodes in 3D space, it passes each node's XYZ
coordinates to the LiveKit spatial audio API. Players wearing headphones hear
voices originate from the direction of the speaking node, creating an intuitive
sense of who is talking.

## Step 3 -- Enable Vircadia Spatial Sync

For XR users, VisionFlow publishes entity state to a Vircadia domain server:

1. Set `VIRCADIA_DOMAIN_URL` in your `.env` file to point at your Vircadia
   instance.
2. Restart the VisionFlow container.
3. Each player node is now mirrored as a Vircadia entity. Users on Meta Quest 3
   can enter the domain, see floating player avatars arranged in the same
   layout as the 2D graph, and use hand gestures to interact.

Desktop and XR views stay synchronised -- moving a node in the browser updates
its position in Vircadia and vice versa.

## Step 4 -- Build a Matchmaking Trigger

Use VisionFlow's graph analytics to group players into balanced teams. The
Leiden clustering algorithm, accelerated on the GPU, partitions the lobby graph
based on connectivity:

```bash
curl -s -X POST http://localhost:3030/api/analytics/cluster \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "leiden", "resolution": 1.0}'
```

The response assigns each player to a community. Map communities to teams:

```bash
# Example response: Alice,Bob -> community 0; Carol,Dave -> community 1; Eve,Frank -> community 2
```

In the 3D view, nodes are coloured by community, giving players a visual
preview of their team assignment before the match starts.

## Step 5 -- Handle Real-Time Events

As players join, leave, or change status, push updates over the binary
WebSocket connection. VisionFlow's 34-byte frame format supports status flag
changes at minimal bandwidth cost:

| Event | Action |
|-------|--------|
| Player joins | Create node via REST, physics engine adds it smoothly |
| Player leaves | Remove node; edges retract and remaining nodes rebalance |
| Player speaks | LiveKit track event triggers node glow animation |
| Ready-up | Update node `status` property; node colour shifts to green |
| Match start | Cluster nodes fly apart into team groupings (animated transition) |

## Performance Expectations

| Lobby Size | FPS (RTX 4090) | FPS (Integrated GPU) |
|-----------|----------------|---------------------|
| 10 players | 60 | 60 |
| 50 players | 60 | 45 |
| 200 players | 60 | 20 (recommend dedicated GPU) |

## Next Steps

- [Case Study: P2P Gaming Network](../use-cases/case-studies/gaming-p2p.md) --
  A production deployment using these patterns at scale.
- [Real-Time Sync Concepts](../explanation/concepts/real-time-sync.md) -- How
  VisionFlow keeps all clients in lockstep.
- [Industry Applications -- Gaming](../use-cases/industry-applications.md#1-gaming--interactive-media)
  -- Broader context on gaming use cases.

---

**Document Version**: 1.0
**Last Updated**: 2026-02-12
**Maintained By**: VisionFlow Documentation Team
