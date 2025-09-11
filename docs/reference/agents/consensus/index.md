# Consensus Agents

This section contains documentation for agents that implement distributed consensus algorithms and coordination mechanisms.

## Contents

### Core Consensus Agents
- **[byzantine-coordinator.md](byzantine-coordinator.md)** - Byzantine fault-tolerant consensus coordination
- **[crdt-synchronizer.md](crdt-synchronizer.md)** - Conflict-free Replicated Data Type synchronisation
- **[gossip-coordinator.md](gossip-coordinator.md)** - Gossip protocol implementation for distributed systems
- **[quorum-manager.md](quorum-manager.md)** - Quorum-based decision making and coordination
- **[raft-manager.md](raft-manager.md)** - Raft consensus algorithm implementation

### Supporting Agents
- **[performance-benchmarker.md](performance-benchmarker.md)** - Performance testing for consensus algorithms
- **[security-manager.md](security-manager.md)** - Security management for distributed consensus

## Purpose

Consensus agents provide distributed coordination capabilities including:

- Byzantine fault tolerance for unreliable networks
- Eventual consistency through CRDT synchronisation
- Efficient gossip-based information propagation
- Quorum-based decision making
- Leader election and log replication via Raft

## Agent Categories

### Fault-Tolerant Agents
Agents designed to operate correctly even when some nodes fail or behave maliciously.

### Synchronisation Agents
Agents focused on maintaining data consistency across distributed systems.

### Communication Agents
Agents that handle information propagation and coordination messaging.

### Security Agents
Agents that ensure secure operation of consensus mechanisms.

## Consensus Algorithms

### Byzantine Fault Tolerance
Handles arbitrary node failures and malicious behaviour in distributed systems.

### CRDT Synchronisation
Provides eventual consistency without requiring coordination.

### Gossip Protocols
Efficient epidemic-style information dissemination.

### Raft Consensus
Leader-based consensus with strong consistency guarantees.

### Quorum Systems
Majority-based decision making for high availability.