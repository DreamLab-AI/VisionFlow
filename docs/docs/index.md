# Documentation Index

This directory contains all organized documentation for the client application.

## Structure

```
docs/
├── README.md                          # Main documentation entry point
├── index.md                          # This file - documentation index
├── architecture/                     # System architecture documentation
│   ├── API_ARCHITECTURE_ANALYSIS.md  # API layer analysis
│   └── telemetry-system-analysis.md  # Telemetry and WebSocket architecture
├── api/                              # API-specific documentation
│   └── unified-api-client.md         # Unified API client guide
├── features/                         # Feature-specific documentation
│   ├── telemetry.md                  # Telemetry system guide
│   └── polling-system.md             # Bot polling system
├── guides/                           # User and developer guides
│   └── testing.md                    # Testing approach and integration tests
└── troubleshooting/                  # Problem resolution guides
    ├── DUPLICATE_POLLING_FIX_SUMMARY.md
    └── SECURITY_ALERT.md
```

## Navigation

- **Start Here**: [README.md](./README.md) - Main documentation overview
- **Architecture**: [architecture/](./architecture/) - System design and analysis
- **API Reference**: [api/](./api/) - API client documentation
- **Features**: [features/](./features/) - Individual feature documentation
- **Guides**: [guides/](./guides/) - Step-by-step instructions
- **Troubleshooting**: [troubleshooting/](./troubleshooting/) - Issue resolution

## Quick Access

### Most Important Documents
1. [API Architecture Analysis](./architecture/API_ARCHITECTURE_ANALYSIS.md) - Understanding the three-layer API system
2. [Unified API Client](./api/unified-api-client.md) - Using the centralized HTTP client
3. [Telemetry System](./features/telemetry.md) - Debugging and monitoring tools

### Recently Updated
- Unified API Client documentation (comprehensive usage guide)
- Telemetry system integration (React hooks and debug overlay)
- Architecture analysis (identifies API inconsistencies)

### External References
- [Vircadia Web](../src/vircadia/vircadia-web/README.md) - Main Vircadia web client
- [Service APIs](../src/services/api/README.md) - Unified API service
- [Telemetry Source](../src/telemetry/README.md) - Telemetry implementation