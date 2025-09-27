# Client Documentation

Welcome to the comprehensive documentation for the hive mind visualization client application.

## 📖 Table of Contents

### 🏗️ Architecture
- [API Architecture Analysis](./architecture/API_ARCHITECTURE_ANALYSIS.md) - Comprehensive analysis of the three-layer API system
- [Telemetry System Analysis](./architecture/telemetry-system-analysis.md) - WebSocket protocol and data flow architecture

### 🔌 API Documentation
- [Unified API Client](./api/unified-api-client.md) - Centralized HTTP client for all API operations

### ✨ Features
- [Telemetry System](./features/telemetry.md) - Real-time debugging and performance monitoring
- [Polling System](./features/polling-system.md) - Bot status and data polling implementation

### 📚 Guides
- [Testing Guide](./guides/testing.md) - Integration test summary and testing approach

### 🔧 Troubleshooting
- [Duplicate Polling Fix](./troubleshooting/DUPLICATE_POLLING_FIX_SUMMARY.md) - Resolution of polling duplication issues
- [Security Alert](./troubleshooting/SECURITY_ALERT.md) - Important security considerations

## 🚀 Quick Start

1. **API Usage**: Start with the [Unified API Client](./api/unified-api-client.md) for making HTTP requests
2. **Debugging**: Enable the [Telemetry System](./features/telemetry.md) for real-time monitoring
3. **Architecture**: Review the [API Architecture Analysis](./architecture/API_ARCHITECTURE_ANALYSIS.md) to understand system structure

## 🎯 Key Features

- **Three-Layer API System**: Unified client, specialized services, and legacy endpoints
- **Real-time Telemetry**: Performance monitoring and debugging overlay
- **WebSocket Protocol**: Binary data streaming for agent positions
- **Comprehensive Testing**: Integration tests with detailed reporting

## 📋 System Status

### Documentation Coverage
- **Total Files**: 15 markdown files organized
- **Architecture Docs**: 2 comprehensive analyses
- **Feature Docs**: 3 detailed guides
- **API Docs**: 1 unified client guide
- **Troubleshooting**: 2 resolution guides

### Implementation Status
- ✅ **Unified API Client**: Fully implemented and documented
- ✅ **Telemetry System**: Complete with React hooks and debug overlay
- ✅ **WebSocket Protocol**: Binary streaming for real-time data
- ✅ **Integration Tests**: Comprehensive test coverage

## 🔗 Related Documentation

### Vircadia Integration
- [Vircadia Web Client](../src/vircadia/vircadia-web/README.md) - Web-based 3D client
- [Vircadia World](../src/vircadia/vircadia-world/README.md) - Virtual world server
- [Desktop Client](../src/vircadia/vircadia-web/desktop/README.md) - Tauri desktop wrapper

### Technical Specifications
- [Connection States](../src/vircadia/vircadia-web/docs/NotesOnConnectionAccountStates.md) - Account and connection management
- [Architecture Overview](../src/vircadia/vircadia-web/docs/Architecture.md) - Vircadia web architecture

## 🤝 Contributing

When updating documentation:

1. Keep files organized in appropriate subdirectories
2. Update this main README when adding new sections
3. Ensure cross-references use relative paths
4. Validate implementation matches documentation

## 📞 Support

For technical issues or questions:
- Check [Troubleshooting](./troubleshooting/) section first
- Review relevant architecture documentation
- Enable telemetry system for debugging