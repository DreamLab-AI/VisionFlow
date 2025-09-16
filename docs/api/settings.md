# Settings API Reference

*This file redirects to the comprehensive settings API documentation.*

See [Settings API Reference](rest/settings.md) for complete settings API documentation.

## Quick Links

- [REST Settings Endpoints](rest/settings.md) - Complete settings API reference
- [Settings Schema](rest/settings.md#schema) - Configuration structure
- [Validation Rules](rest/settings.md#validation) - Input validation
- [WebSocket Settings Sync](websocket-streams.md#settings-sync) - Real-time synchronization

## Core Endpoints

### Settings Management
- `GET /api/settings` - Retrieve all settings
- `POST /api/settings` - Update settings
- `POST /api/settings/reset` - Reset to defaults
- `GET /api/settings/schema` - Get settings schema

### Configuration Categories
- **Visualization** - 3D rendering and UI settings
- **Physics** - Graph physics and simulation
- **Performance** - System performance options
- **GPU** - GPU acceleration settings
- **Network** - WebSocket and API configuration

### Validation and Security
- Server-side validation
- Type checking and constraints
- Sanitization of inputs
- Rate limiting protection

---

[← Back to API Documentation](README.md)