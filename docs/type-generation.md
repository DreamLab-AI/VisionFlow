# TypeScript Type Generation from Rust

This document describes the automatic TypeScript type generation system that keeps Rust and TypeScript types synchronized.

## Overview

The type generation system automatically converts Rust struct definitions from the `webxr::config` module into TypeScript interfaces. This ensures type safety between the Rust backend and TypeScript frontend.

## Key Features

- **Automatic snake_case to camelCase conversion** - Rust field names are automatically converted to TypeScript naming conventions
- **Comprehensive type coverage** - All settings structures are included
- **Type guards** - Runtime type checking functions are generated
- **Utility types** - Helper types like `DeepPartial` and `NestedSettings` are included
- **Build integration** - Types are generated as part of the build process

## Generated Types Structure

The generated types are organized into several main categories:

### Core Settings
- `AppFullSettings` - Main application configuration
- `VisualisationSettings` - 3D visualization and rendering settings
- `SystemSettings` - Network, WebSocket, and system configuration
- `XRSettings` - WebXR and VR/AR specific settings
- `AuthSettings` - Authentication configuration

### Physics & Rendering
- `PhysicsSettings` - Physics simulation parameters
- `AutoBalanceConfig` - Auto-balancing configuration
- `RenderingSettings` - 3D rendering options
- `AnimationSettings` - Animation and effects settings
- `GlowSettings` - Glow and lighting effects
- `HologramSettings` - Hologram visualization settings

### External Services
- `RagFlowSettings` - RAG Flow AI service configuration
- `PerplexitySettings` - Perplexity AI API settings
- `OpenAISettings` - OpenAI API configuration
- `KokoroSettings` - Kokoro TTS service settings
- `WhisperSettings` - Whisper STT service settings

### Utility Types
- `Position` - 3D coordinates (x, y, z)
- `MovementAxes` - Movement control axes
- `Sensitivity` - Input sensitivity settings
- `DeepPartial<T>` - Deep partial type for updates
- `NestedSettings` - Partial application settings

## File Locations

- **Generated Types**: `/client/src/types/generated/settings.ts`
- **Generation Script**: `/src/bin/generate_types.rs` (Rust implementation)
- **Standalone Generator**: `/generate_types_standalone.py` (Python fallback)

## Build Scripts

### NPM Scripts

```json
{
  "types:generate": "cd .. && cargo run --bin generate_types",
  "types:watch": "cd .. && cargo watch -x 'run --bin generate_types'",
  "types:clean": "rm -rf src/types/generated",
  "prebuild": "npm run types:generate",
  "build": "npm run types:generate && vite build"
}
```

### Cargo Configuration

```toml
[[bin]]
name = "generate_types"
path = "src/bin/generate_types.rs"
```

## Type Generation Process

1. **Analysis Phase**: The Rust binary reads all struct definitions from `src/config/mod.rs`
2. **Conversion Phase**: Field names are converted from snake_case to camelCase
3. **Generation Phase**: TypeScript interfaces are generated with proper types
4. **Enhancement Phase**: Type guards and utility types are added
5. **Output Phase**: Final `.ts` file is written to the client types directory

## Case Conversion Examples

| Rust Field | TypeScript Field |
|------------|------------------|
| `base_color` | `baseColor` |
| `enable_bounds` | `enableBounds` |
| `auto_balance_interval_ms` | `autoBalanceIntervalMs` |
| `api_key` | `apiKey` |
| `max_request_size` | `maxRequestSize` |

## Usage in TypeScript

### Basic Usage

```typescript
import { AppFullSettings, Settings, Position } from '@/types/generated/settings';

// Type-safe settings object
const settings: AppFullSettings = {
  visualisation: {
    rendering: {
      backgroundColor: '#000000',
      ambientLightIntensity: 0.5
    }
    // ... other settings
  }
  // ... other settings
};

// Partial updates
const update: Partial<Settings> = {
  xr: {
    enableHandTracking: true
  }
};
```

### Type Guards

```typescript
import { isAppFullSettings, isPosition } from '@/types/generated/settings';

// Runtime type checking
if (isAppFullSettings(data)) {
  // data is now typed as AppFullSettings
  console.log(data.visualisation.rendering.backgroundColor);
}

if (isPosition(coords)) {
  // coords is now typed as Position
  console.log(`Position: ${coords.x}, ${coords.y}, ${coords.z}`);
}
```

### Utility Types

```typescript
import { DeepPartial, NestedSettings } from '@/types/generated/settings';

// Deep partial updates
const partialUpdate: NestedSettings = {
  visualisation: {
    rendering: {
      backgroundColor: '#ffffff'
    }
  }
};

// Apply partial update function
function updateSettings(current: Settings, update: DeepPartial<Settings>): Settings {
  return { ...current, ...update };
}
```

## Maintenance

### Keeping Types Synchronized

Types are automatically regenerated on every build, but you can manually trigger generation:

```bash
# Generate types manually
npm run types:generate

# Watch for changes and regenerate automatically
npm run types:watch

# Clean generated types
npm run types:clean
```

### Adding New Rust Types

1. Add your new struct to `/src/config/mod.rs`
2. Ensure it derives `serde::Serialize`, `serde::Deserialize`, and `specta::Type`
3. Add it to the list in `/src/bin/generate_types.rs`
4. Run `npm run types:generate` to regenerate types

### Updating Field Names

When changing Rust field names:

1. Update the struct definition in `/src/config/mod.rs`
2. Regenerate types with `npm run types:generate`
3. Update any TypeScript code using the old field names
4. The camelCase conversion is automatic

## Troubleshooting

### Common Issues

**Types not updating after Rust changes**
- Solution: Run `npm run types:generate` manually
- Check that the Rust binary compiles successfully

**Build fails with missing types**
- Solution: Ensure `prebuild` script is working: `npm run types:generate`
- Check that the generated file exists: `client/src/types/generated/settings.ts`

**Field name mismatches**
- Solution: Verify the snake_case to camelCase conversion is working correctly
- Check the generated file for the expected field names

**TypeScript compilation errors**
- Solution: Ensure all interfaces are properly exported
- Check for missing optional field markers (`?`)

### Validation

To verify the type generation is working:

```bash
# Check generated file exists and has content
ls -la client/src/types/generated/settings.ts
wc -l client/src/types/generated/settings.ts

# Verify types compile
cd client && npm run build
```

## Development Workflow

1. **Make changes to Rust config structs**
2. **Test Rust compilation**: `cargo build`
3. **Generate TypeScript types**: `npm run types:generate`
4. **Update TypeScript code** to use new/changed types
5. **Test TypeScript compilation**: `npm run build`
6. **Commit both Rust and TypeScript changes**

## Future Improvements

- **Automated validation**: Add tests to ensure type compatibility
- **Documentation generation**: Generate JSDoc comments from Rust docs
- **Schema validation**: Generate JSON schema for runtime validation
- **IDE integration**: Better TypeScript language server support
- **Watch mode**: Automatically regenerate on Rust file changes