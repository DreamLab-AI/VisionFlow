# Bloom/Glow Field Mapping Implementation

## Overview

This document describes the implementation of bloom/glow field mapping between client and server in the VisionFlow application. The client uses both `bloom` and `glow` field names for backward compatibility, while the server internally uses `glow` as the primary field name.

## Problem Statement

- **Client**: Historical usage of `visualisation.bloom.*` fields
- **Server**: Internal usage of `glow` fields for consistency with rendering pipeline
- **Need**: Seamless compatibility without breaking existing components

## Solution Architecture

### 1. Transformation Functions (`/client/src/utils/caseConversion.ts`)

#### `transformBloomToGlow(settings)`
Transforms client bloom settings to server glow format:
- `bloom.strength` → `glow.intensity`
- `bloom.nodeBloomStrength` → `glow.nodeGlowStrength`
- `bloom.edgeBloomStrength` → `glow.edgeGlowStrength`
- `bloom.environmentBloomStrength` → `glow.environmentGlowStrength`

#### `transformGlowToBloom(settings)`
Transforms server glow settings to client bloom format for compatibility:
- `glow.intensity` → `bloom.strength`
- `glow.nodeGlowStrength` → `bloom.nodeBloomStrength`
- `glow.edgeGlowStrength` → `bloom.edgeBloomStrength`
- `glow.environmentGlowStrength` → `bloom.environmentBloomStrength`

#### `normalizeBloomGlowSettings(settings, direction)`
Main normalization function:
- `direction: 'toServer'` - Transforms bloom → glow for server submission
- `direction: 'toClient'` - Transforms glow → bloom for client compatibility

### 2. Settings Store Integration (`/client/src/store/settingsStore.ts`)

#### Server Communication
- **On Save**: Transform bloom fields to glow before sending to server
- **On Load**: Transform server glow fields to client bloom fields for compatibility
- **Logging**: Enhanced debugging for field transformation

#### Key Changes
```typescript
// Before sending to server
const serverSettings = transformBloomToGlow(settings);
await apiService.post('/settings', serverSettings, headers);

// After receiving from server
const clientCompatibleSettings = normalizeBloomGlowSettings(rawServerSettings, 'toClient');
const mergedSettings = deepMerge(defaultSettings, clientCompatibleSettings);
```

### 3. API Client Updates (`/client/src/api/settingsApi.ts`)

All API methods now handle field transformation:
- `fetchSettings()` - Normalizes server response to client format
- `updateSettings()` - Transforms client request to server format
- `saveSettings()` - Handles full settings transformation
- `resetSettings()` - Normalizes reset response

### 4. TypeScript Interface Updates (`/client/src/features/settings/config/settings.ts`)

#### New BloomSettings Interface
Added `BloomSettings` interface for backward compatibility:
```typescript
export interface BloomSettings {
  enabled: boolean;
  strength: number; // maps to glow.intensity
  nodeBloomStrength: number; // maps to glow.nodeGlowStrength
  edgeBloomStrength: number; // maps to glow.edgeGlowStrength
  environmentBloomStrength: number; // maps to glow.environmentGlowStrength
  // ... other compatibility fields
}
```

#### Updated VisualisationSettings
```typescript
export interface VisualisationSettings {
  glow: GlowSettings; // Primary (server-preferred)
  bloom?: BloomSettings; // Optional (client compatibility)
  // ... other settings
}
```

### 5. Default Settings Configuration

Updated `defaultSettings.ts` to include both bloom and glow with synchronized values:
- Glow settings as primary
- Bloom settings as compatibility layer with mapped field names
- Consistent default values between both interfaces

### 6. UI Component Updates

#### BackgroundEnvironmentControls
- Added effective settings computation for backward compatibility
- Dual update handlers that maintain both bloom and glow field synchronization
- Fallback logic: `glow.field ?? bloom.field ?? default`

#### Settings UI Definition
- Added glow section as primary UI
- Maintained bloom section for legacy compatibility
- Clear labeling to indicate which is preferred

#### Integrated Control Panel
- Prioritized glow settings in UI
- Added legacy bloom controls for backward compatibility
- Increased slider ranges to match glow settings (0-5 instead of 0-1)

### 7. Viewport Settings
Added glow patterns to immediate update triggers:
- `visualisation.glow`
- `visualisation.graphs.*.glow`

## Implementation Details

### Field Mapping Table

| Client Bloom Field | Server Glow Field | Description |
|-------------------|------------------|-------------|
| `bloom.enabled` | `glow.enabled` | Enable/disable effects |
| `bloom.strength` | `glow.intensity` | Overall effect intensity |
| `bloom.nodeBloomStrength` | `glow.nodeGlowStrength` | Node-specific strength |
| `bloom.edgeBloomStrength` | `glow.edgeGlowStrength` | Edge-specific strength |
| `bloom.environmentBloomStrength` | `glow.environmentGlowStrength` | Environment strength |
| `bloom.radius` | `glow.radius` | Effect radius |
| `bloom.threshold` | `glow.threshold` | Luminance threshold |

### Backward Compatibility Strategy

1. **Dual Field Support**: Both bloom and glow fields exist in client settings
2. **Automatic Synchronisation**: Updates to bloom fields update glow fields and vice versa
3. **Server Translation**: Client automatically translates bloom to glow when communicating with server
4. **Legacy Component Support**: Existing components referencing bloom continue to work

### Error Handling

- **Null/Undefined Input**: All transformation functions handle null/undefined gracefully
- **Missing Fields**: Fallback to defaults when fields are missing
- **Mixed Settings**: Handle settings objects containing both bloom and glow fields
- **API Errors**: Enhanced error logging for field transformation issues

## Testing

### Test Coverage (`/client/src/tests/bloom-glow-mapping.test.ts`)

- **transformBloomToGlow**: Tests bloom → glow transformation
- **transformGlowToBloom**: Tests glow → bloom transformation  
- **normalizeBloomGlowSettings**: Tests bidirectional normalization
- **Real-world Scenarios**: Tests server/client communication patterns
- **Edge Cases**: Tests null/undefined/empty inputs

### Test Scenarios

1. **Server Response Processing**: Glow settings → Client bloom compatibility
2. **Client Submission**: Bloom settings → Server glow format
3. **Mixed Settings**: Settings containing both bloom and glow fields
4. **Empty/Null Handling**: Graceful handling of missing data
5. **Field Precedence**: Proper precedence when both fields exist

## Usage Examples

### Component Implementation
```typescript
// Get effective settings with fallback
const effectiveSettings = {
  enabled: glowSettings.enabled ?? bloomSettings.enabled ?? false,
  strength: glowSettings.intensity ?? bloomSettings.strength ?? 0
};

// Update both fields for compatibility
const handleChange = useCallback((value) => {
  updateSettings((draft) => {
    // Update glow (server-preferred)
    if (!draft.visualisation.glow) draft.visualisation.glow = {} as any;
    draft.visualisation.glow.intensity = value;
    
    // Update bloom (client compatibility)
    if (!draft.visualisation.bloom) draft.visualisation.bloom = {} as any;
    draft.visualisation.bloom.strength = value;
  });
}, [updateSettings]);
```

### API Usage
```typescript
// Client automatically handles transformation
const settings = await settingsApi.fetchSettings(); // Glow → Bloom mapping applied
await settingsApi.saveSettings(settings); // Bloom → Glow mapping applied
```

## Migration Guide

### For New Components
- **Prefer glow fields**: Use `visualisation.glow.*` for new components
- **Add fallbacks**: Include bloom field fallbacks for robustness
- **Use effective settings**: Compute effective values from both field sets

### For Existing Components
- **No breaking changes**: Existing bloom references continue to work
- **Optional migration**: Gradually migrate to glow fields when convenient
- **Dual updates**: Ensure updates synchronize both bloom and glow fields

## Performance Considerations

- **Minimal Overhead**: Transformation functions are lightweight
- **Caching**: Settings store caches transformed values
- **Lazy Computation**: Transformations only run when needed
- **Memory Efficiency**: Objects are shallow-copied with field additions

## Future Considerations

1. **Phase Out Bloom**: Eventually deprecate bloom fields (breaking change)
2. **Server Validation**: Add server-side validation for glow fields
3. **Documentation**: Update component documentation to prefer glow fields
4. **Migration Tools**: Provide automated migration tools for large codebases

## Conclusion

This implementation provides seamless bloom/glow field compatibility while maintaining backward compatibility and enabling a smooth transition to server-preferred glow field naming. The solution is robust, well-tested, and designed for long-term maintainability.