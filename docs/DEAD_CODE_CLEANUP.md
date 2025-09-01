# Frontend Dead Code Cleanup - September 1, 2025

## Overview
Cleaned up frontend-only settings that have no corresponding backend support in the EdgeSettings and LabelSettings types.

## Removed Settings

### Edge Flow Effects (Graph-Specific Paths)
The following settings were removed from `settingsUIDefinition.ts` because they don't exist in the backend `EdgeSettings` type:

- `visualisation.graphs.{graphName}.edges.enableFlowEffect` - Toggle for animated flow effect along edges
- `visualisation.graphs.{graphName}.edges.flowSpeed` - Speed of the flow effect (0.1-5.0)
- `visualisation.graphs.{graphName}.edges.flowIntensity` - Intensity of the flow effect (0-10)
- `visualisation.graphs.{graphName}.edges.glowStrength` - Strength of the edge glow effect (0-5)
- `visualisation.graphs.{graphName}.edges.distanceIntensity` - Intensity based on distance (0-10)
- `visualisation.graphs.{graphName}.edges.useGradient` - Use gradient for edge colors
- `visualisation.graphs.{graphName}.edges.gradientColors` - Start and end colors for edge gradient

### Label Settings
- `visualisation.graphs.{graphName}.labels.labelDistance` - Distance of label from node center (0-5.0)

## Files Modified

### 1. `client/src/features/settings/config/settingsUIDefinition.ts`
- Removed dead edge flow effect settings with explanatory comments
- Removed dead labelDistance setting with explanatory comment

### 2. `client/src/features/graph/components/VisualEffectsPanel.tsx`
- Removed `flowEffectEnabled` hook reference
- Removed entire "Flow Effects" UI section
- Updated master toggle to remove edge flow effects
- Updated preset buttons to remove flow effect settings
- Updated effect detection logic

### 3. `client/src/features/graph/components/FlowingEdges.tsx`
- Removed unused flow effect shader code (was not being used with LineBasicMaterial)
- Removed animation loop that tried to access non-existent flow settings
- Added explanatory comments about removed functionality

## Backend EdgeSettings Support
According to `client/src/types/generated/settings.ts`, the backend EdgeSettings only supports:
```typescript
interface EdgeSettings {
  arrowSize: number;
  baseWidth: number;
  color: string;
  enableArrows: boolean;
  opacity: number;
  widthRange: number[];
  quality: string;
}
```

The backend LabelSettings only supports:
```typescript
interface LabelSettings {
  desktopFontSize: number;
  enableLabels: boolean;
  textColor: string;
  textOutlineColor: string;
  textOutlineWidth: number;
  textResolution: number;
  textPadding: number;
  billboardMode: string;
  show_metadata?: boolean;
  max_label_width?: number;
}
```

## Important Notes

### Flow Effects Implementation
The `FlowingEdges` component had implemented visual flow effects using shaders, but:
1. The backend doesn't support storing these settings
2. The UI was trying to access non-existent settings paths
3. The shader code was not actually being used (component uses LineBasicMaterial)

### Future Implementation
If flow effects are desired in the future, the backend would need to:
1. Add flow effect fields to the EdgeSettings struct
2. Update the settings API to support these new fields
3. Regenerate the TypeScript types

### Tests
No test files needed modification as they don't reference the removed settings paths.

## Impact
- Removed non-functional UI controls
- Cleaned up dead code in components
- No breaking changes to existing functionality
- Settings store will no longer have phantom entries for these paths