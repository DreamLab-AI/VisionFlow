# Control Centre Component

A comprehensive UI control panel for managing visualization and environment settings in the graph visualization application.

## Features

### 🎯 **Separated Control Groups**
- **Background Environment Controls**: Manage visual environment backdrop (background color, opacity, lighting, glow effects)
- **Force Graph Controls**: Configure force-directed graph properties (nodes, edges, physics, labels)

### 🔐 **Nostr Authentication Integration**
- Connect via Nostr browser extensions (nos2x, Alby)
- Server settings persistence when authenticated
- Visual authentication status indicators

### ⚙️ **Settings Management**
- Real-time settings updates with localStorage persistence
- Server synchronization for authenticated users
- Proper case conversion handling (camelCase client ↔ snake_case server)

### 🎨 **Modern UI Components**
- Expandable/collapsible design
- Tabbed interface for organized controls
- Dark theme with glass-morphism effects
- Responsive sliders, color pickers, and toggles

## Component Structure

```
ControlCentre/
├── ControlCentre.tsx           # Main container component
├── BackgroundEnvironmentControls.tsx  # Environment/backdrop settings
├── ForceGraphControls.tsx      # Graph visualization settings
├── index.ts                    # Component exports
└── README.md                   # This documentation
```

## Usage

### Basic Implementation

```tsx
import { ControlCentre } from './components/ControlCentre';

function App() {
  return (
    <div className="relative">
      {/* Your main content */}
      <ControlCentre
        defaultExpanded={true}
        showStats={true}
        enableBloom={false}
      />
    </div>
  );
}
```

### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `className` | `string` | `""` | Additional CSS classes |
| `defaultExpanded` | `boolean` | `true` | Initial expansion state |
| `showStats` | `boolean` | `false` | Display statistics indicators |
| `enableBloom` | `boolean` | `false` | Show bloom effect status |

## Settings Categories

### Background Environment Controls

#### **Background Properties**
- **Background Color**: Color picker with hex input
- **Background Opacity**: 0-100% transparency control

#### **Environment Lighting**
- **Ambient Light**: Global illumination intensity (0-2)
- **Directional Light**: Primary light source intensity (0-2)
- **Environment Intensity**: IBL environment mapping strength (0-2)

#### **Environment Effects**
- **Global Glow Toggle**: Enable/disable glow effects system
- **Environment Glow**: Atmospheric glow intensity (0-10)
- **Global Bloom Toggle**: Enable/disable bloom effects system
- **Environment Bloom**: Background bloom contribution (0-100%)

### Force Graph Controls

#### **Graph Selection**
- Switch between Logseq and VisionFlow graphs
- Context-sensitive controls per graph type

#### **Node Appearance**
- **Node Size**: Scale factor (0.1-5.0)
- **Node Color**: Color picker with hex input
- **Node Opacity**: Transparency control (10-100%)

#### **Edge Properties**
- **Edge Width**: Connection line thickness (0.01-5.0)
- **Edge Color**: Color picker with hex input
- **Edge Opacity**: Transparency control (10-100%)

#### **Force Physics**
- **Physics Toggle**: Enable/disable physics simulation
- **Spring Strength**: Edge attraction force (0.001-2.0)
- **Repulsion Strength**: Node separation force (0.01-10.0)
- **Damping**: Velocity decay factor (0.5-0.99)
- **Reset Physics**: Restore graph-specific defaults

#### **Node Labels**
- **Labels Toggle**: Show/hide node text labels
- **Label Size**: Font scaling (0.1-3.0)

## Integration Points

### Settings Store Integration

The Control Centre integrates deeply with the application's settings store:

```tsx
// Automatic settings persistence
const { settings, updateSettings, updateGPUPhysics } = useSettingsStore();

// Real-time updates with validation
updateGPUPhysics(graphName, { springK: newValue });
```

### Nostr Authentication

Authentication flow for server settings persistence:

```tsx
// Check for Nostr extension
if (!window.nostr) {
  // Show install prompt
}

// Authenticate and enable server sync
await nostrAuth.login();
updateSettings((draft) => {
  draft.system.persistSettings = true;
});
```

### WebSocket Integration

Physics parameter updates are broadcast in real-time:

```tsx
// Automatic WebSocket notifications for physics changes
const message = {
  type: 'physics_parameter_update',
  graph: graphName,
  parameters: updatedParams
};
```

## Styling

The component uses a dark theme with glass-morphism effects:

- **Background**: `bg-black/90` with `backdrop-blur-sm`
- **Borders**: `border-white/20` for subtle contrast
- **Cards**: `bg-white/5` with `border-white/10`
- **Text**: White with varying opacity levels
- **Controls**: Consistent theming across all input elements

## State Management

### Local State
- `isExpanded`: Panel expansion state
- `activeTab`: Current tab selection ('background' | 'graph')
- `activeGraph`: Selected graph type ('logseq' | 'visionflow')

### Global State (via useSettingsStore)
- `settings`: Complete settings object
- `authenticated`: Nostr authentication status
- `user`: Current user information
- `isPowerUser`: Power user privileges

## Error Handling

- Toast notifications for successful/failed operations
- Graceful fallbacks for missing settings
- Proper error boundaries for authentication failures

## Performance Considerations

- Debounced settings updates (500ms delay for server sync)
- Efficient re-renders with useCallback hooks
- Minimal re-computations with proper dependency arrays
- Settings validation and range clamping

## Browser Compatibility

- **Modern Browsers**: Full feature support
- **Nostr Extensions**: Requires nos2x, Alby, or compatible NIP-07 extension
- **WebHID**: For advanced hardware controller support (SpacePilot)

## Future Enhancements

- [ ] Keyboard shortcuts for common operations
- [ ] Settings presets and profiles
- [ ] Advanced physics parameter visualization
- [ ] Drag-and-drop settings import/export
- [ ] Touch-optimized mobile interface
- [ ] Accessibility improvements (ARIA labels, keyboard navigation)