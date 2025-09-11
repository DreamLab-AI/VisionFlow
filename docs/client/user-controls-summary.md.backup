# User Controls Summary - Settings Panel

*[Client](../index.md)*

⚠️ **CURRENT STATUS**: Controls may not respond correctly due to dual store issue. See settings-migration.md for details.

## Intuitive Features Implemented

### 1. **Smart Control Type Selection** (✅ IMPLEMENTED)
The system automatically selects the most appropriate control type based on the `controlType` specified in the `UISettingDefinition` (from [`client/src/features/settings/config/settingsUIDefinition.ts`](../../client/src/features/settings/config/settingsUIDefinition.ts)) and the data type of the setting. Key control types rendered by [`SettingControlComponent.tsx`](../../client/src/features/settings/components/SettingControlComponent.tsx) include:

-   **`toggle`**: For boolean values (on/off settings) -> Renders a Switch.
-   **`slider`**: For numeric values with defined `min`, `max`, and `step` -> Renders a Slider.
-   **`numberInput`**: For general numeric values -> Renders a Number Input.
-   **`textInput`**: For string values.
-   **`passwordInput`**: A variant of `textInput` for sensitive string fields (API keys, secrets), providing masking.
-   **`select`**: For predefined options (enum-like strings) defined in `options` array in `UISettingDefinition` -> Renders a Select Dropdown.
-   **`colourPicker`**: For single string colour values -> Renders a colour Picker with hex input.
-   **`rangeSlider`**: For `[number, number]` array values, representing a min/max range -> Renders a specialised Range Slider.
-   **`dualcolourPicker`**: For `[string, string]` array values, representing two colours (e.g., for gradients) -> Renders two colour Pickers.
-   **`radioGroup`**: For selecting one option from a list (mutually exclusive choices) defined in `options` -> Renders a Radio Group.
-   **`buttonAction`**: For triggering an action (e.g., reset a section, trigger a backend process) -> Renders a Button. The action is defined by `actionId` in `UISettingDefinition`.

### 2. **User Experience Enhancements**

#### Visual Feedback
- **Live Value Display** - Shows current value next to sliders with appropriate decimal places
- **Unit Display** - Shows units (px, ms, etc.) where applicable
- **Hover Effects** - Subtle background highlight on hover for better interactivity
- **Tooltips** - Info icons with descriptions for complex settings

#### Input Handling
- **Debounced Inputs** - 300ms delay prevents excessive updates while typing
- **Validation** - colour inputs validate hex format and auto-correct invalid entries
- **Password Visibility Toggle** - Eye icon to show/hide sensitive values
- **Placeholder Text** - Contextual hints for input fields

#### Layout & Styling
- **Responsive Design** - Controls adapt to available space
- **Consistent Spacing** - Proper padding and margins for readability
- **Visual Hierarchy** - Clear label/control separation
- **Smooth Transitions** - CSS transitions for hover states

### 3. **Task-Appropriate Features**

#### For visualisation Settings
- **Real-time Updates** - Changes to visualisation settings update the viewport immediately
- **Slider Preference** - Numeric inputs with ranges automatically use sliders for easier adjustment
- **Precise Control** - Step values configured appropriately (0.01 for decimals, 1 for integers)

#### For Security/Authentication
- **Automatic Masking** - API keys, secrets, and tokens are masked by default
- **Secure Placeholders** - "Enter secure value" for sensitive fields
- **Power User Gating** - Advanced settings only visible to authenticated power users

#### For colour Settings
- **Dual Input** - Both visual picker and text input for flexibility
- **Validation** - Ensures only valid hex colours are saved
- **Fallback Values** - Defaults to black (#000000) if invalid

### 4. **Accessibility Features**
- **Proper Labels** - All controls have associated labels
- **Keyboard Navigation** - Full keyboard support for all controls
- **ARIA Attributes** - Proper IDs and relationships
- **Focus Indicators** - Clear focus states for keyboard users

## Control Types by Use Case

### Basic Settings
- Enable/Disable features → **Toggle Switch**
- Adjust sizes/distances → **Slider with value display**
- Enter text/names → **Text Input with placeholder**

### Advanced Settings
- API Configuration → **Password Input with visibility toggle**
- colour Themes → **colour Picker with hex validation**
- Performance Ranges → **Range Slider for min/max**
- Display Modes → **Select Dropdown**

### Power User Settings
- Debug Options → **Hidden unless authenticated**
- Advanced XR Settings → **Gated by Nostr auth**
- AI Model Parameters → **Only visible to power users**

## Implementation Details

The controls are implemented in [`SettingControlComponent.tsx`](../../client/src/features/settings/components/SettingControlComponent.tsx) with:
- React hooks for state management (getting/setting values via `useSettingsStore`)
- ⚠️ **CRITICAL**: Must use correct store import: `/store/settingsStore` not `/features/settings/store/settingsStore`
- Logic to determine appropriate UI control based on `UISettingDefinition`
- Custom debounce hook for input optimisation (300ms delay)
- TypeScript for type safety with proper Settings interface
- Tailwind CSS for consistent styling
- Lucide React icons for visual elements (tooltips, password visibility)

**Working Store Import**:
```typescript
// ✅ CORRECT
import { useSettingsStore } from '../../../store/settingsStore';

// ❌ WRONG (breaks functionality)
import { useSettingsStore } from '../store/settingsStore';
```

**Multi-Graph Settings Support**:
```typescript
// Access graph-specific settings
const logseqNodecolour = useSettingsStore(
  state => state.settings?.visualisation?.graphs?.logseq?.nodes?.basecolour
);

// Update graph-specific settings
setValue(newValue) {
  updateSettings(draft => {
    draft.visualisation.graphs.logseq.nodes.basecolour = newValue;
  });
}
```

All controls follow the same pattern:
1. Receive value and onChange from parent (via settings store)
2. Manage local state for debouncing if needed (300ms for text/number inputs)
3. Validate input before calling onChange (especially for colours, numbers)
4. Provide appropriate visual feedback (hover, focus, validation states)
5. ✅ **Real-time updates**: Changes immediately affect visualisation via viewport settings
6. ✅ **Auto-save**: Changes are automatically persisted to server after debounce

**Control Responsiveness**:
- ✅ Sliders: Live value updates with immediate visual feedback
- ✅ colour pickers: Real-time colour changes in 3D scene  
- ✅ Toggles: Instant enable/disable of features
- ⚠️ **IF BROKEN**: Check store import in component files

## Related Topics

- [Client Architecture](../client/architecture.md)
- [Client Core Utilities and Hooks](../client/core.md)
- [Client Rendering System](../client/rendering.md)
- [Client TypeScript Types](../client/types.md)
- [Client side DCO](../archive/legacy/old_markdown/Client side DCO.md)
- [Client-Side visualisation Concepts](../client/visualization.md)
- [Command Palette](../client/command-palette.md)
- [GPU-Accelerated Analytics](../client/features/gpu-analytics.md)
- [Graph System](../client/graph-system.md)
- [Help System](../client/help-system.md)
- [Onboarding System](../client/onboarding.md)
- [Parallel Graphs Feature](../client/parallel-graphs.md)
- [RGB and Client Side Validation](../archive/legacy/old_markdown/RGB and Client Side Validation.md)
- [Settings Panel](../client/settings-panel.md)
- [State Management](../client/state-management.md)
- [UI Component Library](../client/ui-components.md)
- [VisionFlow Client Documentation](../client/index.md)
- [WebSocket Communication](../client/websocket.md)
- [WebXR Integration](../client/xr-integration.md)
