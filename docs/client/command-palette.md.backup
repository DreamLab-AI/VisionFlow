# Command Palette

*[Client](../index.md)*

## Overview
The Command Palette provides a keyboard-driven interface for quickly accessing application features and executing commands. Inspired by VS Code and similar modern applications, it offers fuzzy search, keyboard shortcuts, and command categorization.

## Architecture

### Core Components

#### CommandRegistry (`client/src/features/command-palette/CommandRegistry.ts`)
Central registry managing all available commands in the application.

```typescript
class CommandRegistry {
  register(command: Command): void
  unregister(commandId: string): void
  getCommand(commandId: string): Command | undefined
  getCommandsByCategory(categoryId: string): Command[]
  searchCommands(query: string): Command[]
  executeCommand(commandId: string): Promise<void>
}
```

#### Command Interface (`client/src/features/command-palette/types.ts`)
```typescript
interface Command {
  id: string                    // Unique identifier
  title: string                 // Display name
  description?: string          // Optional description
  category: string              // Category for grouping
  keywords?: string[]           // Additional search terms
  icon?: React.ComponentType    // Optional icon component
  shortcut?: {                  // Keyboard shortcut
    key: string
    ctrl?: boolean
    alt?: boolean
    shift?: boolean
    meta?: boolean
  }
  handler: () => void | Promise<void>  // Command execution
  enabled?: boolean             // Dynamic enable/disable
}
```

### User Interface

#### CommandPalette Component (`client/src/features/command-palette/components/CommandPalette.tsx`)
Main UI component rendering the command palette overlay.

Key Features:
- Fuzzy search with highlighting
- Keyboard navigation (↑/↓ arrows)
- Category grouping
- Recent commands section
- Keyboard shortcut display

#### useCommandPalette Hook (`client/src/features/command-palette/hooks/useCommandPalette.ts`)
React hook providing command palette functionality:

```typescript
const {
  isOpen,
  openCommandPalette,
  closeCommandPalette,
  toggleCommandPalette,
  registerCommand,
  executeCommand,
  searchQuery,
  setSearchQuery,
  filteredCommands
} = useCommandPalette();
```

## Default Commands

### Navigation Commands
- **Go to Settings**: Opens settings panel
- **Go to Home**: Navigates to main view
- **Toggle Full Screen**: Enters/exits full screen mode

### View Commands
- **Toggle XR Mode**: Switches to WebXR view
- **Reset Camera**: Resets 3D camera position
- **centre Graph**: centres the graph visualisation

### System Commands
- **Refresh Data**: Reloads graph data
- **Clear Cache**: Clears local storage
- **Show Debug Info**: Displays debug panel

## Usage

### Opening the Command Palette
- Default shortcut: `Ctrl/Cmd + K`
- Programmatically: `openCommandPalette()`

### Registering Custom Commands
```typescript
import { useCommandPalette } from '@/features/command-palette';

function MyComponent() {
  const { registerCommand } = useCommandPalette();
  
  useEffect(() => {
    registerCommand({
      id: 'my-custom-command',
      title: 'My Custom Command',
      category: 'custom',
      keywords: ['custom', 'example'],
      shortcut: { key: 'M', ctrl: true },
      handler: async () => {
        // Command logic here
      }
    });
  }, []);
}
```

### Keyboard Shortcuts
- `Esc`: Close command palette
- `Enter`: Execute selected command
- `↑/↓`: Navigate through commands
- `Tab`: Focus search input

## Integration Points

### Default Commands

#### Navigation Commands
- `nav.settings` - Open Settings (Settings icon)
- `nav.help` - Show Help (HelpCircle icon)

#### Help Commands  
- `help.search` - Search Help Topics (Search icon)
- `help.keyboard` - Show Keyboard Shortcuts (`Shift+?`, Terminal icon)
- `help.tour` - Start Tutorial Tour (HelpCircle icon)

#### Settings Commands
- `settings.undo` - Undo Settings Change (`Ctrl+Z`, Undo2 icon)
- `settings.redo` - Redo Settings Change (`Ctrl+Shift+Z`, Redo2 icon)
- `settings.save` - Save Settings to Server (Save icon)
- `settings.reset` - Reset to Defaults (RotateCcw icon)
- `settings.export` - Export Settings (Download icon)
- `settings.import` - Import Settings (Upload icon)

#### View Commands
- `view.fullscreen` - Toggle Fullscreen (Maximize2 icon)
- `view.theme.toggle` - Toggle Light/Dark Theme (Moon/Sun icon)

#### System Commands
- `system.refresh` - Refresh Application (`F5`, RefreshCw icon)

Each command includes:
- **Keywords**: Alternative search terms for discovery
- **Icons**: Visual identifiers from Lucide React
- **Shortcuts**: Optional keyboard shortcuts
- **Categories**: Organised grouping for better navigation

## Performance Considerations

### Fuzzy Search optimisation
- Uses efficient string matching algorithm
- Caches search results
- Debounces search input (150ms)

### Command Execution
- Async command handlers prevent UI blocking
- Loading states for long-running commands
- Error boundaries for command failures

## Customisation

### Theming
Command palette respects application theme:
- Dark/light mode support
- Custom colour schemes
- Font size adjustments

### Configuration Options
```typescript
interface CommandRegistryOptions {
  maxRecentCommands?: number      // Default: 5
  fuzzySearchThreshold?: number   // Default: 0.3
}
```

## Best Practices

1. **Command Naming**: Use clear, action-oriented titles
2. **Categories**: Group related commands logically
3. **Keywords**: Add relevant search terms for discoverability
4. **Shortcuts**: Avoid conflicts with browser/OS shortcuts
5. **Handlers**: Keep command logic focused and fast

## Future Enhancements

- Command chaining/macros
- Context-aware commands
- Command history with undo
- Plugin system for external commands
- Voice command integration

## Related Topics

- [Client Architecture](../client/architecture.md)
- [Client Core Utilities and Hooks](../client/core.md)
- [Client Rendering System](../client/rendering.md)
- [Client TypeScript Types](../client/types.md)
- [Client side DCO](../archive/legacy/old_markdown/Client side DCO.md)
- [Client-Side visualisation Concepts](../client/visualization.md)
- [GPU-Accelerated Analytics](../client/features/gpu-analytics.md)
- [Graph System](../client/graph-system.md)
- [Help System](../client/help-system.md)
- [Onboarding System](../client/onboarding.md)
- [Parallel Graphs Feature](../client/parallel-graphs.md)
- [RGB and Client Side Validation](../archive/legacy/old_markdown/RGB and Client Side Validation.md)
- [Settings Panel](../client/settings-panel.md)
- [State Management](../client/state-management.md)
- [UI Component Library](../client/ui-components.md)
- [User Controls Summary - Settings Panel](../client/user-controls-summary.md)
- [VisionFlow Client Documentation](../client/index.md)
- [WebSocket Communication](../client/websocket.md)
- [WebXR Integration](../client/xr-integration.md)
