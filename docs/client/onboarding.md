# Onboarding System

*[Client](../index.md)*

## Overview
The onboarding system provides guided tours and interactive tutorials to help new users understand the application's features. It supports multi-step flows, element highlighting, and progress tracking.

## Architecture

### Core Components

#### OnboardingProvider (`client/src/features/onboarding/components/OnboardingProvider.tsx`)
React context provider managing onboarding state and flow control.

```typescript
interface OnboardingContextValue {
  startFlow: (flowId: string) => void
  endFlow: () => void
  nextStep: () => void
  previousStep: () => void
  skipFlow: () => void
  isFlowCompleted: (flowId: string) => boolean
  resetFlow: (flowId: string) => void
  currentFlow: OnboardingFlow | null
  currentStep: OnboardingStep | null
  isActive: boolean
}
```

#### OnboardingFlow Interface (`client/src/features/onboarding/types.ts`)
```typescript
interface OnboardingFlow {
  id: string                    // Unique flow identifier
  name: string                  // Display name
  description: string           // Flow description
  steps: OnboardingStep[]       // Ordered steps
  completionKey?: string        // localStorage tracking key
}

interface OnboardingStep {
  id: string                    // Step identifier
  title: string                 // Step title
  description: string           // Step description
  target?: string              // CSS selector for highlighting
  position?: 'top' | 'bottom' | 'left' | 'right' | 'centre'
  action?: () => void | Promise<void>  // Optional action
  skipable?: boolean           // Can user skip this step
  nextButtonText?: string      // Custom next button text
  prevButtonText?: string      // Custom previous button text
}
```

### User Interface

#### OnboardingOverlay (`client/src/features/onboarding/components/OnboardingOverlay.tsx`)
Main UI component rendering the onboarding overlay with:
- Step content display
- Element highlighting
- Navigation controls
- Progress indicator

#### OnboardingEventHandler (`client/src/features/onboarding/components/OnboardingEventHandler.tsx`)
Handles automatic flow triggers based on:
- First-time user detection
- Feature access
- User actions

## Default Flows

### Welcome Tour Flow
```typescript
{
  id: 'welcome',
  name: 'Welcome Tour',
  description: 'Get started with the application',
  steps: [
    {
      id: 'welcome',
      title: 'Welcome to LogSeq Spring Thing!',
      description: 'This interactive tour will help you get familiar with the main features of the application. You can skip this tour at any time or restart it later from the help menu.',
      position: 'centre'
    },
    {
      id: 'graph-view',
      title: 'Graph visualisation',
      description: 'This is your main workspace where you can visualise and interact with your knowledge graph. Use your mouse to pan, zoom, and select nodes.',
      target: 'canvas',
      position: 'right'
    },
    {
      id: 'settings-panel',
      title: 'Settings Panel',
      description: 'Customise your visualisation with various settings. You can adjust colours, node sizes, and many other visual properties.',
      target: '.setting-control',
      position: 'left'
    },
    {
      id: 'command-palette',
      title: 'Command Palette',
      description: 'Press Ctrl+K (or Cmd+K on Mac) to open the command palette. It provides quick access to all available commands and features.',
      position: 'centre',
      action: () => {
        // Demonstrates command palette functionality
      }
    }
  ]
}
```

### XR Mode Introduction
Activated when user first enters XR mode:
- Hand tracking basics
- Gesture controls
- Navigation in VR/AR
- Safety guidelines

### Settings Tour
Comprehensive walkthrough of settings:
- visualisation options
- Performance settings
- AI features configuration
- Keyboard shortcuts

## Usage

### Starting an Onboarding Flow
```typescript
import { useOnboarding } from '@/features/onboarding';

function MyComponent() {
  const { startFlow } = useOnboarding();
  
  const handleNewFeature = () => {
    startFlow('new-feature-intro');
  };
}
```

### Creating Custom Flows
```typescript
const customFlow: OnboardingFlow = {
  id: 'custom-feature',
  name: 'Custom Feature Tour',
  description: 'Learn about our new feature',
  steps: [
    {
      id: 'intro',
      title: 'New Feature',
      description: 'Introducing our latest addition',
      position: 'centre'
    },
    {
      id: 'usage',
      title: 'How to Use',
      description: 'Click here to activate',
      target: '#new-feature-button',
      position: 'bottom',
      action: () => {
        // Highlight or activate feature
      }
    }
  ],
  completionKey: 'custom-feature-completed'
};
```

### Checking Flow Completion
```typescript
const { isFlowCompleted } = useOnboarding();

if (!isFlowCompleted('first-time-user')) {
  // Show prompt to start tour
}
```

## Element Highlighting

### Spotlight Effect
- Dynamic overlay with cutout for target element
- Smooth transitions between steps
- Responsive to element position changes

### Pointer Positioning
- Smart positioning to avoid viewport edges
- Arrow pointing to highlighted element
- Mobile-responsive layouts

## Progress Tracking

### Local Storage
- Completed flows stored in localStorage
- Per-user tracking with Nostr integration
- Reset options for testing

### Analytics Integration
- Step completion events
- Flow abandonment tracking
- Time spent per step

## Customisation

### Styling
```css
/* Custom onboarding theme */
.onboarding-overlay {
  --onboarding-primary: #007bff;
  --onboarding-backdrop: rgba(0, 0, 0, 0.5);
  --onboarding-card-bg: white;
}
```

### Content Formatting
- Markdown support in descriptions
- Image embedding
- Video tutorials
- Interactive elements

## Best Practices

1. **Step Design**: Keep steps focused and concise
2. **Targeting**: Use stable CSS selectors
3. **Flow Length**: Limit to 5-7 steps per flow
4. **Interactivity**: Include hands-on actions
5. **Skip Options**: Always provide exit options

## Accessibility

- Keyboard navigation support
- Screen reader announcements
- High contrast mode
- Reduced motion options

## Integration with Other Systems

### Command Palette
- "Start Tutorial" commands
- Quick access to help flows

### Help System
- Links to detailed documentation
- Context-sensitive help triggers

### Settings
- Onboarding preferences
- Tutorial replay options

## Performance Considerations

- Lazy loading of flow content
- Efficient DOM queries for highlighting
- Debounced resize handlers
- Memory cleanup on flow completion

## Future Enhancements

- Interactive tutorials with validation
- Branching flows based on user choices
- Video-based onboarding
- Collaborative onboarding for teams
- AI-guided personalized tours

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
- [Parallel Graphs Feature](../client/parallel-graphs.md)
- [RGB and Client Side Validation](../archive/legacy/old_markdown/RGB and Client Side Validation.md)
- [Settings Panel](../client/settings-panel.md)
- [State Management](../client/state-management.md)
- [UI Component Library](../client/ui-components.md)
- [User Controls Summary - Settings Panel](../client/user-controls-summary.md)
- [VisionFlow Client Documentation](../client/index.md)
- [WebSocket Communication](../client/websocket.md)
- [WebXR Integration](../client/xr-integration.md)
