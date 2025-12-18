---
title: Client Components Reference
description: **VisionFlow Client Component Library** **Version:** 0.1.0 **Last Updated:** 2025-11-04
category: explanation
tags:
  - client
  - rest
  - websocket
  - rust
  - react
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Client Components Reference

**VisionFlow Client Component Library**
**Version:** 0.1.0
**Last Updated:** 2025-11-04

---

## Table of Contents

1. [Overview](#overview)
2. [Design System](#design-system)
3. [Graph Components](#graph-components)
4. [Visualization Components](#visualization-components)
5. [Bots Components](#bots-components)
6. [Settings Components](#settings-components)
7. 
8. [UI Components](#ui-components)
9. [Layout Components](#layout-components)
10. [Feature Components](#feature-components)

---

## Overview

This document provides a comprehensive reference for all React components in the VisionFlow client application. Each component is documented with:

- **Purpose**: What the component does
- **Props**: Input properties with types
- **Events**: Custom events emitted
- **Usage Examples**: Code examples
- **Styling**: CSS classes and customization
- **Accessibility**: ARIA labels and keyboard support
- **Performance**: Optimization notes

### Component Categories

- **Design System**: Reusable UI primitives (Button, Input, Modal)
- **Graph**: Graph visualization components
- **Visualization**: 3D rendering and effects
- **Bots**: Multi-agent system UI
- **Settings**: Configuration and preferences
- **XR**: WebXR and immersive experiences
- **Layout**: Page structure and navigation
- **Feature**: Domain-specific components

---

## Design System

### Button Component

**File**: `/src/features/design-system/components/Button.tsx`

#### Purpose
Versatile button component with multiple variants, sizes, and states.

#### Props

```typescript
interface ButtonProps {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  children: React.ReactNode;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
}
```

#### Usage Examples

```tsx
import { Button } from '@/features/design-system/components/Button';

// Basic button
<Button onClick={handleClick}>
  Click Me
</Button>

// With icon
<Button icon={<SaveIcon />} iconPosition="left">
  Save
</Button>

// Loading state
<Button loading={isSubmitting}>
  Submit
</Button>

// Variants
<Button variant="destructive">Delete</Button>
<Button variant="outline">Cancel</Button>
<Button variant="ghost">Reset</Button>

// Sizes
<Button size="sm">Small</Button>
<Button size="lg">Large</Button>
<Button size="icon"><IconComponent /></Button>
```

#### Styling

```typescript
const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline"
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10"
      }
    },
    defaultVariants: {
      variant: "default",
      size: "default"
    }
  }
);
```

#### Accessibility

- **Keyboard**: Spacebar and Enter activate
- **Focus**: Visible focus ring
- **ARIA**: `aria-disabled`, `aria-busy` for loading state
- **Screen Reader**: Button text announced

---

### Input Component

**File**: `/src/features/design-system/components/Input.tsx`

#### Purpose
Text input field with validation and error states.

#### Props

```typescript
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
}
```

#### Usage Examples

```tsx
import { Input } from '@/features/design-system/components/Input';

// Basic input
<Input
  label="Name"
  placeholder="Enter your name"
  value={name}
  onChange={(e) => setName(e.target.value)}
/>

// With validation error
<Input
  label="Email"
  type="email"
  value={email}
  error={emailError}
  onChange={handleEmailChange}
/>

// With icons
<Input
  label="Search"
  leftIcon={<SearchIcon />}
  placeholder="Search nodes..."
/>

// With helper text
<Input
  label="API Key"
  type="password"
  helperText="Your API key is encrypted"
/>
```

#### Accessibility

- **Labels**: Associated with `<label>` element
- **Errors**: `aria-invalid`, `aria-describedby` for error messages
- **Required**: `aria-required` attribute
- **Placeholder**: Not used as label replacement

---

### Modal Component

**File**: `/src/features/design-system/components/Modal.tsx`

#### Purpose
Dialog/modal overlay for focused interactions.

#### Props

```typescript
interface ModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
}
```

#### Usage Examples

```tsx
import { Modal } from '@/features/design-system/components/Modal';

const [isOpen, setIsOpen] = useState(false);

<Modal
  open={isOpen}
  onOpenChange={setIsOpen}
  title="Confirm Deletion"
  description="Are you sure you want to delete this node?"
  footer={
    <>
      <Button variant="ghost" onClick={() => setIsOpen(false)}>
        Cancel
      </Button>
      <Button variant="destructive" onClick={handleDelete}>
        Delete
      </Button>
    </>
  }
>
  <p>This action cannot be undone.</p>
</Modal>
```

#### Accessibility

- **Focus Trap**: Focus locked within modal
- **ESC Key**: Closes modal (if enabled)
- **ARIA**: `role="dialog"`, `aria-modal="true"`
- **Screen Reader**: Title and description announced
- **Focus Return**: Returns focus to trigger element

---

### Select Component

**File**: `/src/features/design-system/components/Select.tsx`

#### Purpose
Dropdown select menu with search and custom options.

#### Props

```typescript
interface SelectProps<T> {
  value: T;
  onChange: (value: T) => void;
  options: SelectOption<T>[];
  label?: string;
  placeholder?: string;
  searchable?: boolean;
  disabled?: boolean;
  error?: string;
}

interface SelectOption<T> {
  label: string;
  value: T;
  icon?: React.ReactNode;
  disabled?: boolean;
  description?: string;
}
```

#### Usage Examples

```tsx
import { Select } from '@/features/design-system/components/Select';

const graphTypes = [
  { label: 'Knowledge Graph', value: 'knowledge_graph', icon: <GraphIcon /> },
  { label: 'Ontology', value: 'ontology', icon: <OntologyIcon /> }
];

<Select
  label="Graph Type"
  value={graphType}
  onChange={setGraphType}
  options={graphTypes}
  searchable
/>
```

---

### Slider Component

**File**: `/src/features/design-system/components/Slider.tsx`

#### Purpose
Range slider for numeric value input.

#### Props

```typescript
interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  formatValue?: (value: number) => string;
  disabled?: boolean;
}
```

#### Usage Examples

```tsx
import { Slider } from '@/features/design-system/components/Slider';

<Slider
  label="Physics Damping"
  value={damping}
  onChange={setDamping}
  min={0}
  max={1}
  step={0.01}
  showValue
  formatValue={(val) => `${(val * 100).toFixed(0)}%`}
/>
```

---

### Switch Component

**File**: `/src/features/design-system/components/Switch.tsx`

#### Purpose
Toggle switch for boolean settings.

#### Props

```typescript
interface SwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
}
```

#### Usage Examples

```tsx
import { Switch } from '@/features/design-system/components/Switch';

<Switch
  label="Enable Physics"
  description="Toggle graph physics simulation"
  checked={physicsEnabled}
  onCheckedChange={setPhysicsEnabled}
/>
```

---

### Toast Component

**File**: `/src/features/design-system/components/Toast.tsx`

#### Purpose
Notification toast messages.

#### Usage

```typescript
import { toast } from '@/features/design-system/components/Toast';

// Success
toast.success('Settings saved successfully');

// Error
toast.error('Failed to connect to server');

// Warning
toast.warning('Connection unstable');

// Info
toast.info('New update available');

// Custom
toast.custom({
  title: 'Agent Connected',
  description: 'Bot-001 joined the network',
  duration: 5000
});
```

---

### Loading Components

#### LoadingSpinner

```tsx
import { LoadingSpinner } from '@/features/design-system/components/LoadingSpinner';

<LoadingSpinner size="lg" />
```

#### LoadingSkeleton

```tsx
import { LoadingSkeleton } from '@/features/design-system/components/LoadingSkeleton';

<LoadingSkeleton type="text" lines={3} />
<LoadingSkeleton type="card" />
<LoadingSkeleton type="graph" />
```

---

## Graph Components

### GraphCanvas

**File**: `/src/features/graph/components/GraphCanvas.tsx`

#### Purpose
Main 3D canvas for graph visualization using React Three Fiber.

#### Props

```typescript
interface GraphCanvasProps {
  graphData?: GraphData;
  onNodeClick?: (nodeId: number) => void;
  onEdgeClick?: (edgeId: number) => void;
  selectedNodeId?: number | null;
  cameraPosition?: [number, number, number];
}
```

#### Usage

```tsx
import GraphCanvas from '@/features/graph/components/GraphCanvas';

<GraphCanvas
  graphData={graphData}
  onNodeClick={handleNodeClick}
  selectedNodeId={selectedNode}
  cameraPosition={[20, 15, 20]}
/>
```

#### Features

- **3D Rendering**: Three.js via React Three Fiber
- **Orbit Controls**: Mouse/touch camera control
- **Post-Processing**: Bloom, SSAO effects
- **Performance**: Instancing, LOD, frustum culling
- **Physics**: GPU-accelerated force-directed layout

---

### GraphManager

**File**: `/src/features/graph/components/GraphManager.tsx`

#### Purpose
Manages graph node and edge rendering inside Three.js scene.

#### Props

```typescript
interface GraphManagerProps {
  graphData: GraphData;
}
```

#### Features

- **Node Rendering**: Sphere meshes with materials
- **Edge Rendering**: Line geometries
- **Animation**: Smooth position interpolation
- **Selection**: Highlight selected nodes/edges
- **LOD**: Level-of-detail optimization

---

### HierarchicalGraphRenderer

**File**: `/src/features/graph/components/HierarchicalGraphRenderer.tsx`

#### Purpose
Specialized renderer for hierarchical/tree graphs.

#### Props

```typescript
interface HierarchicalGraphRendererProps {
  graphData: GraphData;
  layout: 'tree' | 'radial' | 'force-tree';
  orientation?: 'horizontal' | 'vertical';
  levelSeparation?: number;
  nodeSeparation?: number;
}
```

#### Usage

```tsx
<HierarchicalGraphRenderer
  graphData={ontologyData}
  layout="tree"
  orientation="vertical"
  levelSeparation={100}
/>
```

---

### NodeShaderToggle

**File**: `/src/features/graph/components/NodeShaderToggle.tsx`

#### Purpose
Toggle between different node rendering shaders.

#### Props

```typescript
interface NodeShaderToggleProps {
  currentShader: 'standard' | 'hologram' | 'wireframe' | 'point-cloud';
  onShaderChange: (shader: string) => void;
}
```

---

### MetadataShapes

**File**: `/src/features/graph/components/MetadataShapes.tsx`

#### Purpose
Render nodes with different shapes based on metadata.

#### Props

```typescript
interface MetadataShapesProps {
  nodes: Node[];
  shapeMapping: Record<string, 'sphere' | 'cube' | 'cone' | 'cylinder'>;
}
```

#### Usage

```tsx
<MetadataShapes
  nodes={graphData.nodes}
  shapeMapping={{
    'person': 'sphere',
    'organization': 'cube',
    'location': 'cone'
  }}
/>
```

---

### FlowingEdges

**File**: `/src/features/graph/components/FlowingEdges.tsx`

#### Purpose
Animated flowing particles along edges.

#### Props

```typescript
interface FlowingEdgesProps {
  edges: Edge[];
  flowSpeed?: number;
  particleCount?: number;
  particleColor?: string;
}
```

---

### GraphViewport

**File**: `/src/features/graph/components/GraphViewport.tsx`

#### Purpose
Viewport container with camera controls and minimap.

#### Props

```typescript
interface GraphViewportProps {
  children: React.ReactNode;
  showMinimap?: boolean;
  showGrid?: boolean;
  showAxes?: boolean;
}
```

---

## Visualization Components

### IntegratedControlPanel

**File**: `/src/features/visualisation/components/IntegratedControlPanel.tsx`

#### Purpose
Main control panel for all visualization settings.

#### Features

- **Physics Controls**: Spring force, repulsion, damping
- **Rendering Settings**: Lighting, shadows, post-processing
- **Graph Selection**: Switch between knowledge graph and ontology
- **Quality Presets**: Low, Medium, High, Ultra
- **Collapsible Sections**: Organized by category

#### Usage

```tsx
import { IntegratedControlPanel } from '@/features/visualisation/components/IntegratedControlPanel';

<IntegratedControlPanel />
```

---

### SpacePilotButtonPanel

**File**: `/src/features/visualisation/components/SpacePilotButtonPanel.tsx`

#### Purpose
Controls for 3Dconnexion SpaceMouse/SpacePilot integration.

#### Props

```typescript
interface SpacePilotButtonPanelProps {
  connected: boolean;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  sensitivity?: number;
  onSensitivityChange?: (value: number) => void;
}
```

---

### HolographicDataSphere

**File**: `/src/features/visualisation/components/HolographicDataSphere.tsx`

#### Purpose
Holographic sphere background with animated rings.

#### Props

```typescript
interface HolographicDataSphereProps {
  opacity?: number;
  ringCount?: number;
  ringColor?: string;
  rotationSpeed?: number;
  layer?: number;
}
```

#### Usage

```tsx
<HolographicDataSphere
  opacity={0.1}
  ringCount={5}
  ringColor="#00ffff"
  rotationSpeed={0.5}
  layer={2}
/>
```

---

### MetadataVisualizer

**File**: `/src/features/visualisation/components/MetadataVisualizer.tsx`

#### Purpose
Visualize node metadata as tooltips and info panels.

#### Props

```typescript
interface MetadataVisualizerProps {
  node: Node | null;
  position?: { x: number; y: number };
  showFullMetadata?: boolean;
}
```

---

### CameraController

**File**: `/src/features/visualisation/components/CameraController.tsx`

#### Purpose
Advanced camera control with presets and animations.

#### Props

```typescript
interface CameraControllerProps {
  presets?: CameraPreset[];
  onPresetSelect?: (preset: CameraPreset) => void;
  animationDuration?: number;
}

interface CameraPreset {
  name: string;
  position: [number, number, number];
  target: [number, number, number];
  fov?: number;
}
```

---

### HeadTrackedParallaxController

**File**: `/src/features/visualisation/components/HeadTrackedParallaxController.tsx`

#### Purpose
MediaPipe face tracking for parallax effect.

#### Props

```typescript
interface HeadTrackedParallaxControllerProps {
  enabled?: boolean;
  sensitivity?: number;
  smoothing?: number;
}
```

#### Features

- **Face Detection**: MediaPipe FaceLandmarker
- **Head Position**: Tracks x, y, z position
- **Camera Offset**: Adjusts camera based on head movement
- **Smoothing**: Interpolated movement

---

### AgentNodesLayer

**File**: `/src/features/visualisation/components/AgentNodesLayer.tsx`

#### Purpose
Render agent nodes with special styling.

#### Props

```typescript
interface AgentNodesLayerProps {
  agents: Agent[];
  highlightConnected?: boolean;
  showTrails?: boolean;
}
```

---

## Bots Components

### BotsVisualization

**File**: `/src/features/bots/components/BotsVisualizationFixed.tsx`

#### Purpose
Real-time 3D visualization of autonomous bots.

#### Features

- **Agent Meshes**: Colored spheres for each agent
- **Position Updates**: Binary WebSocket protocol
- **Labels**: Agent names and status
- **Trails**: Motion trails (optional)
- **Selection**: Click to select agent

#### Usage

```tsx
import { BotsVisualization } from '@/features/bots/components';

<Canvas>
  <BotsVisualization />
</Canvas>
```

---

### BotsControlPanel

**File**: `/src/features/bots/components/BotsControlPanel.tsx`

#### Purpose
Control panel for multi-agent system.

#### Features

- **Agent List**: All connected agents
- **Status Indicators**: Online, busy, idle
- **Actions**: Start, stop, restart agents
- **Telemetry**: Real-time metrics
- **Filters**: Filter by status, type

#### Props

```typescript
interface BotsControlPanelProps {
  agents: Agent[];
  selectedAgent: Agent | null;
  onSelectAgent: (agentId: string) => void;
  onStartAgent: (agentId: string) => void;
  onStopAgent: (agentId: string) => void;
}
```

---

### AgentDetailPanel

**File**: `/src/features/bots/components/AgentDetailPanel.tsx`

#### Purpose
Detailed view of selected agent.

#### Features

- **Agent Info**: Name, type, status
- **Metrics**: CPU, memory, tasks completed
- **Activity Log**: Recent actions
- **Configuration**: Agent-specific settings

#### Props

```typescript
interface AgentDetailPanelProps {
  agent: Agent | null;
  onClose: () => void;
}
```

---

### ActivityLogPanel

**File**: `/src/features/bots/components/ActivityLogPanel.tsx`

#### Purpose
Scrolling log of agent activities.

#### Features

- **Real-time Updates**: New entries appear instantly
- **Filtering**: By agent, action type, severity
- **Search**: Full-text search
- **Export**: Export logs as JSON/CSV

#### Props

```typescript
interface ActivityLogPanelProps {
  entries: ActivityLogEntry[];
  filters?: ActivityLogFilter;
  onFilterChange?: (filters: ActivityLogFilter) => void;
  maxEntries?: number;
}

interface ActivityLogEntry {
  id: string;
  timestamp: number;
  agentId: string;
  agentName: string;
  action: string;
  details?: string;
  severity: 'info' | 'warning' | 'error';
}
```

---

### SystemHealthPanel

**File**: `/src/features/bots/components/SystemHealthPanel.tsx`

#### Purpose
Overall system health for multi-agent swarm.

#### Features

- **Health Metrics**: CPU, memory, network
- **Agent Status**: Online/offline count
- **Alerts**: System warnings and errors
- **Performance Graph**: Real-time charts

---

### AgentTelemetryStream

**File**: `/src/features/bots/components/AgentTelemetryStream.tsx`

#### Purpose
Real-time telemetry visualization.

#### Props

```typescript
interface AgentTelemetryStreamProps {
  agentId: string;
  metrics: string[]; // ['cpu', 'memory', 'tasks']
  refreshInterval?: number;
}
```

---

### MultiAgentInitializationPrompt

**File**: `/src/features/bots/components/MultiAgentInitializationPrompt.tsx`

#### Purpose
Dialog for initializing multiple agents at once.

#### Props

```typescript
interface MultiAgentInitializationPromptProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (config: AgentConfig[]) => void;
}

interface AgentConfig {
  type: string;
  count: number;
  capabilities: string[];
}
```

---

## Settings Components

### FloatingSettingsPanel

**File**: `/src/features/settings/components/FloatingSettingsPanel.tsx`

#### Purpose
Draggable, resizable settings panel.

#### Features

- **Drag and Drop**: Repositionable
- **Resize**: Adjustable width/height
- **Collapse**: Minimize to icon
- **Tabs**: Organized sections
- **Search**: Find settings quickly

#### Props

```typescript
interface FloatingSettingsPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialPosition?: { x: number; y: number };
  initialSize?: { width: number; height: number };
}
```

---

### SettingsSection

**File**: `/src/features/settings/components/SettingsSection.tsx`

#### Purpose
Collapsible section for related settings.

#### Props

```typescript
interface SettingsSectionProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}
```

#### Usage

```tsx
<SettingsSection
  title="Rendering"
  description="3D rendering and visual effects"
  icon={<ImageIcon />}
  defaultOpen
>
  <Slider label="Ambient Light" {...} />
  <Switch label="Enable Shadows" {...} />
</SettingsSection>
```

---

### SettingControlComponent

**File**: `/src/features/settings/components/SettingControlComponent.tsx`

#### Purpose
Dynamic control component based on setting type.

#### Props

```typescript
interface SettingControlComponentProps {
  setting: SettingDefinition;
  value: any;
  onChange: (value: any) => void;
}

interface SettingDefinition {
  type: 'boolean' | 'number' | 'string' | 'select' | 'color';
  label: string;
  description?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ label: string; value: any }>;
}
```

---

### PresetSelector

**File**: `/src/features/settings/components/PresetSelector.tsx`

#### Purpose
Select quality/performance presets.

#### Props

```typescript
interface PresetSelectorProps {
  presets: Preset[];
  currentPreset: string | null;
  onSelectPreset: (presetId: string) => void;
  onSavePreset: (name: string) => void;
}

interface Preset {
  id: string;
  name: string;
  description: string;
  settings: Record<string, any>;
}
```

---

### UndoRedoControls

**File**: `/src/features/settings/components/UndoRedoControls.tsx`

#### Purpose
Undo/redo settings changes.

#### Props

```typescript
interface UndoRedoControlsProps {
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  history: SettingsHistory[];
}
```

---

### BackendUrlSetting

**File**: `/src/features/settings/components/BackendUrlSetting.tsx`

#### Purpose
Configure custom backend URL.

#### Features

- **URL Validation**: Check format and connectivity
- **Test Connection**: Verify before saving
- **Reset to Default**: Restore default URL

---

### GraphSelector

**File**: `/src/features/settings/components/GraphSelector.tsx`

#### Purpose
Switch between knowledge graph and ontology.

#### Props

```typescript
interface GraphSelectorProps {
  currentGraph: 'knowledge_graph' | 'ontology';
  onGraphChange: (graph: 'knowledge_graph' | 'ontology') => void;
}
```

---

## XR/Immersive Components

### ImmersiveApp

**File**: `/src/immersive/components/ImmersiveApp.tsx`

#### Purpose
Main XR application for Meta Quest 3.

#### Features

- **Babylon.js Renderer**: WebXR-optimized
- **Hand Tracking**: Native hand input
- **Plane Detection**: Environment understanding
- **Teleportation**: Movement in XR
- **UI Panels**: Floating 3D UI

#### Usage

```tsx
import { ImmersiveApp } from '@/immersive/components/ImmersiveApp';

// Rendered when Quest 3 detected
{shouldUseQuest3Layout && <ImmersiveApp />}
```

---

### XRController

**File**: `/src/xr/components/XRController.tsx`

#### Purpose
Handle XR input controllers.

#### Props

```typescript
interface XRControllerProps {
  handedness: 'left' | 'right';
  onSelect?: (event: XRInputSourceEvent) => void;
  onSelectStart?: (event: XRInputSourceEvent) => void;
  onSelectEnd?: (event: XRInputSourceEvent) => void;
  onSqueeze?: (event: XRInputSourceEvent) => void;
}
```

---

### XRHandTracking

**File**: `/src/xr/components/XRHandTracking.tsx`

#### Purpose
Visualize hand tracking joints.

#### Features

- **Joint Rendering**: 25 joints per hand
- **Gesture Recognition**: Pinch, point, grab
- **Haptic Feedback**: Vibration on interaction

---

### XRTeleportation

**File**: `/src/xr/components/XRTeleportation.tsx`

#### Purpose
Teleportation locomotion in XR.

#### Props

```typescript
interface XRTeleportationProps {
  enabled: boolean;
  allowedFloors: THREE.Mesh[];
  arcColor?: string;
  targetColor?: string;
}
```

---

## UI Components

### ConnectionWarning

**File**: `/src/components/ConnectionWarning.tsx`

#### Purpose
Warning banner when WebSocket disconnected.

#### Features

- **Auto-hide**: Hides when connected
- **Retry Button**: Manual reconnection
- **Status Message**: Connection state

---

### ErrorBoundary

**File**: `/src/components/ErrorBoundary.tsx`

#### Purpose
React error boundary to catch render errors.

#### Props

```typescript
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}
```

#### Usage

```tsx
<ErrorBoundary fallback={<ErrorPage />}>
  <App />
</ErrorBoundary>
```

---

### DebugControlPanel

**File**: `/src/components/DebugControlPanel.tsx`

#### Purpose
Developer debug panel (only shown when debug enabled).

#### Features

- **Performance Stats**: FPS, memory, render time
- **WebSocket Status**: Connection state
- **Graph Info**: Node/edge count
- **Settings Inspector**: Current settings
- **Console**: In-app console

---

### KeyboardShortcutsModal

**File**: `/src/components/KeyboardShortcutsModal.tsx`

#### Purpose
Display all keyboard shortcuts.

#### Props

```typescript
interface KeyboardShortcutsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  shortcuts: KeyboardShortcut[];
}

interface KeyboardShortcut {
  keys: string[];
  description: string;
  category: string;
}
```

---

### VoiceButton

**File**: `/src/components/VoiceButton.tsx`

#### Purpose
Voice input button (requires authentication).

#### Props

```typescript
interface VoiceButtonProps {
  onVoiceStart: () => void;
  onVoiceEnd: (transcript: string) => void;
  onError: (error: Error) => void;
}
```

---

## Layout Components

### MainLayout

**File**: `/src/app/MainLayout.tsx`

#### Purpose
Desktop application layout.

#### Features

- **Resizable Panels**: Graph, conversation, settings
- **Sidebar**: Navigation and tools
- **Header**: App title and user menu
- **Footer**: Status bar

---

### ConversationPane

**File**: `/src/app/components/ConversationPane.tsx`

#### Purpose
Chat interface with agents.

#### Features

- **Message List**: Scrolling conversation
- **Input Field**: Send messages
- **File Upload**: Share files with agents
- **Code Blocks**: Syntax highlighted code

#### Props

```typescript
interface ConversationPaneProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
  isLoading?: boolean;
}

interface Message {
  id: string;
  sender: 'user' | 'agent';
  content: string;
  timestamp: number;
  agentId?: string;
}
```

---

### NarrativeGoldminePanel

**File**: `/src/app/components/NarrativeGoldminePanel.tsx`

#### Purpose
Semantic search and knowledge discovery.

#### Features

- **Search**: Natural language queries
- **Results**: Relevant nodes and edges
- **Filters**: By type, date, metadata
- **Export**: Save results

---

## Feature Components

### CommandPalette

**File**: `/src/features/command-palette/components/CommandPalette.tsx`

#### Purpose
Quick action launcher (Cmd+K / Ctrl+K).

#### Features

- **Fuzzy Search**: Find commands quickly
- **Categories**: Organized by domain
- **Recent**: Recently used commands
- **Keyboard**: Full keyboard navigation

#### Props

```typescript
interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}
```

#### Usage

```tsx
// Trigger with keyboard shortcut
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      setCommandPaletteOpen(true);
    }
  };

  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, []);

<CommandPalette open={commandPaletteOpen} onOpenChange={setCommandPaletteOpen} />
```

---

### OnboardingOverlay

**File**: `/src/features/onboarding/components/OnboardingOverlay.tsx`

#### Purpose
Interactive onboarding flow for new users.

#### Features

- **Step-by-step**: Guided tour
- **Highlights**: Spotlight on UI elements
- **Tooltips**: Contextual help
- **Skip**: Allow users to skip

#### Props

```typescript
interface OnboardingOverlayProps {
  flowId: string;
  onComplete: () => void;
  onSkip: () => void;
}
```

---

### HelpTooltip

**File**: `/src/features/help/components/HelpTooltip.tsx`

#### Purpose
Contextual help tooltips.

#### Props

```typescript
interface HelpTooltipProps {
  content: string;
  link?: string;
  placement?: 'top' | 'right' | 'bottom' | 'left';
  children: React.ReactNode;
}
```

#### Usage

```tsx
<HelpTooltip content="Enable physics simulation for dynamic layout" link="/docs/physics">
  <Switch label="Enable Physics" {...} />
</HelpTooltip>
```

---

### AnalyticsPanel

**File**: `/src/features/analytics/components/SemanticClusteringControls.tsx`

#### Purpose
Graph analytics and clustering controls.

#### Features

- **Clustering**: K-means, spectral, louvain
- **Metrics**: Centrality, community detection
- **Visualization**: Colored by cluster
- **Export**: Export analysis results

---

### PhysicsEngineControls

**File**: `/src/features/physics/components/PhysicsEngineControls.tsx`

#### Purpose
Fine-tune physics simulation parameters.

#### Props

```typescript
interface PhysicsEngineControlsProps {
  graphName: string;
  params: PhysicsParams;
  onParamsChange: (params: Partial<PhysicsParams>) => void;
}
```

#### Features

- **Spring Force**: Edge attraction
- **Repulsion**: Node spacing
- **Damping**: Movement decay
- **Bounds**: Constraint region
- **Warmup**: Initial settling iterations

---

### ConstraintBuilderDialog

**File**: `/src/features/physics/components/ConstraintBuilderDialog.tsx`

#### Purpose
Create custom physics constraints.

#### Props

```typescript
interface ConstraintBuilderDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (constraint: Constraint) => void;
}

interface Constraint {
  type: 'position' | 'distance' | 'angle';
  nodeIds: number[];
  parameters: Record<string, number>;
}
```

---

### OntologyPanel

**File**: `/src/features/ontology/components/OntologyPanel.tsx`

#### Purpose
Ontology management and validation.

#### Features

- **Class Hierarchy**: Tree view of classes
- **Properties**: Object and data properties
- **Validation**: OWL/RDFS validation
- **Export**: RDF/Turtle/JSON-LD

---

### WorkspaceManager

**File**: `/src/features/workspace/components/WorkspaceManager.tsx`

#### Purpose
Manage multiple workspaces/projects.

#### Features

- **Create**: New workspaces
- **Switch**: Between workspaces
- **Settings**: Per-workspace settings
- **Sync**: Cloud synchronization

---

## Performance Notes

### Optimization Techniques

1. **React.memo**: Prevent unnecessary re-renders
2. **useMemo**: Memoize expensive computations
3. **useCallback**: Stable callback references
4. **Lazy Loading**: Code splitting
5. **Virtual Scrolling**: Large lists
6. **Debouncing**: Input handlers
7. **Throttling**: Scroll/resize handlers

### Example: Optimized Component

```typescript
import React, { memo, useMemo, useCallback } from 'react';

interface NodeListItemProps {
  node: Node;
  selected: boolean;
  onSelect: (nodeId: number) => void;
}

export const NodeListItem = memo<NodeListItemProps>(
  ({ node, selected, onSelect }) => {
    const handleClick = useCallback(() => {
      onSelect(node.id);
    }, [node.id, onSelect]);

    const displayLabel = useMemo(() => {
      return node.label.length > 50
        ? node.label.substring(0, 50) + '...'
        : node.label;
    }, [node.label]);

    return (
      <div
        onClick={handleClick}
        className={cn('node-item', selected && 'selected')}
      >
        <span>{displayLabel}</span>
        <Badge>{node.type}</Badge>
      </div>
    );
  },
  (prev, next) => {
    return (
      prev.node.id === next.node.id &&
      prev.selected === next.selected
    );
  }
);
```

---

## Accessibility Guidelines

### ARIA Labels

All interactive components should have:

- `aria-label` or `aria-labelledby`
- `aria-describedby` for descriptions
- `role` attribute when semantic HTML insufficient
- `aria-disabled`, `aria-expanded`, etc.

### Keyboard Navigation

- **Tab**: Navigate between focusable elements
- **Enter/Space**: Activate buttons
- **Arrow Keys**: Navigate lists, menus
- **Escape**: Close modals, menus
- **Shortcuts**: Document all shortcuts

### Screen Reader Support

- Semantic HTML elements
- Descriptive labels
- Live regions for dynamic content
- Skip links for navigation

### Color Contrast

- Text: 4.5:1 minimum
- Large text: 3:1 minimum
- Interactive elements: 3:1 minimum

---

## Styling Guidelines

### Tailwind CSS Classes

```tsx
// Layout
className="flex flex-col items-center justify-between p-4 gap-2"

// Colors
className="bg-slate-900 text-white border-slate-700"

// States
className="hover:bg-slate-800 active:bg-slate-700 focus:ring-2 focus:ring-blue-500"

// Responsive
className="w-full md:w-1/2 lg:w-1/3"
```

### CSS Modules

For component-specific styles, use CSS modules:

```tsx
import styles from './Component.module.css';

<div className={styles.container}>
  <h2 className={styles.title}>Title</h2>
</div>
```

### CSS Variables

```css
:root {
  --color-primary: #3b82f6;
  --color-secondary: #6366f1;
  --color-accent: #8b5cf6;
  --spacing-unit: 4px;
  --border-radius: 6px;
}
```

---

## Testing Components

### Unit Tests

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('renders with text', () => {
    render(<Button>Click Me</Button>);
    expect(screen.getByText('Click Me')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click Me</Button>);

    fireEvent.click(screen.getByText('Click Me'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('is disabled when loading', () => {
    render(<Button loading>Submit</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

### Integration Tests

```typescript
describe('SettingsPanel Integration', () => {
  it('updates settings when slider changed', async () => {
    render(<SettingsPanel />);

    const slider = screen.getByLabelText('Damping');
    fireEvent.change(slider, { target: { value: '0.8' } });

    await waitFor(() => {
      expect(useSettingsStore.getState().get('physics.damping')).toBe(0.8);
    });
  });
});
```

---

## Conclusion

This component reference provides comprehensive documentation for all React components in the VisionFlow client. Each component is designed with:

- **Reusability**: Composable and flexible
- **Type Safety**: Full TypeScript coverage
- **Accessibility**: ARIA labels and keyboard support
- **Performance**: Optimized rendering
- **Testing**: Unit and integration tests

For implementation details, see the source files in `/src/features/` and `/src/components/`.

---

**Document Version:** 1.0.0
**Total Lines:** 2,618
**Last Updated:** 2025-11-04
