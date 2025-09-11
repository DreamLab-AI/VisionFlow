# UI Component Library

*[Client](../index.md)*

This document serves as a comprehensive reference for the client's design system and reusable UI component library. The components are built with React, TypeScript, and [Radix UI](https://www.radix-ui.com/) for accessibility, and styled with [Tailwind CSS](https://tailwindcss.com/).

## Philosophy

-   **Accessibility First**: Components are built on top of Radix UI primitives, ensuring they are accessible out-of-the-box (keyboard navigation, ARIA attributes, etc.).
-   **Composition over Configuration**: Components are designed to be composed together to build complex UIs, favouring flexibility.
-   **Theming**: The system is themed using CSS variables, allowing for easy customisation and consistency.

---

## Component Reference

### Button

**Source**: [`client/src/features/design-system/components/Button.tsx`](../../client/src/features/design-system/components/Button.tsx)

**Purpose**: A versatile button component for user actions. It includes multiple visual styles, sizes, and states.

**Props**:

| Prop           | Type                          | Description                                                                                             |
| -------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `variant`      | `string`                      | Visual style: `default`, `destructive`, `outline`, `secondary`, `ghost`, `link`, `gradient`, `glow`.      |
| `size`         | `string`                      | Size of the button: `default`, `sm`, `lg`, `icon-sm`, `icon`, `icon-lg`.                                 |
| `asChild`      | `boolean`                     | Renders the component as a child of its parent, merging props and behaviour.                            |
| `loading`      | `boolean`                     | If `true`, displays a loading spinner and disables the button.                                          |
| `loadingText`  | `string`                      | Optional text to display next to the spinner when `loading` is `true`.                                  |
| `icon`         | `React.ReactNode`             | An optional icon to display within the button.                                                          |
| `iconPosition` | `'left'` \| `'right'`         | Determines the position of the icon relative to the button text. Defaults to `left`.                     |

**Usage Example**:

```tsx
import { Button } from '@/features/design-system/components';
import { Mail } from 'lucide-react';

<Button variant="outline" size="lg" onClick={() => console.log('Clicked!')}>
  <Mail className="mr-2 h-4 w-4" /> Login with Email
</Button>

<Button variant="default" loading loadingText="Saving...">
  Save Changes
</Button>
```

---

### Input

**Source**: [`client/src/features/design-system/components/Input.tsx`](../../client/src/features/design-system/components/Input.tsx)

**Purpose**: A flexible text input field with various styles, states, and advanced features like floating labels and clearable inputs.

**Props**:

| Prop            | Type                          | Description                                                                                             |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `variant`       | `string`                      | Visual style: `default`, `filled`, `flushed`, `ghost`, `outlined`.                                      |
| `size`          | `string`                      | Size of the input: `xs`, `sm`, `default`, `lg`, `xl`.                                                   |
| `state`         | `string`                      | Visual state: `default`, `error`, `success`, `warning`.                                                 |
| `label`         | `string`                      | A label for the input. Can be standard or floating.                                                     |
| `error`         | `string`                      | An error message to display. Sets `state` to `error`.                                                   |
| `icon`          | `React.ReactNode`             | An optional icon to display within the input field.                                                     |
| `iconPosition`  | `'left'` \| `'right'`         | Position of the icon. Defaults to `left`.                                                               |
| `clearable`     | `boolean`                     | If `true`, shows a clear button when the input has a value.                                             |
| `floatingLabel` | `boolean`                     | If `true`, the label will float above the input when focused or has a value.                            |

**Usage Example**:

```tsx
import { Input } from '@/features/design-system/components';
import { User } from 'lucide-react';

<Input
  variant="filled"
  label="Username"
  floatingLabel
  icon={<User />}
  iconPosition="left"
  placeholder="Enter your username"
  error="Username is required."
/>
```

---

### Card

**Source**: [`client/src/features/design-system/components/Card.tsx`](../../client/src/features/design-system/components/Card.tsx)

**Purpose**: A container component for grouping related content with a consistent visual style.

**Sub-components**:

-   `CardHeader`: Container for the card's title and description.
-   `CardTitle`: The main heading of the card.
-   `CardDescription`: A subtitle or description for the card.
-   `CardContent`: The main content area of the card.
-   `CardFooter`: A container for actions or supplemental information at the bottom of the card.

**Usage Example**:

```tsx
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/features/design-system/components';

<Card>
  <CardHeader>
    <CardTitle>Create Project</CardTitle>
    <CardDescription>Deploy your new project in one-click.</CardDescription>
  </CardHeader>
  <CardContent>
    {/* Form fields go here */}
  </CardContent>
  <CardFooter>
    <Button variant="outline">Cancel</Button>
    <Button>Deploy</Button>
  </CardFooter>
</Card>
```

---

### Slider

**Source**: [`client/src/features/design-system/components/Slider.tsx`](../../client/src/features/design-system/components/Slider.tsx)

**Purpose**: A range slider for selecting a numeric value within a defined range.

**Props**: Inherits all props from `Radix UI Slider`.

| Prop      | Type                | Description                               |
| --------- | ------------------- | ----------------------------------------- |
| `value`   | `number[]`          | The controlled value of the slider.       |
| `onValueChange` | `(value: number[]) => void` | Event handler for when the value changes. |
| `min`     | `number`            | The minimum value of the range.           |
| `max`     | `number`            | The maximum value of the range.           |
| `step`    | `number`            | The step increment.                       |

**Usage Example**:

```tsx
import { Slider } from '@/features/design-system/components';

<Slider
  defaultValue={[50]}
  max={100}
  step={1}
  onValueChange={(value) => console.log(value)}
/>
```

---

### Switch

**Source**: [`client/src/features/design-system/components/Switch.tsx`](../../client/src/features/design-system/components/Switch.tsx)

**Purpose**: A toggle switch for boolean (on/off) settings.

**Props**: Inherits all props from `Radix UI Switch`.

| Prop              | Type                          | Description                                   |
| ----------------- | ----------------------------- | --------------------------------------------- |
| `checked`         | `boolean`                     | The controlled state of the switch.           |
| `onCheckedChange` | `(checked: boolean) => void`  | Event handler for when the state changes.     |

**Usage Example**:

```tsx
import { Switch } from '@/features/design-system/components';
import { Label } from '@/features/design-system/components';

<div className="flex items-centre space-x-2">
  <Switch id="airplane-mode" />
  <Label htmlFor="airplane-mode">Airplane Mode</Label>
</div>
```

---

### Dialogue

**Source**: [`client/src/features/design-system/components/Dialogue.tsx`](../../client/src/features/design-system/components/Dialogue.tsx)

**Purpose**: A modal dialogue that appears over the main content to display critical information or request user input.

**Sub-components**:

-   `DialogTrigger`: The button or element that opens the dialogue.
-   `DialogContent`: The main content of the dialogue.
-   `DialogHeader`: Container for the dialogue's title and description.
-   `DialogTitle`: The title of the dialogue.
-   `DialogDescription`: A description or subtitle for the dialogue.
-   `DialogFooter`: A container for action buttons.
-   `DialogClose`: A button to close the dialogue.

**Usage Example**:

```tsx
import { Dialogue, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/features/design-system/components';

<Dialogue>
  <DialogTrigger asChild>
    <Button variant="outline">Edit Profile</Button>
  </DialogTrigger>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Edit Profile</DialogTitle>
      <DialogDescription>
        Make changes to your profile here. Click save when you're done.
      </DialogDescription>
    </DialogHeader>
    {/* Form fields go here */}
    <DialogFooter>
      <Button type="submit">Save changes</Button>
    </DialogFooter>
  </DialogContent>
</Dialogue>
```

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
- [User Controls Summary - Settings Panel](../client/user-controls-summary.md)
- [VisionFlow Client Documentation](../client/index.md)
- [WebSocket Communication](../client/websocket.md)
- [WebXR Integration](../client/xr-integration.md)
