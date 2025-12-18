---
name: frontend-creator
description: Create distinctive, production-grade frontend interfaces with modern tooling. Combines design philosophy with React + TypeScript + Vite + Tailwind CSS + shadcn/ui for claude.ai artifacts and full web applications.
license: Complete terms in LICENSE.txt
version: 2.0.0
---

# Frontend Creator

**Unified skill combining design philosophy with modern implementation tooling.**

Create production-grade frontend interfaces that are:
- **Visually distinctive** - Avoid generic AI aesthetics
- **Technically robust** - React 18 + TypeScript + Vite
- **Fully featured** - shadcn/ui components, Tailwind CSS, routing, state management

---

## Part 1: Design Philosophy

Before coding, commit to a **BOLD aesthetic direction**:

### Design Thinking Process

1. **Purpose**: What problem does this interface solve? Who uses it?
2. **Tone**: Choose an extreme aesthetic:
   - Brutally minimal
   - Maximalist chaos
   - Retro-futuristic
   - Organic/natural
   - Luxury/refined
   - Playful/toy-like
   - Editorial/magazine
   - Brutalist/raw
   - Art deco/geometric
   - Soft/pastel
   - Industrial/utilitarian
3. **Constraints**: Technical requirements (framework, performance, accessibility)
4. **Differentiation**: What makes this UNFORGETTABLE?

**CRITICAL**: Choose a clear conceptual direction and execute with precision.

### Frontend Aesthetics Guidelines

#### Typography
- Choose **distinctive, characterful fonts** that elevate aesthetics
- ❌ Avoid: Arial, Inter, Roboto, system fonts
- ✅ Use: Unexpected pairings (display + refined body font)
- Font hierarchy with intentional sizing

#### Color & Theme
- Commit to a **cohesive aesthetic**
- Use CSS variables for consistency
- Dominant colors with sharp accents
- ❌ Avoid: Purple gradients on white, cliched schemes
- ✅ Create: Context-specific palettes

#### Motion & Animation
- High-impact moments over scattered micro-interactions
- Staggered reveals with `animation-delay`
- Scroll-triggered effects
- Hover states that surprise
- **Priority**: CSS-only for HTML, Motion library for React

#### Spatial Composition
- Unexpected layouts
- Asymmetry and overlap
- Diagonal flow
- Grid-breaking elements
- Generous negative space OR controlled density

#### Backgrounds & Visual Details
Create atmosphere rather than defaulting to solid colors:
- Gradient meshes
- Noise textures
- Geometric patterns
- Layered transparencies
- Dramatic shadows
- Decorative borders
- Custom cursors
- Grain overlays

### Anti-Patterns (Never Use)

❌ **Generic AI aesthetics**:
- Overused fonts (Inter, Roboto, Arial, system fonts)
- Purple gradients on white
- Predictable layouts
- Cookie-cutter design
- Space Grotesk convergence

❌ **"AI slop" indicators**:
- Excessive centered layouts
- Uniform rounded corners
- Predictable component patterns

### Complexity Matching

- **Maximalist designs** → Elaborate code with extensive animations
- **Minimalist designs** → Restraint, precision, careful spacing
- **Elegance** = Executing the vision well

---

## Part 2: Implementation Tooling

### Tech Stack

**Core**:
- React 18 + TypeScript
- Vite (development)
- Parcel (bundling to single HTML)

**Styling**:
- Tailwind CSS 3.4.1
- shadcn/ui theming system
- 40+ pre-installed components

**Features**:
- Path aliases (`@/`) configured
- All Radix UI dependencies
- Node 18+ compatibility

---

## Workflow

### Step 1: Initialize Project

```bash
bash scripts/init-artifact.sh <project-name>
cd <project-name>
```

Creates fully configured project with:
- ✅ React + TypeScript (via Vite)
- ✅ Tailwind CSS + shadcn/ui
- ✅ 40+ components pre-installed
- ✅ Parcel bundling configured

### Step 2: Develop

Edit the generated files following design philosophy above.

**Common tasks**:

#### Add shadcn/ui Components
```bash
# All 40+ components already installed
# Available components:
# - accordion, alert, avatar, badge, button, calendar, card, checkbox
# - dialog, dropdown-menu, input, label, popover, select, tabs, toast
# - And 25+ more...
```

#### Customize Theme
```css
/* tailwind.config.js - Edit CSS variables */
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --primary: 222.2 47.4% 11.2%;
  /* ... customize all theme variables */
}
```

#### Add Routing (if needed)
```bash
npm install react-router-dom
```

### Step 3: Bundle to Single HTML

```bash
bash scripts/bundle-artifact.sh
```

**Output**: `artifact.html` - Self-contained, production-ready

**What it does**:
1. Builds optimized production bundle (Vite)
2. Inlines all JavaScript
3. Inlines all CSS (including Tailwind utilities)
4. Creates single ~100KB HTML file
5. Ready for claude.ai artifact display

---

## Advanced Usage

### Testing the Artifact

```bash
# Development server
npm run dev

# Production preview
npm run build
npm run preview

# Test bundled artifact
python3 -m http.server 8000
# Open: http://localhost:8000/artifact.html
```

### Customization

**Vite config** (`vite.config.ts`):
- Adjust build options
- Add plugins
- Configure aliases

**Parcel config** (`.parcelrc`):
- Already configured for optimal bundling
- Includes HTMLInline plugin

**Tailwind config** (`tailwind.config.js`):
- Extend theme
- Add custom utilities
- Configure plugins

---

## When to Use

✅ **Use frontend-creator for**:
- Complex artifacts requiring state management
- Multi-component applications
- Projects needing routing
- shadcn/ui component integration
- Production-grade web applications
- Artifacts requiring distinctive design

❌ **Don't use for**:
- Simple single-file HTML artifacts
- Quick prototypes (use direct HTML/JSX instead)

---

## Examples

### Minimal + Elegant
```tsx
// Clean, refined, exceptional spacing
<div className="min-h-screen flex items-center justify-center bg-stone-50">
  <Card className="max-w-md p-8 shadow-2xl">
    <h1 className="font-serif text-4xl mb-4 text-stone-900">Elegant</h1>
    <p className="text-stone-600 leading-relaxed">Restraint with precision.</p>
  </Card>
</div>
```

### Maximalist + Bold
```tsx
// Chaos with intentionality
<div className="min-h-screen bg-gradient-to-br from-yellow-400 via-pink-500 to-purple-600 p-8">
  <div className="transform rotate-2 bg-black text-white p-12 shadow-brutal">
    <h1 className="font-display text-8xl mb-4 animate-pulse">CHAOS</h1>
    <div className="grid grid-cols-3 gap-4">
      {/* 12 animated cards with staggered delays */}
    </div>
  </div>
</div>
```

---

## Design Decision Matrix

| Aesthetic | Typography | Colors | Layout | Motion |
|-----------|-----------|--------|--------|--------|
| Minimal | Serif + Sans | Monochrome + 1 accent | Generous whitespace | Subtle fades |
| Brutalist | Display mono | Black + bright accent | Raw grid breaks | Hard cuts |
| Luxury | Refined serif | Deep jewel tones | Asymmetric balance | Smooth transitions |
| Playful | Round display | Vibrant pastels | Overlapping chaos | Bouncy springs |

---

## Technical Notes

### Bundle Size
- Typical artifact: 80-150KB (HTML + inlined CSS/JS)
- Includes full Tailwind utilities
- All dependencies bundled

### Browser Support
- Modern browsers (ES2015+)
- No polyfills included
- Tested: Chrome, Firefox, Safari, Edge

### Performance
- Vite: Lightning-fast HMR
- Parcel: Optimized production builds
- Tree-shaking enabled
- Minification automatic

---

## Troubleshooting

**Issue**: Bundle too large
**Fix**: Remove unused shadcn/ui components, purge Tailwind

**Issue**: Vite dev server won't start
**Fix**: Check Node version (requires 18+), delete `node_modules`, reinstall

**Issue**: Parcel build fails
**Fix**: Ensure `.parcelrc` exists, check for conflicting plugins

---

## Philosophy

> **Claude is capable of extraordinary creative work.**
>
> Don't hold back. Show what can truly be created when thinking outside the box and committing fully to a distinctive vision.

Match implementation complexity to aesthetic vision. Elegance comes from executing the vision well, whether that's maximalist elaborate code or minimalist precision.

**Remember**: No two designs should be the same. Vary themes, fonts, aesthetics. Make unexpected choices that feel genuinely designed for the context.
