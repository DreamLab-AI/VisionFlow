# Investigation Report: Website Color & Theme Issues

## Executive Summary
This report details the investigation and resolution of persistent styling issues on the Narrative Goldmine website, specifically regarding the "white boundary" and "white cards" that appeared despite the application of a dark theme. The investigation revealed a combination of missing explicit background styles on root elements, hardcoded legacy styles, and incorrect Tailwind CSS configuration. All issues have been resolved, ensuring a consistent premium dark theme across the application.

## Issue 1: White Boundary (Root Background)

### Symptom
Users reported a "white boundary" or white background areas visible around the main content, particularly when scrolling or on larger screens where content didn't fill the viewport.

### Investigation
1.  **Browser Inspection:** Using a browser subagent, we inspected the computed styles of the `html` and `body` elements on the live site.
2.  **Findings:**
    *   The `html` element had `class="dark"`, but its computed `background-color` was defaulting to white (or user-agent default) in some contexts.
    *   The `body` element also lacked a forceful dark background definition that would cover the entire viewport height reliably.
    *   Critical CSS was missing from `index.html`, causing a potential "white flash" before the main CSS bundle loaded.

### Root Cause
*   **Missing Explicit Styles:** The `index.css` file relied on inherited or default styles which were insufficient to override browser defaults for the root elements.
*   **Height Issues:** `html` and `body` were not explicitly set to `height: 100%`, potentially leaving gaps if content was short.

### Resolution
1.  **Critical CSS:** Added a `<style>` block to `index.html` to immediately set `background-color: #111827` and `color: #cdd6f4` for `html` and `body`. This prevents white flashes during load.
2.  **CSS Enforcement:** Updated `src/index.css` to explicitly target `html` and `body` with:
    ```css
    html, body {
      height: 100%;
      min-height: 100vh;
      background-color: #020817; /* Fallback */
      background-color: hsl(var(--background));
      color: hsl(var(--foreground));
    }
    ```
    This ensures the dark theme background covers the entire viewport at all times.

---

## Issue 2: White Cards (Content Styling)

### Symptom
Even after fixing the root background, cards on the homepage (specifically the Statistics card and others) appeared with white backgrounds or light gradients, clashing with the dark theme.

### Investigation
1.  **Codebase Scan:** We performed a comprehensive scan of the codebase to identify style definitions for `.home-card` and `.info-card`.
2.  **Findings:**
    *   **Hardcoded Gradient:** `src/pages/HomePage.css` contained a hardcoded light gradient for `.info-card`:
        ```css
        .info-card {
          background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); /* Light blue/purple */
        }
        ```
    *   **Tailwind Configuration:** `tailwind.config.js` was set to `darkMode: "media"`. Since the application forces dark mode via `class="dark"` on the `html` element, this configuration caused Tailwind to ignore the class and potentially apply light mode defaults (if the user's OS was in light mode).

### Root Cause
*   **Legacy Styles:** The hardcoded gradient in `HomePage.css` explicitly overrode the dark theme variables.
*   **Configuration Mismatch:** `darkMode: "media"` prevented Tailwind from correctly interpreting the application's forced dark mode state.

### Resolution
1.  **Remove Legacy Styles:** Commented out/removed the linear gradient in `src/pages/HomePage.css`, allowing the card to inherit the correct `bg-card` style.
2.  **Update Configuration:** Changed `tailwind.config.js` to use `darkMode: "class"`. This ensures Tailwind respects the `class="dark"` attribute on the `html` tag, correctly applying dark mode variants and variables.

## Conclusion
The combination of these fixes ensures that:
1.  The entire page background is consistently dark (`#020817` / `hsl(217.5 36.7% 10%)`).
2.  Content elements like cards correctly use the defined dark theme variables (`--card`).
3.  No white flashes occur during page load.

The website now presents a unified, premium dark interface as intended.
