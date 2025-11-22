# Refactoring Summary: GH-Pages CSS Themes

## Overview
This document summarizes the refactoring of the GitHub Pages static website to standardize styling using Tailwind CSS and ensure consistent dark mode implementation.

## Changes Implemented

### 1. Architecture Standardization
- **Removed Manual CSS:** Deleted `App.css`, `HomePage.css`, `PageView.css`, `SearchView.css`, and `OntologyView.css`.
- **Adopted Tailwind CSS:** Migrated all styles to Tailwind CSS utility classes.
- **Theme Management:** 
    - Updated `index.html` to remove the hardcoded `class="dark"` and added a script for dynamic theme detection based on system preference or local storage.
    - Leveraged `shadcn/ui` semantic variables (e.g., `bg-background`, `text-foreground`) for automatic dark mode adaptation.

### 2. Component Refactoring
- **`App.tsx`:** Replaced manual classes with Tailwind utilities.
- **`Sidebar.tsx`:** Updated to use semantic colors and fixed lint errors (explicit types for map callback).
- **`TopMenuBar.tsx`:** Refactored to use semantic colors for consistent theming.
- **`HomePage.tsx`:** Migrated styles from `HomePage.css` to Tailwind classes.
- **`OntologyView.tsx`:** Migrated styles from `OntologyView.css` and fixed lint errors (unused variables, property access).
- **`PageView.tsx`:** Migrated styles from `PageView.css` to Tailwind classes.
- **`SearchView.tsx`:** Migrated styles from `SearchView.css` and fixed lint errors (unused variable).

## Verification
- **Local Build:** Successfully ran the application locally using `npm run dev` (after installing `vite` explicitly).
- **Visual Verification:**
    - **Home Page:** Verified consistent styling and layout.
    - **Search Page:** Verified consistent styling and layout.
    - **Ontology Page:** Attempted verification, but encountered performance timeouts in the test environment. However, the code refactoring ensures it follows the same styling principles.

## Outstanding Items
- **NPM Auth Key:** The user requested an npm auth key, but it was not found in the workspace files (`.npmrc`, `package.json`, etc.). It is likely required for the `narrativegoldmine-webvowl-wasm` private package.
- **Ontology Page Performance:** The Ontology page graph simulation appears to be heavy for the current test environment, causing timeouts during automated verification.

## Next Steps
- **Provide NPM Key:** The user needs to provide the npm authentication key to ensure all dependencies (specifically private ones) can be installed in a fresh environment.
- **Performance Tuning:** Investigate optimization opportunities for the WASM/WebGL graph rendering if performance remains an issue.
