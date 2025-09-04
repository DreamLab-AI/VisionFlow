# Build Configuration Instructions - package.json Dependencies

## **Task 4.1: Reorganize package.json Dependencies**
*   **Goal:** Ensure dependencies are correctly categorized between runtime and development dependencies
*   **Actions:**
    1. In `package.json`: Move development-only packages to `devDependencies`:
       - Move `@types/node` from `dependencies` to `devDependencies`
       - Move `wscat` from `dependencies` to `devDependencies`
    
    2. Verify categorization:
       - `@types/node` is TypeScript type definitions - development only
       - `wscat` is WebSocket testing tool - development only
       - These should not be included in production builds

## **Task 4.2: Update Build Scripts for Type Generation**
*   **Goal:** Add TypeScript type generation scripts to build process
*   **Actions:**
    1. Add type generation scripts:
       - Add `"types:generate": "cd .. && cargo build"` to scripts section
       - This script builds Rust project which generates TypeScript types via specta
    
    2. Consider adding development workflow scripts:
       - Add `"types:watch": "cd .. && cargo watch -x build"` for development
       - Add `"types:clean": "rm -rf src/types/generated"` for cleanup

## **Implementation Notes:**
- Proper dependency categorization reduces production bundle size
- Type generation integration enables shared types between Rust backend and frontend
- Build scripts support the new specta-based type generation system