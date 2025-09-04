# Build Configuration Instructions - Settings and Configuration

## **Task 5.1: Update Visualization Settings**
*   **Goal:** Enable glow effects and reorganize controller button mappings
*   **Actions:**
    1. In `data/settings.yaml`: Enable glow visualization:
       - Change `glow.enabled` from `false` to `true`
       - This enables visual glow effects in the WebXR interface
    
    2. Reorganize controller button functions:
       - Reorder button mappings for better UX:
         - `'1': resetView` (unchanged)
         - `'2': cycleMode` (unchanged) 
         - `'3': fitToView` (unchanged)
         - `'4': topView` (unchanged)
         - `'5': rightView` (unchanged)
         - `'6': frontView` (moved from position, unchanged function)
         - `'7': leftView` (new mapping)
         - `'8': bottomView` (new mapping)
    
    3. Enhanced controller functionality:
       - Added `leftView` and `bottomView` options
       - Provides complete 6-axis view control (front, back, left, right, top, bottom)
       - Maintains existing button assignments while adding new views

## **Implementation Notes:**
- Glow effects enhance visual feedback in VR/AR environments
- Extended view controls provide better spatial navigation
- Settings changes affect runtime behavior without code changes