# Precise Implementation Plan (Version 1)

This plan outlines the exact file changes required to implement the lightweight validation feature.

## Backend Changes

1.  **New Actor: `OntologyActor`**
    *   **File:** `src/actors/ontology_actor.rs` (New)
    *   **Purpose:** Implements the lightweight rule validation logic. It will receive `GraphData`, apply the rules defined in `rules_v1.md`, and return a `ValidationReport`. This actor will use `tokio::task::spawn_blocking` to ensure its computations do not block the main actor threads.
2.  **New Messages**
    *   **File:** `src/actors/messages.rs`
    *   **Add:** `ValidateGraph`, `TriggerValidation`, `GetValidationResult` messages and the `ValidationReport`, `ValidationTaskStatus` DTOs.
3.  **Orchestration**
    *   **File:** `src/actors/graph_actor.rs`
    *   **Changes:**
        *   Add `ontology_actor_addr: Addr<OntologyActor>` and `validation_tasks: Arc<Mutex<HashMap<String, ValidationTaskStatus>>>` fields.
        *   Implement handlers for `TriggerValidation` (spawns the async validation task) and `GetValidationResult` (retrieves task status).
4.  **Application Wiring**
    *   **File:** `src/app_state.rs`
    *   **Changes:** Instantiate the `OntologyActor` at startup and pass its address to the `GraphServiceActor` constructor.
5.  **API Routes**
    *   **File:** `src/handlers/api_handler/analytics/mod.rs`
    *   **Changes:**
        *   Add `trigger_validation` and `get_validation_status` handler functions.
        *   Register the new routes (`/validate` and `/validate/status/{task_id}`) in the `config` function.

## Frontend Changes

1.  **State Management**
    *   **File:** `client/src/features/analytics/store/analyticsStore.ts`
    *   **Changes:** Add state properties for `validationStatus`, `validationReport`, etc., and actions like `triggerValidation` to handle the API interaction (POST then poll).
2.  **UI Component**
    *   **File:** `client/src/features/analytics/components/OntologyControls.tsx` (New)
    *   **Purpose:** A React component with a button to trigger validation, a display for the results summary, and a toggle for inferred edges (for future use).
3.  **UI Integration**
    *   **File:** `client/src/features/visualisation/components/tabs/GraphAnalysisTab.tsx`
    *   **Changes:** Import and render the new `OntologyControls` component.
4.  **Visualization**
    *   **File:** `client/src/features/graph/components/GraphManager.tsx`
    *   **Changes:**
        *   Subscribe to the `validationReport` from the `analyticsStore`.
        *   When a report is present, iterate through inconsistencies and update the `instanceColor` for the corresponding nodes to red, similar to the existing SSSP visualization logic.