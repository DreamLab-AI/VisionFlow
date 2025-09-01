# API Specification & DTOs (Version 1)

This document outlines the API endpoints and data transfer objects for the ontology validation feature.

## Endpoints

### `POST /api/analytics/validate`

Triggers the validation process on the current graph data. This is an asynchronous operation.

-   **Request Body:** Empty.
-   **Response (202 Accepted):**
    ```json
    {
      "success": true,
      "message": "Validation task started",
      "taskId": "b7f1a2f4-1c41-4a1a-8a0e-0b41f084a1b4",
      "statusUrl": "/api/analytics/validate/status/b7f1a2f4-1c41-4a1a-8a0e-0b41f084a1b4"
    }
    ```

### `GET /api/analytics/validate/status/{task_id}`

Polls for the result of a validation task.

-   **URL Parameters:**
    -   `task_id` (string, required): The ID of the validation task.
-   **Response (Pending):**
    ```json
    {
      "success": true,
      "status": { "Pending": null }
    }
    ```
-   **Response (Completed):**
    ```json
    {
      "success": true,
      "status": {
        "Completed": {
          "inconsistencies": [
            { "message": "Self-edge not allowed on 'relatedTo' property", "source_node": 42, "target_node": 42 },
            { "message": "'communicationWith' property must connect two Agent nodes", "source_node": 10012, "target_node": 7 }
          ],
          "inferred_triples": []
        }
      }
    }
    ```
-   **Response (Failed):**
    ```json
    {
        "success": true,
        "status": {
            "Failed": "Reason for failure."
        }
    }
    ```

## Data Transfer Objects (DTOs)

These shapes will be defined in Rust ([`src/actors/messages.rs`](src/actors/messages.rs)) and TypeScript ([`client/src/features/analytics/store/analyticsStore.ts`](client/src/features/analytics/store/analyticsStore.ts)).

-   **`ValidationReport`**
    -   `inconsistencies`: `Inconsistency[]`
    -   `inferred_triples`: `InferredTriple[]` (will be empty in v1)
-   **`Inconsistency`**
    -   `message`: `string`
    -   `source_node?`: `u32`
    -   `target_node?`: `u32`
-   **`InferredTriple`**
    -   `source_node`: `u32`
    -   `property`: `string`
    -   `target_node`: `u32`
-   **`ValidationTaskStatus`** (Enum/Union)
    -   `Pending`
    -   `Completed(ValidationReport)`
    -   `Failed(string)`