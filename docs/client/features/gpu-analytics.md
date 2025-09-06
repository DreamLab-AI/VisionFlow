# GPU-Accelerated Analytics

This document describes the client-side implementation for triggering and visualizing GPU-accelerated graph analytics, specifically the Single-Source Shortest Path (SSSP) feature.

## Overview

The client provides an interface to compute and visualize the shortest paths from a selected source node to all other nodes in the graph. While the heavy computation is offloaded to the backend, potentially using GPU acceleration (e.g., CUDA), the client is responsible for:
1.  Providing a user interface to configure and trigger the analysis.
2.  Managing the state of the request (loading, error, success).
3.  Storing and caching the results.
4.  Visualizing the results by colour-coding nodes in the graph.

## UI Component: `ShortestPathControls.tsx`

The primary user interface for this feature is the `ShortestPathControls` component, located at [`client/src/features/analytics/components/ShortestPathControls.tsx`](../../client/src/features/analytics/components/ShortestPathControls.tsx).

### Responsibilities
-   **Configuration**: Allows the user to select a **Source Node** and the **Algorithm** to be used (e.g., Dijkstra, Bellman-Ford).
-   **Execution**: Provides a "Calculate" button to initiate the SSSP computation and a "Clear" button to reset the visualisation.
-   **Feedback**: Displays the loading state, progress, and any errors that occur during computation.
-   **Results Display**: Shows a detailed table of results, including the distance to each node from the source and the path taken. It also provides summary statistics like the number of reachable/unreachable nodes and computation time.

## State Management: `analyticsStore.ts`

The state for the analytics feature is managed by a dedicated Zustand store defined in [`client/src/features/analytics/store/analyticsStore.ts`](../../client/src/features/analytics/store/analyticsStore.ts).

### State (`AnalyticsState`)
-   `currentResult: SSSPResult | null`: Stores the most recent SSSP computation result.
-   `loading: boolean`: A flag indicating if a computation is in progress.
-   `error: string | null`: Stores any error message from the last computation attempt.
-   `cache: SSSPCache`: An in-memory cache to store results of previous computations, keyed by source node ID and a hash of the graph structure.
-   `metrics: AnalyticsMetrics`: Tracks performance metrics like cache hits/misses and average computation time.

### Actions
-   `computeSSSP(...)`: The core action that triggers the analysis. It first checks the cache for a valid result. If a cached result is not found, it proceeds to compute the shortest paths.
-   `clearResults()`: Resets `currentResult` and `error` to their initial states, effectively clearing the visualisation.
-   `normalizeDistances(...)`: A helper function to normalize the path distances to a 0-1 range, useful for visualisation.
-   `getUnreachableNodes(...)`: A helper to get a list of nodes that are not reachable from the source.

## API Interaction

While the client-side store contains TypeScript implementations of Dijkstra's and Bellman-Ford's algorithms (likely as a fallback or for smaller graphs), the primary interaction for large-scale, GPU-accelerated computation happens via the `apiService`.

The `computeSSSP` action in `analyticsStore.ts` is expected to call a method like `apiService.computeShortestPaths`, passing the graph data and configuration. The client then waits for the API response, which contains the computed distances and paths, before updating its state.

*Note: The direct API call is abstracted within the store's actions and may not be immediately visible in the component code.*

## visualisation

The results of the SSSP analysis are visualized directly on the graph. This is typically handled by `GraphManager.tsx`.

### Process
1.  **Subscription**: `GraphManager.tsx` subscribes to the `useAnalyticsStore`.
2.  **State Change**: When `currentResult` in the store is updated, `GraphManager.tsx` is notified.
3.  **colour-Coding**: The component iterates through the nodes and uses the distances from `currentResult.distances` to update their colours. The colour-coding scheme is typically:
    -   **Source Node**: A distinct colour (e.g., blue).
    -   **Reachable Nodes**: coloured based on their distance from the source, often using a gradient. Normalized distances are used to map the distance to a colour range.
    -   **Unreachable Nodes**: A specific colour (e.g., grey or red) to indicate they cannot be reached from the source.

This provides an intuitive and immediate visual representation of the shortest path analysis on the graph.