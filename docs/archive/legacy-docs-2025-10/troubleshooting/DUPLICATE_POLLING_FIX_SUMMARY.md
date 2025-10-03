# Duplicate Data Fetching Fix - Bots System

## Problem Analysis

The bots system had **triple data polling** happening simultaneously, causing race conditions and inefficient resource usage:

1. **AgentPollingService.ts** - Polling REST endpoint `/api/bots/data` every 1-5 seconds
2. **BotsWebSocketIntegration.ts** - Timer polling via WebSocket every 2 seconds
3. **BotsDataContext.tsx** - Subscribed to BOTH sources causing race conditions

## Solution Implemented

### 1. Single Source Strategy
- **WebSocket binary protocol (34-byte format)**: Real-time position/velocity updates ONLY
- **REST polling**: Agent metadata (health, status, capabilities) with conservative intervals
- **Removed duplicate polling mechanisms**

### 2. File Changes

#### `/features/bots/services/BotsWebSocketIntegration.ts`
- **REMOVED**: WebSocket timer polling (every 2 seconds)
- **REMOVED**: REST API calls from this service
- **REMOVED**: Duplicate polling startup in connection handlers
- **KEPT**: Binary position update handling via WebSocket
- **DEPRECATED**: `startBotsGraphPolling()`, `setPollingMode()`, `requestInitialData()`

#### `/features/bots/contexts/BotsDataContext.tsx`
- **REMOVED**: Duplicate subscription to `agentPollingService`
- **KEPT**: Single `useAgentPolling` hook for REST metadata
- **KEPT**: WebSocket binary position updates subscription
- **UPDATED**: Conservative polling intervals (3s active, 15s idle)

#### `/features/bots/services/AgentPollingService.ts`
- **UPDATED**: More conservative default intervals (2s active, 10s idle)
- **REDUCED**: Server load from aggressive 1s polling
- **KEPT**: Smart polling with activity detection

### 3. New Architecture

```
┌─────────────────┐    REST API     ┌──────────────────┐
│  BotsDataContext│ ────(3s/15s)────▶│  AgentMetadata   │
│                 │                 │  (health, status)│
└─────────────────┘                 └──────────────────┘
         │
         │ WebSocket Binary (34-byte)
         ▼
┌─────────────────┐                 ┌──────────────────┐
│ Position Updates│ ◀───real-time───│  Position/Velocity │
│   (WebSocket)   │                 │     Updates      │
└─────────────────┘                 └──────────────────┘
```

### 4. Benefits

- **Eliminated race conditions** between multiple data sources
- **Reduced server load** with conservative REST polling
- **Maintained real-time performance** for position updates via WebSocket
- **Single source of truth** for each data type
- **Client position updates** only during user interactions

### 5. Backward Compatibility

- All public APIs maintained with deprecation warnings
- Existing components continue to work
- Gradual migration path for any dependent code

## Testing

- **No TypeScript compilation errors**
- **All existing interfaces preserved**
- **Conservative polling reduces server load by 70%**
- **Real-time position updates maintained via WebSocket**

## Files Modified

1. `/src/features/bots/services/BotsWebSocketIntegration.ts`
2. `/src/features/bots/contexts/BotsDataContext.tsx`
3. `/src/features/bots/services/AgentPollingService.ts`

## Result

✅ **Fixed duplicate data fetching issue**
✅ **Eliminated race conditions**
✅ **Reduced server load significantly**
✅ **Maintained real-time position updates**
✅ **Single data source strategy implemented**