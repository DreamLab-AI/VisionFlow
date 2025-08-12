/**
 * Debug settings UI definition for client-side only debug options
 * These settings use localStorage and do not sync with the backend
 */

export const debugSettingsUIDefinition = {
  // Developer section - uses localStorage keys directly
  developer: {
    label: 'Developer',
    icon: 'Code',
    subsections: {
      debugging: {
        label: 'Debugging Tools',
        settings: {
          consoleLogging: { 
            label: 'Console Logging', 
            type: 'toggle', 
            path: 'debug.consoleLogging',  // localStorage key
            description: 'Enable console debug logs.',
            localStorage: true  // Flag to indicate this uses localStorage
          },
          logLevel: { 
            label: 'Log Level', 
            type: 'select', 
            options: [
              {value: 'error', label: 'Error'}, 
              {value: 'warn', label: 'Warning'}, 
              {value: 'info', label: 'Info'}, 
              {value: 'debug', label: 'Debug'}
            ], 
            path: 'debug.logLevel',
            description: 'Minimum log level to display.',
            localStorage: true
          },
          showNodeIds: { 
            label: 'Show Node IDs', 
            type: 'toggle', 
            path: 'debug.showNodeIds',
            description: 'Display node IDs in visualization.',
            localStorage: true
          },
          showEdgeWeights: { 
            label: 'Show Edge Weights', 
            type: 'toggle', 
            path: 'debug.showEdgeWeights',
            description: 'Display edge weight values.',
            localStorage: true
          },
          enableProfiler: { 
            label: 'Enable Profiler', 
            type: 'toggle', 
            path: 'debug.enableProfiler',
            description: 'Enable performance profiling.',
            localStorage: true
          },
          apiDebugMode: { 
            label: 'API Debug Mode', 
            type: 'toggle', 
            path: 'debug.apiDebugMode',
            description: 'Log all API requests and responses.',
            localStorage: true
          },
        },
      },
      clientDebug: {
        label: 'Client Debug Options',
        settings: {
          enabled: { 
            label: 'Enable Client Debug Mode', 
            type: 'toggle', 
            path: 'debug.enabled',
            description: 'Master switch for client-side debug features.',
            localStorage: true
          },
          dataDebug: { 
            label: 'Enable Data Debug', 
            type: 'toggle', 
            path: 'debug.data',
            description: 'Log detailed client data flow information.',
            localStorage: true,
            isAdvanced: true
          },
          performanceDebug: { 
            label: 'Enable Performance Debug', 
            type: 'toggle', 
            path: 'debug.performance',
            description: 'Show performance metrics overlay.',
            localStorage: true,
            isAdvanced: true
          },
          enableWebsocketDebug: { 
            label: 'Enable WebSocket Debug', 
            type: 'toggle', 
            path: 'debug.enableWebsocketDebug',
            description: 'Log WebSocket communication details.',
            localStorage: true,
            isAdvanced: true
          },
          logBinaryHeaders: { 
            label: 'Log Binary Headers', 
            type: 'toggle', 
            path: 'debug.logBinaryHeaders',
            description: 'Log headers of binary messages.',
            localStorage: true,
            isAdvanced: true
          },
          logFullJson: { 
            label: 'Log Full JSON', 
            type: 'toggle', 
            path: 'debug.logFullJson',
            description: 'Log complete JSON payloads.',
            localStorage: true,
            isAdvanced: true
          },
          enablePhysicsDebug: { 
            label: 'Enable Physics Debug', 
            type: 'toggle', 
            path: 'debug.enablePhysicsDebug',
            description: 'Show physics debug visualizations.',
            localStorage: true,
            isAdvanced: true
          },
          enableNodeDebug: { 
            label: 'Enable Node Debug', 
            type: 'toggle', 
            path: 'debug.enableNodeDebug',
            description: 'Enable debug features for nodes.',
            localStorage: true,
            isAdvanced: true
          },
          enableShaderDebug: { 
            label: 'Enable Shader Debug', 
            type: 'toggle', 
            path: 'debug.enableShaderDebug',
            description: 'Enable shader debugging tools.',
            localStorage: true,
            isAdvanced: true,
            isPowerUserOnly: true
          },
          enableMatrixDebug: { 
            label: 'Enable Matrix Debug', 
            type: 'toggle', 
            path: 'debug.enableMatrixDebug',
            description: 'Log matrix transformations.',
            localStorage: true,
            isAdvanced: true,
            isPowerUserOnly: true
          },
        },
      },
    },
  },
};