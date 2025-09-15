/**
 * Configuration for secure iframe communication
 */

export const IFRAME_COMMUNICATION_CONFIG = {
  // Allowed origins for postMessage communication
  // In production, this should be restricted to specific domains
  allowedOrigins: [
    'https://narrativegoldmine.com',
    'https://www.narrativegoldmine.com',
    // Add other trusted origins here
  ],
  
  // Target origin for sending messages - SECURITY FIX: Use specific origin
  // In development, use current origin; in production, use specific domain
  targetOrigin: process.env.NODE_ENV === 'production'
    ? 'https://narrativegoldmine.com'
    : window.location.origin, // Safe same-origin in development
  
  // Message action types
  messageActions: {
    NAVIGATE: 'navigate',
    UPDATE_NODE: 'updateNode',
    SYNC_STATE: 'syncState',
  } as const,
  
  // Validation settings
  validation: {
    requireValidUrl: true,
    requireKnownDomain: true,
    logMessages: true, // Set to false in production
  }
};

export type MessageAction = typeof IFRAME_COMMUNICATION_CONFIG.messageActions[keyof typeof IFRAME_COMMUNICATION_CONFIG.messageActions];