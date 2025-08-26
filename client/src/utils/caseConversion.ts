/**
 * Utility functions for converting between different case styles (snake_case, camelCase)
 * and handling bloom/glow field mapping between client and server
 * Used primarily for API communication where server uses snake_case and client uses camelCase
 * Also handles the bloom->glow field mapping where client uses 'bloom' but server uses 'glow'
 */

/**
 * Converts a snake_case string to camelCase
 * @param str The snake_case string to convert
 * @returns The camelCase version of the string
 */
export function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

/**
 * Converts a camelCase string to snake_case
 * @param str The camelCase string to convert
 * @returns The snake_case version of the string
 */
export function camelToSnake(str: string): string {
  return str.replace(/([A-Z])/g, (match, letter) => `_${letter.toLowerCase()}`)
    .replace(/^_/, ''); // Remove leading underscore if present
}

/**
 * Recursively converts all keys in an object from snake_case to camelCase
 * @param obj The object with snake_case keys
 * @returns A new object with all keys converted to camelCase
 */
export function convertSnakeToCamelCase<T extends Record<string, any>>(obj: T): Record<string, any> {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => convertSnakeToCamelCase(item)) as any;
  }

  return Object.keys(obj).reduce((result, key) => {
    const camelKey = snakeToCamel(key);
    const value = obj[key];

    result[camelKey] = typeof value === 'object' && value !== null
      ? convertSnakeToCamelCase(value)
      : value;

    return result;
  }, {} as Record<string, any>);
}

/**
 * Recursively converts all keys in an object from camelCase to snake_case
 * @param obj The object with camelCase keys
 * @returns A new object with all keys converted to snake_case
 */
export function convertCamelToSnakeCase<T extends Record<string, any>>(obj: T): Record<string, any> {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => convertCamelToSnakeCase(item)) as any;
  }

  return Object.keys(obj).reduce((result, key) => {
    const snakeKey = camelToSnake(key);
    const value = obj[key];

    result[snakeKey] = typeof value === 'object' && value !== null
      ? convertCamelToSnakeCase(value)
      : value;

    return result;
  }, {} as Record<string, any>);
}

/**
 * Transforms bloom settings to glow settings for server compatibility
 * The server internally uses 'glow' but client may use 'bloom' for backward compatibility
 * @param settings The settings object potentially containing bloom fields
 * @returns Settings object with bloom fields mapped to glow
 */
export function transformBloomToGlow(settings: any): any {
  if (!settings || typeof settings !== 'object') {
    return settings;
  }

  const result = { ...settings };
  
  // Handle visualisation.bloom -> visualisation.glow transformation
  if (result.visualisation?.bloom && !result.visualisation?.glow) {
    result.visualisation.glow = {
      ...result.visualisation.bloom,
      // Map bloom-specific fields to glow equivalents
      nodeGlowStrength: result.visualisation.bloom.nodeBloomStrength,
      edgeGlowStrength: result.visualisation.bloom.edgeBloomStrength,
      environmentGlowStrength: result.visualisation.bloom.environmentBloomStrength,
      // Map common fields
      intensity: result.visualisation.bloom.strength || result.visualisation.bloom.intensity,
    };
    
    // Remove bloom-specific fields that have been mapped
    const { nodeBloomStrength, edgeBloomStrength, environmentBloomStrength, strength, ...remainingBloom } = result.visualisation.bloom;
    
    // Keep remaining bloom fields for compatibility but prioritize glow
    if (Object.keys(remainingBloom).length > 0) {
      result.visualisation.bloom = remainingBloom;
    } else {
      delete result.visualisation.bloom;
    }
  }
  
  // Handle nested objects recursively
  Object.keys(result).forEach(key => {
    if (typeof result[key] === 'object' && result[key] !== null && !Array.isArray(result[key])) {
      result[key] = transformBloomToGlow(result[key]);
    }
  });
  
  return result;
}

/**
 * Transforms glow settings to bloom settings for client compatibility
 * Ensures the client can work with bloom field names while server uses glow
 * @param settings The settings object containing glow fields
 * @returns Settings object with glow fields mapped to bloom for client use
 */
export function transformGlowToBloom(settings: any): any {
  if (!settings || typeof settings !== 'object') {
    return settings;
  }

  const result = { ...settings };
  
  // Handle visualisation.glow -> visualisation.bloom transformation for client compatibility
  if (result.visualisation?.glow) {
    // Create bloom object for client compatibility
    result.visualisation.bloom = {
      ...result.visualisation.bloom, // Preserve existing bloom if any
      enabled: result.visualisation.glow.enabled,
      // Map glow fields to bloom equivalents
      strength: result.visualisation.glow.intensity,
      nodeBloomStrength: result.visualisation.glow.nodeGlowStrength,
      edgeBloomStrength: result.visualisation.glow.edgeGlowStrength,
      environmentBloomStrength: result.visualisation.glow.environmentGlowStrength,
      // Copy other compatible fields
      radius: result.visualisation.glow.radius,
      threshold: result.visualisation.glow.threshold,
    };
    
    // Keep glow settings as primary (server uses these)
    // Both bloom and glow will exist for full compatibility
  }
  
  // Handle nested objects recursively
  Object.keys(result).forEach(key => {
    if (typeof result[key] === 'object' && result[key] !== null && !Array.isArray(result[key])) {
      result[key] = transformGlowToBloom(result[key]);
    }
  });
  
  return result;
}

/**
 * Normalizes settings by ensuring proper bloom/glow field mapping
 * This function ensures both client and server compatibility by maintaining both field sets
 * @param settings Raw settings object
 * @param direction 'toServer' transforms bloom->glow, 'toClient' transforms glow->bloom
 * @returns Normalized settings object
 */
export function normalizeBloomGlowSettings(settings: any, direction: 'toServer' | 'toClient' = 'toClient'): any {
  if (!settings) return settings;
  
  return direction === 'toServer' 
    ? transformBloomToGlow(settings)
    : transformGlowToBloom(settings);
}
