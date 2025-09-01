import { SettingsPath } from '../types/generated/settings';
import { isValidSettingsPath, getValueByPath } from './settingsHelpers';

/**
 * Settings validation utilities and schemas
 */

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  path: SettingsPath;
  message: string;
  expectedType?: string;
  actualType?: string;
  expectedRange?: [any, any];
  actualValue?: any;
}

export interface ValidationWarning {
  path: SettingsPath;
  message: string;
  suggestion?: string;
}

export type ValidationType = 
  | 'string' 
  | 'number' 
  | 'boolean' 
  | 'array' 
  | 'object' 
  | 'color'
  | 'url'
  | 'email'
  | 'integer'
  | 'float'
  | 'enum'
  | 'range';

export interface ValidationRule {
  type: ValidationType;
  required?: boolean;
  min?: number;
  max?: number;
  pattern?: RegExp;
  enum?: any[];
  customValidator?: (value: any) => boolean;
  errorMessage?: string;
}

/**
 * Settings validation schema
 * Maps settings paths to their validation rules
 */
export const SETTINGS_VALIDATION_SCHEMA: Record<string, ValidationRule> = {
  // System settings
  'system.debug': {
    type: 'boolean',
    required: false
  },
  'system.customBackendUrl': {
    type: 'url',
    required: false
  },
  'system.apiVersion': {
    type: 'string',
    required: false,
    pattern: /^v?\d+(\.\d+)*$/,
    errorMessage: 'API version must be in format vX.Y.Z or X.Y.Z'
  },
  
  // UI settings
  'ui.theme': {
    type: 'enum',
    enum: ['light', 'dark', 'auto', 'system'],
    required: true
  },
  'ui.fontSize': {
    type: 'enum',
    enum: ['small', 'medium', 'large', 'extra-large'],
    required: false
  },
  'ui.accentColor': {
    type: 'color',
    required: false
  },
  'ui.borderRadius': {
    type: 'number',
    min: 0,
    max: 50,
    required: false
  },
  
  // Performance settings
  'system.performance.frameRateLimit': {
    type: 'integer',
    min: 15,
    max: 240,
    required: false
  },
  'system.performance.memoryOptimization': {
    type: 'boolean',
    required: false
  },
  
  // WebSocket settings
  'system.websocket.reconnectAttempts': {
    type: 'integer',
    min: 0,
    max: 100,
    required: false
  },
  'system.websocket.reconnectInterval': {
    type: 'integer',
    min: 100,
    max: 60000,
    required: false
  },
  'system.websocket.bufferSize': {
    type: 'integer',
    min: 256,
    max: 65536,
    required: false
  },
  
  // Visualization settings
  'visualisation.quality': {
    type: 'enum',
    enum: ['low', 'medium', 'high', 'ultra'],
    required: false
  },
  'visualisation.camera.fov': {
    type: 'number',
    min: 10,
    max: 180,
    required: false
  },
  
  // Voice settings
  'kokoro.defaultSpeed': {
    type: 'number',
    min: 0.1,
    max: 3.0,
    required: false
  },
  'kokoro.stream': {
    type: 'boolean',
    required: false
  }
};

/**
 * Validates a single value against a validation rule
 * @param value - Value to validate
 * @param rule - Validation rule to apply
 * @param path - Settings path for error reporting
 * @returns Validation result
 */
export function validateValue(value: any, rule: ValidationRule, path: SettingsPath): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  
  // Check if required
  if (rule.required && (value === undefined || value === null)) {
    errors.push({
      path,
      message: rule.errorMessage || `${path} is required`
    });
    return { isValid: false, errors, warnings };
  }
  
  // Skip validation if value is undefined/null and not required
  if (value === undefined || value === null) {
    return { isValid: true, errors, warnings };
  }
  
  // Type validation
  if (!validateType(value, rule.type)) {
    errors.push({
      path,
      message: rule.errorMessage || `Expected ${rule.type}, got ${typeof value}`,
      expectedType: rule.type,
      actualType: typeof value,
      actualValue: value
    });
    return { isValid: false, errors, warnings };
  }
  
  // Enum validation
  if (rule.enum && !rule.enum.includes(value)) {
    errors.push({
      path,
      message: rule.errorMessage || `Value must be one of: ${rule.enum.join(', ')}`,
      actualValue: value
    });
    return { isValid: false, errors, warnings };
  }
  
  // Range validation for numbers
  if ((rule.type === 'number' || rule.type === 'integer' || rule.type === 'float') && typeof value === 'number') {
    if (rule.min !== undefined && value < rule.min) {
      errors.push({
        path,
        message: rule.errorMessage || `Value must be at least ${rule.min}`,
        expectedRange: [rule.min, rule.max || Infinity],
        actualValue: value
      });
    }
    
    if (rule.max !== undefined && value > rule.max) {
      errors.push({
        path,
        message: rule.errorMessage || `Value must be at most ${rule.max}`,
        expectedRange: [rule.min || -Infinity, rule.max],
        actualValue: value
      });
    }
  }
  
  // Pattern validation
  if (rule.pattern && typeof value === 'string') {
    if (!rule.pattern.test(value)) {
      errors.push({
        path,
        message: rule.errorMessage || `Value does not match required pattern`,
        actualValue: value
      });
    }
  }
  
  // Custom validator
  if (rule.customValidator && !rule.customValidator(value)) {
    errors.push({
      path,
      message: rule.errorMessage || `Value failed custom validation`,
      actualValue: value
    });
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Validates a complete settings object
 * @param settings - Settings object to validate
 * @param schema - Validation schema (defaults to built-in schema)
 * @returns Validation result
 */
export function validateSettings(
  settings: any, 
  schema: Record<string, ValidationRule> = SETTINGS_VALIDATION_SCHEMA
): ValidationResult {
  const allErrors: ValidationError[] = [];
  const allWarnings: ValidationWarning[] = [];
  
  // Validate each path in the schema
  for (const [pathStr, rule] of Object.entries(schema)) {
    if (!isValidSettingsPath(pathStr)) {
      allWarnings.push({
        path: pathStr as SettingsPath,
        message: 'Invalid settings path in validation schema',
        suggestion: 'Check path format'
      });
      continue;
    }
    
    const path = pathStr as SettingsPath;
    const value = getValueByPath(settings, path);
    const result = validateValue(value, rule, path);
    
    allErrors.push(...result.errors);
    allWarnings.push(...result.warnings);
  }
  
  return {
    isValid: allErrors.length === 0,
    errors: allErrors,
    warnings: allWarnings
  };
}

/**
 * Type validation helper
 * @param value - Value to check
 * @param type - Expected type
 * @returns True if value matches type
 */
function validateType(value: any, type: ValidationType): boolean {
  switch (type) {
    case 'string':
      return typeof value === 'string';
      
    case 'number':
    case 'float':
      return typeof value === 'number' && !isNaN(value);
      
    case 'integer':
      return typeof value === 'number' && Number.isInteger(value);
      
    case 'boolean':
      return typeof value === 'boolean';
      
    case 'array':
      return Array.isArray(value);
      
    case 'object':
      return value !== null && typeof value === 'object' && !Array.isArray(value);
      
    case 'color':
      return typeof value === 'string' && isValidColor(value);
      
    case 'url':
      return typeof value === 'string' && isValidUrl(value);
      
    case 'email':
      return typeof value === 'string' && isValidEmail(value);
      
    default:
      return true; // Unknown types pass validation
  }
}

/**
 * Validates a color string (hex, rgb, hsl, named colors)
 * @param color - Color string to validate
 * @returns True if valid color
 */
function isValidColor(color: string): boolean {
  // Hex colors
  if (/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(color)) {
    return true;
  }
  
  // RGB/RGBA
  if (/^rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(,\s*[\d.]+)?\s*\)$/.test(color)) {
    return true;
  }
  
  // HSL/HSLA
  if (/^hsla?\(\s*\d+\s*,\s*\d+%\s*,\s*\d+%\s*(,\s*[\d.]+)?\s*\)$/.test(color)) {
    return true;
  }
  
  // Named colors (basic set)
  const namedColors = [
    'black', 'white', 'red', 'green', 'blue', 'yellow', 'orange', 'purple',
    'pink', 'brown', 'gray', 'grey', 'transparent', 'currentColor'
  ];
  
  return namedColors.includes(color.toLowerCase());
}

/**
 * Validates a URL string
 * @param url - URL string to validate
 * @returns True if valid URL
 */
function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validates an email string
 * @param email - Email string to validate
 * @returns True if valid email
 */
function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Creates a validation rule for a specific path
 * @param type - Validation type
 * @param options - Additional validation options
 * @returns Validation rule
 */
export function createValidationRule(
  type: ValidationType,
  options: Partial<ValidationRule> = {}
): ValidationRule {
  return {
    type,
    ...options
  };
}

/**
 * Adds a validation rule to the schema
 * @param path - Settings path
 * @param rule - Validation rule
 */
export function addValidationRule(path: SettingsPath, rule: ValidationRule): void {
  SETTINGS_VALIDATION_SCHEMA[path] = rule;
}

/**
 * Removes a validation rule from the schema
 * @param path - Settings path
 */
export function removeValidationRule(path: SettingsPath): void {
  delete SETTINGS_VALIDATION_SCHEMA[path];
}

/**
 * Gets validation rule for a specific path
 * @param path - Settings path
 * @returns Validation rule or undefined
 */
export function getValidationRule(path: SettingsPath): ValidationRule | undefined {
  return SETTINGS_VALIDATION_SCHEMA[path];
}

/**
 * Validates a single setting by path
 * @param settings - Settings object
 * @param path - Settings path to validate
 * @returns Validation result
 */
export function validateSettingByPath(settings: any, path: SettingsPath): ValidationResult {
  const rule = getValidationRule(path);
  if (!rule) {
    return {
      isValid: true,
      errors: [],
      warnings: [{
        path,
        message: 'No validation rule found for this path',
        suggestion: 'Consider adding a validation rule'
      }]
    };
  }
  
  const value = getValueByPath(settings, path);
  return validateValue(value, rule, path);
}

/**
 * Formats validation errors for display
 * @param errors - Array of validation errors
 * @returns Formatted error messages
 */
export function formatValidationErrors(errors: ValidationError[]): string[] {
  return errors.map(error => {
    let message = `${error.path}: ${error.message}`;
    
    if (error.actualValue !== undefined) {
      message += ` (received: ${JSON.stringify(error.actualValue)})`;
    }
    
    if (error.expectedType && error.actualType) {
      message += ` [expected ${error.expectedType}, got ${error.actualType}]`;
    }
    
    return message;
  });
}

/**
 * Formats validation warnings for display
 * @param warnings - Array of validation warnings
 * @returns Formatted warning messages
 */
export function formatValidationWarnings(warnings: ValidationWarning[]): string[] {
  return warnings.map(warning => {
    let message = `${warning.path}: ${warning.message}`;
    
    if (warning.suggestion) {
      message += ` (${warning.suggestion})`;
    }
    
    return message;
  });
}