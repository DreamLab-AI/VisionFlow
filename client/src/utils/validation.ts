import { Vec3, BinaryNodeData } from '../types/binaryProtocol';
import { createLogger } from './loggerConfig';

const logger = createLogger('Validation');

export interface ValidationResult {
  valid: boolean;
  errors?: string[];
}

export interface ValidationConfig {
  maxNodes: number;
  maxCoordinate: number;
  minCoordinate: number;
  maxVelocity: number;
  allowNaN: boolean;
  allowInfinity: boolean;
}

const DEFAULT_VALIDATION_CONFIG: ValidationConfig = {
  maxNodes: 10000,
  maxCoordinate: 10000,
  minCoordinate: -10000,
  maxVelocity: 1000,
  allowNaN: false,
  allowInfinity: false
};

/**
 * Validate a Vec3 position or velocity
 */
export function validateVec3(
  vec: Vec3,
  fieldName: string,
  config: Partial<ValidationConfig> = {}
): ValidationResult {
  const cfg = { ...DEFAULT_VALIDATION_CONFIG, ...config };
  const errors: string[] = [];

  // Check for NaN
  if (!cfg.allowNaN) {
    if (isNaN(vec.x)) errors.push(`${fieldName}.x is NaN`);
    if (isNaN(vec.y)) errors.push(`${fieldName}.y is NaN`);
    if (isNaN(vec.z)) errors.push(`${fieldName}.z is NaN`);
  }

  // Check for Infinity
  if (!cfg.allowInfinity) {
    if (!isFinite(vec.x)) errors.push(`${fieldName}.x is not finite`);
    if (!isFinite(vec.y)) errors.push(`${fieldName}.y is not finite`);
    if (!isFinite(vec.z)) errors.push(`${fieldName}.z is not finite`);
  }

  // Check coordinate bounds
  if (vec.x < cfg.minCoordinate || vec.x > cfg.maxCoordinate) {
    errors.push(`${fieldName}.x out of bounds: ${vec.x}`);
  }
  if (vec.y < cfg.minCoordinate || vec.y > cfg.maxCoordinate) {
    errors.push(`${fieldName}.y out of bounds: ${vec.y}`);
  }
  if (vec.z < cfg.minCoordinate || vec.z > cfg.maxCoordinate) {
    errors.push(`${fieldName}.z out of bounds: ${vec.z}`);
  }

  return {
    valid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined
  };
}

/**
 * Validate velocity specific constraints
 */
export function validateVelocity(
  velocity: Vec3,
  config: Partial<ValidationConfig> = {}
): ValidationResult {
  const cfg = { ...DEFAULT_VALIDATION_CONFIG, ...config };
  const baseValidation = validateVec3(velocity, 'velocity', config);
  
  if (!baseValidation.valid) {
    return baseValidation;
  }

  const errors: string[] = [];
  const speed = Math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2);

  if (speed > cfg.maxVelocity) {
    errors.push(`Velocity magnitude ${speed} exceeds maximum ${cfg.maxVelocity}`);
  }

  return {
    valid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined
  };
}

/**
 * Validate a single BinaryNodeData object
 */
export function validateNodeData(
  node: BinaryNodeData,
  config: Partial<ValidationConfig> = {}
): ValidationResult {
  const errors: string[] = [];

  // Validate node ID
  if (node.nodeId < 0) {
    errors.push(`Invalid node ID: ${node.nodeId}`);
  }

  // Validate position
  const positionValidation = validateVec3(node.position, `node[${node.nodeId}].position`, config);
  if (!positionValidation.valid && positionValidation.errors) {
    errors.push(...positionValidation.errors);
  }

  // Validate velocity
  const velocityValidation = validateVelocity(node.velocity, config);
  if (!velocityValidation.valid && velocityValidation.errors) {
    errors.push(...velocityValidation.errors);
  }

  return {
    valid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined
  };
}

/**
 * Validate an array of node positions before sending
 */
export function validateNodePositions(
  nodes: BinaryNodeData[],
  config: Partial<ValidationConfig> = {}
): ValidationResult {
  const cfg = { ...DEFAULT_VALIDATION_CONFIG, ...config };
  const errors: string[] = [];

  // Check array size
  if (nodes.length > cfg.maxNodes) {
    errors.push(`Too many nodes: ${nodes.length} > ${cfg.maxNodes}`);
  }

  // Check for duplicates
  const seenIds = new Set<number>();
  const duplicates: number[] = [];
  
  nodes.forEach(node => {
    if (seenIds.has(node.nodeId)) {
      duplicates.push(node.nodeId);
    }
    seenIds.add(node.nodeId);
  });

  if (duplicates.length > 0) {
    errors.push(`Duplicate node IDs found: ${duplicates.join(', ')}`);
  }

  // Validate each node
  nodes.forEach((node, index) => {
    const nodeValidation = validateNodeData(node, config);
    if (!nodeValidation.valid && nodeValidation.errors) {
      errors.push(`Node at index ${index}: ${nodeValidation.errors.join('; ')}`);
    }
  });

  return {
    valid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined
  };
}

/**
 * Sanitize node data by clamping values to valid ranges
 */
export function sanitizeNodeData(
  node: BinaryNodeData,
  config: Partial<ValidationConfig> = {}
): BinaryNodeData {
  const cfg = { ...DEFAULT_VALIDATION_CONFIG, ...config };

  const clampValue = (value: number, min: number, max: number): number => {
    if (isNaN(value) || !isFinite(value)) return 0;
    return Math.max(min, Math.min(max, value));
  };

  const clampVec3 = (vec: Vec3): Vec3 => ({
    x: clampValue(vec.x, cfg.minCoordinate, cfg.maxCoordinate),
    y: clampValue(vec.y, cfg.minCoordinate, cfg.maxCoordinate),
    z: clampValue(vec.z, cfg.minCoordinate, cfg.maxCoordinate)
  });

  // Clamp velocity to max speed
  const clampVelocity = (vel: Vec3): Vec3 => {
    const speed = Math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2);
    if (speed > cfg.maxVelocity) {
      const scale = cfg.maxVelocity / speed;
      return {
        x: vel.x * scale,
        y: vel.y * scale,
        z: vel.z * scale
      };
    }
    return clampVec3(vel);
  };

  return {
    nodeId: Math.max(0, node.nodeId),
    position: clampVec3(node.position),
    velocity: clampVelocity(node.velocity)
  };
}

/**
 * Batch validate and sanitize node data
 */
export function validateAndSanitizeBatch(
  nodes: BinaryNodeData[],
  config: Partial<ValidationConfig> = {}
): { valid: BinaryNodeData[]; invalid: Array<{ node: BinaryNodeData; errors: string[] }> } {
  const valid: BinaryNodeData[] = [];
  const invalid: Array<{ node: BinaryNodeData; errors: string[] }> = [];

  nodes.forEach(node => {
    const validation = validateNodeData(node, config);
    if (validation.valid) {
      valid.push(node);
    } else {
      // Try to sanitize invalid nodes
      const sanitized = sanitizeNodeData(node, config);
      const revalidation = validateNodeData(sanitized, config);
      
      if (revalidation.valid) {
        valid.push(sanitized);
        logger.debug(`Sanitized invalid node ${node.nodeId}`);
      } else {
        invalid.push({
          node,
          errors: validation.errors || []
        });
      }
    }
  });

  if (invalid.length > 0) {
    logger.warn(`${invalid.length} nodes failed validation after sanitization`);
  }

  return { valid, invalid };
}

/**
 * Create a validation middleware for batch processing
 */
export function createValidationMiddleware(config: Partial<ValidationConfig> = {}) {
  return (nodes: BinaryNodeData[]): BinaryNodeData[] => {
    const { valid, invalid } = validateAndSanitizeBatch(nodes, config);
    
    if (invalid.length > 0) {
      logger.error(`Dropped ${invalid.length} invalid nodes during validation`);
      invalid.forEach(({ node, errors }) => {
        logger.debug(`Node ${node.nodeId} validation errors:`, errors);
      });
    }

    return valid;
  };
}

/**
 * Validate WebSocket message structure
 */
export function validateWebSocketMessage(message: any): boolean {
  if (!message || typeof message !== 'object') {
    return false;
  }

  if (!message.type || typeof message.type !== 'string') {
    return false;
  }

  // Type-specific validation
  switch (message.type) {
    case 'node_position_update':
      return Array.isArray(message.data) && message.data.length > 0;
    
    case 'settings_update':
      return message.data && typeof message.data === 'object';
    
    case 'error':
      return typeof message.message === 'string';
    
    default:
      // Unknown message types are allowed but logged
      logger.debug(`Unknown message type: ${message.type}`);
      return true;
  }
}