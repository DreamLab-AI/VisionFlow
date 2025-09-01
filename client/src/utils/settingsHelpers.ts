import { SettingsPath } from '../types/generated/settings';

/**
 * Utility functions for working with settings paths and values
 */

/**
 * Checks if a settings path is valid
 * @param path - The settings path to validate
 * @returns True if the path is valid
 */
export function isValidSettingsPath(path: string): path is SettingsPath {
  // Basic path validation - checks for proper dot notation
  if (!path || typeof path !== 'string') {
    return false;
  }
  
  // Must not start or end with dots
  if (path.startsWith('.') || path.endsWith('.')) {
    return false;
  }
  
  // Must not have consecutive dots
  if (path.includes('..')) {
    return false;
  }
  
  // Must contain only valid characters
  const validPathRegex = /^[a-zA-Z0-9._-]+$/;
  return validPathRegex.test(path);
}

/**
 * Splits a settings path into its component parts
 * @param path - The settings path to split
 * @returns Array of path components
 */
export function splitSettingsPath(path: SettingsPath): string[] {
  return path.split('.');
}

/**
 * Joins path components into a settings path
 * @param components - Array of path components
 * @returns Joined settings path
 */
export function joinSettingsPath(components: string[]): SettingsPath {
  const joined = components.join('.');
  if (!isValidSettingsPath(joined)) {
    throw new Error(`Invalid settings path: ${joined}`);
  }
  return joined;
}

/**
 * Gets the parent path of a settings path
 * @param path - The settings path
 * @returns Parent path or null if at root
 */
export function getParentPath(path: SettingsPath): SettingsPath | null {
  const components = splitSettingsPath(path);
  if (components.length <= 1) {
    return null;
  }
  return joinSettingsPath(components.slice(0, -1));
}

/**
 * Gets the leaf key of a settings path
 * @param path - The settings path
 * @returns The last component of the path
 */
export function getLeafKey(path: SettingsPath): string {
  const components = splitSettingsPath(path);
  return components[components.length - 1];
}

/**
 * Checks if one path is a descendant of another
 * @param descendant - The potential descendant path
 * @param ancestor - The potential ancestor path
 * @returns True if descendant is under ancestor
 */
export function isDescendantPath(descendant: SettingsPath, ancestor: SettingsPath): boolean {
  return descendant.startsWith(ancestor + '.');
}

/**
 * Checks if one path is a direct child of another
 * @param child - The potential child path
 * @param parent - The potential parent path
 * @returns True if child is a direct child of parent
 */
export function isDirectChild(child: SettingsPath, parent: SettingsPath): boolean {
  if (!isDescendantPath(child, parent)) {
    return false;
  }
  
  const childComponents = splitSettingsPath(child);
  const parentComponents = splitSettingsPath(parent);
  
  return childComponents.length === parentComponents.length + 1;
}

/**
 * Gets all possible parent paths for a given path
 * @param path - The settings path
 * @returns Array of parent paths from immediate to root
 */
export function getAllParentPaths(path: SettingsPath): SettingsPath[] {
  const components = splitSettingsPath(path);
  const parents: SettingsPath[] = [];
  
  for (let i = components.length - 1; i > 0; i--) {
    parents.push(joinSettingsPath(components.slice(0, i)));
  }
  
  return parents;
}

/**
 * Normalizes a settings path by removing redundant segments
 * @param path - The path to normalize
 * @returns Normalized path
 */
export function normalizeSettingsPath(path: string): SettingsPath {
  // Remove leading/trailing dots and normalize multiple consecutive dots
  const normalized = path
    .replace(/^\.+|\.+$/g, '') // Remove leading/trailing dots
    .replace(/\.{2,}/g, '.'); // Replace multiple dots with single dot
  
  if (!isValidSettingsPath(normalized)) {
    throw new Error(`Cannot normalize invalid settings path: ${path}`);
  }
  
  return normalized;
}

/**
 * Deep gets a value from an object using a settings path
 * @param obj - The object to traverse
 * @param path - The settings path
 * @returns The value at the path or undefined
 */
export function getValueByPath(obj: any, path: SettingsPath): any {
  if (!obj || typeof obj !== 'object') {
    return undefined;
  }
  
  const components = splitSettingsPath(path);
  let current = obj;
  
  for (const component of components) {
    if (current === null || current === undefined) {
      return undefined;
    }
    
    if (typeof current !== 'object') {
      return undefined;
    }
    
    current = current[component];
  }
  
  return current;
}

/**
 * Deep sets a value in an object using a settings path
 * @param obj - The object to modify
 * @param path - The settings path
 * @param value - The value to set
 * @returns The modified object
 */
export function setValueByPath(obj: any, path: SettingsPath, value: any): any {
  if (!obj || typeof obj !== 'object') {
    obj = {};
  }
  
  const components = splitSettingsPath(path);
  let current = obj;
  
  // Navigate to the parent of the target property
  for (let i = 0; i < components.length - 1; i++) {
    const component = components[i];
    
    if (!(component in current) || typeof current[component] !== 'object') {
      current[component] = {};
    }
    
    current = current[component];
  }
  
  // Set the final value
  const leafKey = components[components.length - 1];
  current[leafKey] = value;
  
  return obj;
}

/**
 * Deletes a value from an object using a settings path
 * @param obj - The object to modify
 * @param path - The settings path
 * @returns True if the deletion was successful
 */
export function deleteValueByPath(obj: any, path: SettingsPath): boolean {
  if (!obj || typeof obj !== 'object') {
    return false;
  }
  
  const components = splitSettingsPath(path);
  let current = obj;
  
  // Navigate to the parent of the target property
  for (let i = 0; i < components.length - 1; i++) {
    const component = components[i];
    
    if (!(component in current) || typeof current[component] !== 'object') {
      return false; // Path doesn't exist
    }
    
    current = current[component];
  }
  
  // Delete the final property
  const leafKey = components[components.length - 1];
  if (leafKey in current) {
    delete current[leafKey];
    return true;
  }
  
  return false;
}

/**
 * Checks if a path exists in an object
 * @param obj - The object to check
 * @param path - The settings path
 * @returns True if the path exists
 */
export function pathExists(obj: any, path: SettingsPath): boolean {
  return getValueByPath(obj, path) !== undefined;
}

/**
 * Gets all paths that exist in an object (flattened)
 * @param obj - The object to flatten
 * @param prefix - Optional prefix for paths
 * @returns Array of all existing paths
 */
export function getAllPaths(obj: any, prefix: string = ''): SettingsPath[] {
  const paths: SettingsPath[] = [];
  
  if (!obj || typeof obj !== 'object') {
    return paths;
  }
  
  for (const [key, value] of Object.entries(obj)) {
    const currentPath = prefix ? `${prefix}.${key}` : key;
    
    if (isValidSettingsPath(currentPath)) {
      paths.push(currentPath);
      
      // Recursively get nested paths
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        paths.push(...getAllPaths(value, currentPath));
      }
    }
  }
  
  return paths;
}

/**
 * Compares two objects and returns the paths that differ
 * @param obj1 - First object
 * @param obj2 - Second object
 * @returns Array of paths that are different between objects
 */
export function getChangedPaths(obj1: any, obj2: any): SettingsPath[] {
  const allPaths1 = getAllPaths(obj1);
  const allPaths2 = getAllPaths(obj2);
  const allUniquePaths = [...new Set([...allPaths1, ...allPaths2])];
  
  const changedPaths: SettingsPath[] = [];
  
  for (const path of allUniquePaths) {
    const value1 = getValueByPath(obj1, path);
    const value2 = getValueByPath(obj2, path);
    
    if (!deepEqual(value1, value2)) {
      changedPaths.push(path);
    }
  }
  
  return changedPaths;
}

/**
 * Deep equality comparison for values
 * @param a - First value
 * @param b - Second value
 * @returns True if values are deeply equal
 */
function deepEqual(a: any, b: any): boolean {
  if (a === b) {
    return true;
  }
  
  if (a == null || b == null) {
    return a === b;
  }
  
  if (typeof a !== typeof b) {
    return false;
  }
  
  if (typeof a !== 'object') {
    return false;
  }
  
  if (Array.isArray(a) !== Array.isArray(b)) {
    return false;
  }
  
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  
  if (keysA.length !== keysB.length) {
    return false;
  }
  
  for (const key of keysA) {
    if (!keysB.includes(key)) {
      return false;
    }
    
    if (!deepEqual(a[key], b[key])) {
      return false;
    }
  }
  
  return true;
}

/**
 * Creates a deep clone of an object
 * @param obj - Object to clone
 * @returns Deep clone of the object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  
  if (obj instanceof Date) {
    return new Date(obj.getTime()) as any;
  }
  
  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as any;
  }
  
  if (typeof obj === 'object') {
    const cloned: any = {};
    for (const [key, value] of Object.entries(obj)) {
      cloned[key] = deepClone(value);
    }
    return cloned;
  }
  
  return obj;
}

/**
 * Merges two settings objects deeply
 * @param target - Target object to merge into
 * @param source - Source object to merge from
 * @returns Merged object
 */
export function deepMerge<T>(target: T, source: Partial<T>): T {
  const result = deepClone(target);
  
  if (!source || typeof source !== 'object') {
    return result;
  }
  
  for (const [key, value] of Object.entries(source)) {
    if (value === undefined) {
      continue;
    }
    
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      if (!result[key as keyof T] || typeof result[key as keyof T] !== 'object') {
        (result as any)[key] = {};
      }
      (result as any)[key] = deepMerge((result as any)[key], value);
    } else {
      (result as any)[key] = deepClone(value);
    }
  }
  
  return result;
}