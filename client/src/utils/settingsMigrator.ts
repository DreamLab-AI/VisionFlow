import { SettingsPath } from '../types/generated/settings';
import { deepClone, setValueByPath, getValueByPath, deleteValueByPath } from './settingsHelpers';
import { validateSettings, ValidationResult } from './settingsValidator';

/**
 * Settings migration utilities for handling schema changes and upgrades
 */

export interface MigrationResult {
  success: boolean;
  fromVersion: string;
  toVersion: string;
  appliedMigrations: string[];
  errors: string[];
  warnings: string[];
  migratedSettings: any;
  validationResult?: ValidationResult;
}

export interface Migration {
  id: string;
  fromVersion: string;
  toVersion: string;
  description: string;
  up: (settings: any) => Promise<any>;
  down?: (settings: any) => Promise<any>;
  validate?: (settings: any) => boolean;
}

export interface MigrationConfig {
  validateAfterMigration?: boolean;
  backupBeforeMigration?: boolean;
  rollbackOnError?: boolean;
  skipValidation?: boolean;
}

/**
 * Migration registry for all available migrations
 */
const MIGRATIONS: Migration[] = [];

/**
 * Default migration configuration
 */
const DEFAULT_CONFIG: MigrationConfig = {
  validateAfterMigration: true,
  backupBeforeMigration: true,
  rollbackOnError: false,
  skipValidation: false
};

/**
 * Registers a new migration
 * @param migration - Migration to register
 */
export function registerMigration(migration: Migration): void {
  // Check if migration with same ID already exists
  const existingIndex = MIGRATIONS.findIndex(m => m.id === migration.id);
  if (existingIndex !== -1) {
    MIGRATIONS[existingIndex] = migration;
  } else {
    MIGRATIONS.push(migration);
  }
  
  // Sort migrations by version for proper ordering
  MIGRATIONS.sort((a, b) => compareVersions(a.fromVersion, b.fromVersion));
}

/**
 * Gets all registered migrations
 * @returns Array of all migrations
 */
export function getAllMigrations(): Migration[] {
  return [...MIGRATIONS];
}

/**
 * Gets migrations needed to go from one version to another
 * @param fromVersion - Starting version
 * @param toVersion - Target version
 * @returns Array of migrations to apply
 */
export function getMigrationsToApply(fromVersion: string, toVersion: string): Migration[] {
  if (compareVersions(fromVersion, toVersion) >= 0) {
    return []; // No migrations needed
  }
  
  const applicableMigrations = MIGRATIONS.filter(migration => {
    return compareVersions(migration.fromVersion, fromVersion) >= 0 &&
           compareVersions(migration.toVersion, toVersion) <= 0;
  });
  
  return applicableMigrations.sort((a, b) => compareVersions(a.fromVersion, b.fromVersion));
}

/**
 * Applies migrations to settings
 * @param settings - Settings object to migrate
 * @param targetVersion - Version to migrate to
 * @param config - Migration configuration
 * @returns Migration result
 */
export async function migrateSettings(
  settings: any,
  targetVersion: string,
  config: MigrationConfig = {}
): Promise<MigrationResult> {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  const currentVersion = getSettingsVersion(settings);
  const appliedMigrations: string[] = [];
  const errors: string[] = [];
  const warnings: string[] = [];
  
  let migratedSettings = finalConfig.backupBeforeMigration ? deepClone(settings) : settings;
  const originalSettings = deepClone(settings);
  
  try {
    // Get migrations to apply
    const migrations = getMigrationsToApply(currentVersion, targetVersion);
    
    if (migrations.length === 0) {
      return {
        success: true,
        fromVersion: currentVersion,
        toVersion: targetVersion,
        appliedMigrations: [],
        errors: [],
        warnings: ['No migrations needed'],
        migratedSettings
      };
    }
    
    // Apply each migration in sequence
    for (const migration of migrations) {
      try {
        // Validate pre-migration if specified
        if (migration.validate && !migration.validate(migratedSettings)) {
          throw new Error(`Pre-migration validation failed for ${migration.id}`);
        }
        
        // Apply migration
        migratedSettings = await migration.up(migratedSettings);
        appliedMigrations.push(migration.id);
        
        // Update version after successful migration
        setSettingsVersion(migratedSettings, migration.toVersion);
        
      } catch (error) {
        const errorMessage = `Migration ${migration.id} failed: ${error instanceof Error ? error.message : String(error)}`;
        errors.push(errorMessage);
        
        if (finalConfig.rollbackOnError) {
          // Attempt to rollback applied migrations
          try {
            migratedSettings = await rollbackMigrations(originalSettings, appliedMigrations);
            warnings.push('Rolled back due to migration failure');
          } catch (rollbackError) {
            errors.push(`Rollback failed: ${rollbackError instanceof Error ? rollbackError.message : String(rollbackError)}`);
          }
        }
        
        break; // Stop applying further migrations
      }
    }
    
    // Validate final result if requested
    let validationResult: ValidationResult | undefined;
    if (finalConfig.validateAfterMigration && !finalConfig.skipValidation) {
      validationResult = validateSettings(migratedSettings);
      if (!validationResult.isValid) {
        errors.push(`Post-migration validation failed: ${validationResult.errors.length} errors`);
        validationResult.errors.forEach(error => errors.push(`  - ${error.message}`));
      }
    }
    
    return {
      success: errors.length === 0,
      fromVersion: currentVersion,
      toVersion: getSettingsVersion(migratedSettings),
      appliedMigrations,
      errors,
      warnings,
      migratedSettings,
      validationResult
    };
    
  } catch (error) {
    return {
      success: false,
      fromVersion: currentVersion,
      toVersion: currentVersion,
      appliedMigrations,
      errors: [`Migration process failed: ${error instanceof Error ? error.message : String(error)}`],
      warnings,
      migratedSettings: originalSettings
    };
  }
}

/**
 * Rolls back applied migrations
 * @param originalSettings - Original settings before migration
 * @param appliedMigrationIds - IDs of migrations that were applied
 * @returns Rolled back settings
 */
async function rollbackMigrations(originalSettings: any, appliedMigrationIds: string[]): Promise<any> {
  // For now, just return original settings
  // In a more sophisticated implementation, you could apply down migrations
  return deepClone(originalSettings);
}

/**
 * Gets the current version from settings
 * @param settings - Settings object
 * @returns Current version string
 */
export function getSettingsVersion(settings: any): string {
  return getValueByPath(settings, 'system.version' as SettingsPath) || '1.0.0';
}

/**
 * Sets the version in settings
 * @param settings - Settings object
 * @param version - Version to set
 */
export function setSettingsVersion(settings: any, version: string): void {
  setValueByPath(settings, 'system.version' as SettingsPath, version);
}

/**
 * Compares two version strings
 * @param version1 - First version
 * @param version2 - Second version
 * @returns -1 if version1 < version2, 0 if equal, 1 if version1 > version2
 */
export function compareVersions(version1: string, version2: string): number {
  const v1parts = version1.split('.').map(Number);
  const v2parts = version2.split('.').map(Number);
  const maxLength = Math.max(v1parts.length, v2parts.length);
  
  for (let i = 0; i < maxLength; i++) {
    const v1part = v1parts[i] || 0;
    const v2part = v2parts[i] || 0;
    
    if (v1part < v2part) return -1;
    if (v1part > v2part) return 1;
  }
  
  return 0;
}

/**
 * Predefined migrations for common schema changes
 */

// Migration from 1.0.0 to 1.1.0: Add new UI settings
registerMigration({
  id: 'ui-settings-1.1.0',
  fromVersion: '1.0.0',
  toVersion: '1.1.0',
  description: 'Add new UI theme and accessibility settings',
  up: async (settings) => {
    const migrated = deepClone(settings);
    
    // Add default UI settings if they don't exist
    if (!getValueByPath(migrated, 'ui.theme' as SettingsPath)) {
      setValueByPath(migrated, 'ui.theme' as SettingsPath, 'system');
    }
    
    if (!getValueByPath(migrated, 'ui.fontSize' as SettingsPath)) {
      setValueByPath(migrated, 'ui.fontSize' as SettingsPath, 'medium');
    }
    
    if (!getValueByPath(migrated, 'ui.accessibility' as SettingsPath)) {
      setValueByPath(migrated, 'ui.accessibility.reducedMotion' as SettingsPath, false);
      setValueByPath(migrated, 'ui.accessibility.highContrast' as SettingsPath, false);
    }
    
    return migrated;
  },
  down: async (settings) => {
    const migrated = deepClone(settings);
    
    // Remove UI settings added in this migration
    deleteValueByPath(migrated, 'ui.theme' as SettingsPath);
    deleteValueByPath(migrated, 'ui.fontSize' as SettingsPath);
    deleteValueByPath(migrated, 'ui.accessibility' as SettingsPath);
    
    return migrated;
  }
});

// Migration from 1.1.0 to 1.2.0: Restructure performance settings
registerMigration({
  id: 'performance-restructure-1.2.0',
  fromVersion: '1.1.0',
  toVersion: '1.2.0',
  description: 'Restructure performance settings for better organization',
  up: async (settings) => {
    const migrated = deepClone(settings);
    
    // Move old performance settings to new structure
    const oldFrameRate = getValueByPath(migrated, 'system.frameRate' as SettingsPath);
    if (oldFrameRate !== undefined) {
      setValueByPath(migrated, 'system.performance.rendering.frameRate' as SettingsPath, oldFrameRate);
      deleteValueByPath(migrated, 'system.frameRate' as SettingsPath);
    }
    
    const oldMemoryOptimization = getValueByPath(migrated, 'system.memoryOptimization' as SettingsPath);
    if (oldMemoryOptimization !== undefined) {
      setValueByPath(migrated, 'system.performance.memoryOptimization' as SettingsPath, oldMemoryOptimization);
      deleteValueByPath(migrated, 'system.memoryOptimization' as SettingsPath);
    }
    
    // Add new performance settings with defaults
    if (!getValueByPath(migrated, 'system.performance.adaptive' as SettingsPath)) {
      setValueByPath(migrated, 'system.performance.adaptive.enabled' as SettingsPath, false);
      setValueByPath(migrated, 'system.performance.adaptive.targetFPS' as SettingsPath, 60);
    }
    
    return migrated;
  },
  down: async (settings) => {
    const migrated = deepClone(settings);
    
    // Move settings back to old structure
    const newFrameRate = getValueByPath(migrated, 'system.performance.rendering.frameRate' as SettingsPath);
    if (newFrameRate !== undefined) {
      setValueByPath(migrated, 'system.frameRate' as SettingsPath, newFrameRate);
    }
    
    const newMemoryOptimization = getValueByPath(migrated, 'system.performance.memoryOptimization' as SettingsPath);
    if (newMemoryOptimization !== undefined) {
      setValueByPath(migrated, 'system.memoryOptimization' as SettingsPath, newMemoryOptimization);
    }
    
    // Remove new performance structure
    deleteValueByPath(migrated, 'system.performance' as SettingsPath);
    
    return migrated;
  }
});

// Migration from 1.2.0 to 1.3.0: Add visualization settings
registerMigration({
  id: 'visualization-settings-1.3.0',
  fromVersion: '1.2.0',
  toVersion: '1.3.0',
  description: 'Add comprehensive visualization settings',
  up: async (settings) => {
    const migrated = deepClone(settings);
    
    // Add default visualization settings
    if (!getValueByPath(migrated, 'visualisation.quality' as SettingsPath)) {
      setValueByPath(migrated, 'visualisation.quality' as SettingsPath, 'medium');
      setValueByPath(migrated, 'visualisation.antialiasing' as SettingsPath, true);
      setValueByPath(migrated, 'visualisation.shadows' as SettingsPath, true);
    }
    
    // Add camera defaults
    if (!getValueByPath(migrated, 'visualisation.camera' as SettingsPath)) {
      setValueByPath(migrated, 'visualisation.camera.fov' as SettingsPath, 75);
      setValueByPath(migrated, 'visualisation.camera.near' as SettingsPath, 0.1);
      setValueByPath(migrated, 'visualisation.camera.far' as SettingsPath, 1000);
    }
    
    // Add post-processing defaults
    if (!getValueByPath(migrated, 'visualisation.postProcessing' as SettingsPath)) {
      setValueByPath(migrated, 'visualisation.postProcessing.bloom.enabled' as SettingsPath, false);
      setValueByPath(migrated, 'visualisation.postProcessing.bloom.intensity' as SettingsPath, 1.0);
      setValueByPath(migrated, 'visualisation.postProcessing.ssao.enabled' as SettingsPath, false);
      setValueByPath(migrated, 'visualisation.postProcessing.ssao.intensity' as SettingsPath, 0.5);
    }
    
    return migrated;
  }
});

/**
 * Utility functions for creating common migration patterns
 */

/**
 * Creates a migration that renames a setting path
 * @param id - Migration ID
 * @param version - Target version
 * @param oldPath - Old settings path
 * @param newPath - New settings path
 * @returns Migration object
 */
export function createRenameMigration(
  id: string,
  version: string,
  oldPath: SettingsPath,
  newPath: SettingsPath
): Migration {
  return {
    id,
    fromVersion: getPreviousVersion(version),
    toVersion: version,
    description: `Rename ${oldPath} to ${newPath}`,
    up: async (settings) => {
      const migrated = deepClone(settings);
      const value = getValueByPath(migrated, oldPath);
      
      if (value !== undefined) {
        setValueByPath(migrated, newPath, value);
        deleteValueByPath(migrated, oldPath);
      }
      
      return migrated;
    },
    down: async (settings) => {
      const migrated = deepClone(settings);
      const value = getValueByPath(migrated, newPath);
      
      if (value !== undefined) {
        setValueByPath(migrated, oldPath, value);
        deleteValueByPath(migrated, newPath);
      }
      
      return migrated;
    }
  };
}

/**
 * Creates a migration that adds a new setting with a default value
 * @param id - Migration ID
 * @param version - Target version
 * @param path - Settings path
 * @param defaultValue - Default value
 * @returns Migration object
 */
export function createAddSettingMigration(
  id: string,
  version: string,
  path: SettingsPath,
  defaultValue: any
): Migration {
  return {
    id,
    fromVersion: getPreviousVersion(version),
    toVersion: version,
    description: `Add ${path} with default value`,
    up: async (settings) => {
      const migrated = deepClone(settings);
      
      if (getValueByPath(migrated, path) === undefined) {
        setValueByPath(migrated, path, defaultValue);
      }
      
      return migrated;
    },
    down: async (settings) => {
      const migrated = deepClone(settings);
      deleteValueByPath(migrated, path);
      return migrated;
    }
  };
}

/**
 * Gets the previous version (simplistic implementation)
 * @param version - Current version
 * @returns Previous version
 */
function getPreviousVersion(version: string): string {
  const parts = version.split('.').map(Number);
  if (parts[2] > 0) {
    parts[2] -= 1;
  } else if (parts[1] > 0) {
    parts[1] -= 1;
    parts[2] = 9; // Assume max minor version of 9
  } else if (parts[0] > 1) {
    parts[0] -= 1;
    parts[1] = 9;
    parts[2] = 9;
  }
  
  return parts.join('.');
}