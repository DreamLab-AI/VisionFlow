import React, { createContext, useContext, useEffect, useRef, ReactNode } from 'react';
import { useSettingsStore } from '@/store/settingsStore';
import { SettingsPath } from '@/types/generated/settings';
import { validateSettings, ValidationResult } from '@/utils/settingsValidator';
import { migrateSettings, MigrationResult } from '@/utils/settingsMigrator';

/**
 * Settings Provider Context Interface
 */
interface SettingsProviderContextValue {
  // Store access
  store: ReturnType<typeof useSettingsStore>;
  
  // Validation
  validate: () => ValidationResult;
  validatePath: (path: SettingsPath) => ValidationResult;
  
  // Migration
  migrate: (targetVersion: string) => Promise<MigrationResult>;
  
  // Utils
  isLoading: boolean;
  isInitialized: boolean;
  lastError: string | null;
  
  // Events
  onSettingChange: (path: SettingsPath, callback: (value: any) => void) => () => void;
  onValidationError: (callback: (errors: ValidationResult) => void) => () => void;
  onMigrationComplete: (callback: (result: MigrationResult) => void) => () => void;
}

/**
 * Settings Provider Context
 */
const SettingsProviderContext = createContext<SettingsProviderContextValue | null>(null);

/**
 * Settings Provider Props
 */
interface SettingsProviderProps {
  children: ReactNode;
  autoValidate?: boolean;
  autoMigrate?: boolean;
  targetVersion?: string;
  onInitialized?: () => void;
  onError?: (error: string) => void;
}

/**
 * Settings Provider Component
 * Provides settings context and advanced functionality to child components
 */
export function SettingsProvider({
  children,
  autoValidate = false,
  autoMigrate = false,
  targetVersion = '1.0.0',
  onInitialized,
  onError
}: SettingsProviderProps) {
  const store = useSettingsStore();
  const [isLoading, setIsLoading] = React.useState(true);
  const [isInitialized, setIsInitialized] = React.useState(false);
  const [lastError, setLastError] = React.useState<string | null>(null);
  
  // Event listeners management
  const settingChangeListeners = useRef<Map<string, Set<(value: any) => void>>>(new Map());
  const validationErrorListeners = useRef<Set<(errors: ValidationResult) => void>>(new Set());
  const migrationCompleteListeners = useRef<Set<(result: MigrationResult) => void>>(new Set());
  
  // Initialize settings provider
  useEffect(() => {
    const initialize = async () => {
      try {
        setIsLoading(true);
        setLastError(null);
        
        // Wait for store to be ready
        if (!store.isInitialized) {
          // If store has an initialization method, wait for it
          if (typeof store.initialize === 'function') {
            await store.initialize();
          }
        }
        
        // Auto-migrate if requested
        if (autoMigrate) {
          try {
            const migrationResult = await migrate(targetVersion);
            
            // Notify migration listeners
            migrationCompleteListeners.current.forEach(listener => {
              listener(migrationResult);
            });
            
            if (!migrationResult.success) {
              throw new Error(`Migration failed: ${migrationResult.errors.join(', ')}`);
            }
          } catch (migrationError) {\n            const errorMessage = `Migration failed: ${migrationError instanceof Error ? migrationError.message : String(migrationError)}`;\n            setLastError(errorMessage);\n            onError?.(errorMessage);\n          }\n        }\n        \n        // Auto-validate if requested\n        if (autoValidate) {\n          try {\n            const validationResult = validate();\n            \n            if (!validationResult.isValid) {\n              // Notify validation error listeners\n              validationErrorListeners.current.forEach(listener => {\n                listener(validationResult);\n              });\n              \n              const errorMessage = `Settings validation failed: ${validationResult.errors.length} errors`;\n              setLastError(errorMessage);\n              onError?.(errorMessage);\n            }\n          } catch (validationError) {\n            const errorMessage = `Validation failed: ${validationError instanceof Error ? validationError.message : String(validationError)}`;\n            setLastError(errorMessage);\n            onError?.(errorMessage);\n          }\n        }\n        \n        setIsInitialized(true);\n        onInitialized?.();\n        \n      } catch (error) {\n        const errorMessage = `Settings provider initialization failed: ${error instanceof Error ? error.message : String(error)}`;\n        setLastError(errorMessage);\n        onError?.(errorMessage);\n      } finally {\n        setIsLoading(false);\n      }\n    };\n    \n    initialize();\n  }, [autoValidate, autoMigrate, targetVersion, onInitialized, onError]);\n  \n  // Validation functions\n  const validate = React.useCallback((): ValidationResult => {\n    const currentSettings = store.partialSettings || {};\n    return validateSettings(currentSettings);\n  }, [store.partialSettings]);\n  \n  const validatePath = React.useCallback((path: SettingsPath): ValidationResult => {\n    const currentSettings = store.partialSettings || {};\n    return validateSettings(currentSettings); // In practice, you'd validate just the path\n  }, [store.partialSettings]);\n  \n  // Migration function\n  const migrate = React.useCallback(async (targetVersion: string): Promise<MigrationResult> => {\n    const currentSettings = store.partialSettings || {};\n    const result = await migrateSettings(currentSettings, targetVersion);\n    \n    if (result.success && result.migratedSettings) {\n      // Apply migrated settings to store\n      // This would depend on your store implementation\n      // store.replaceAllSettings(result.migratedSettings);\n    }\n    \n    return result;\n  }, [store]);\n  \n  // Event listener functions\n  const onSettingChange = React.useCallback((\n    path: SettingsPath,\n    callback: (value: any) => void\n  ): (() => void) => {\n    // Get or create listener set for this path\n    if (!settingChangeListeners.current.has(path)) {\n      settingChangeListeners.current.set(path, new Set());\n    }\n    \n    const listeners = settingChangeListeners.current.get(path)!;\n    listeners.add(callback);\n    \n    // Return unsubscribe function\n    return () => {\n      listeners.delete(callback);\n      if (listeners.size === 0) {\n        settingChangeListeners.current.delete(path);\n      }\n    };\n  }, []);\n  \n  const onValidationError = React.useCallback((\n    callback: (errors: ValidationResult) => void\n  ): (() => void) => {\n    validationErrorListeners.current.add(callback);\n    \n    return () => {\n      validationErrorListeners.current.delete(callback);\n    };\n  }, []);\n  \n  const onMigrationComplete = React.useCallback((\n    callback: (result: MigrationResult) => void\n  ): (() => void) => {\n    migrationCompleteListeners.current.add(callback);\n    \n    return () => {\n      migrationCompleteListeners.current.delete(callback);\n    };\n  }, []);\n  \n  // Subscribe to store changes to notify listeners\n  useEffect(() => {\n    // This would depend on your store's subscription mechanism\n    // const unsubscribe = store.subscribe((path, value) => {\n    //   const listeners = settingChangeListeners.current.get(path);\n    //   if (listeners) {\n    //     listeners.forEach(listener => listener(value));\n    //   }\n    // });\n    \n    // return unsubscribe;\n  }, [store]);\n  \n  const contextValue: SettingsProviderContextValue = {\n    store,\n    validate,\n    validatePath,\n    migrate,\n    isLoading,\n    isInitialized,\n    lastError,\n    onSettingChange,\n    onValidationError,\n    onMigrationComplete\n  };\n  \n  return (\n    <SettingsProviderContext.Provider value={contextValue}>\n      {children}\n    </SettingsProviderContext.Provider>\n  );\n}\n\n/**\n * Hook to use Settings Provider Context\n */\nexport function useSettingsProvider(): SettingsProviderContextValue {\n  const context = useContext(SettingsProviderContext);\n  \n  if (!context) {\n    throw new Error('useSettingsProvider must be used within a SettingsProvider');\n  }\n  \n  return context;\n}\n\n/**\n * Hook for settings validation within provider context\n */\nexport function useSettingsValidation() {\n  const { validate, validatePath, onValidationError } = useSettingsProvider();\n  const [validationResult, setValidationResult] = React.useState<ValidationResult | null>(null);\n  const [isValidating, setIsValidating] = React.useState(false);\n  \n  const validateSettings = React.useCallback(async () => {\n    setIsValidating(true);\n    try {\n      const result = validate();\n      setValidationResult(result);\n      return result;\n    } finally {\n      setIsValidating(false);\n    }\n  }, [validate]);\n  \n  const validateSettingPath = React.useCallback(async (path: SettingsPath) => {\n    setIsValidating(true);\n    try {\n      const result = validatePath(path);\n      return result;\n    } finally {\n      setIsValidating(false);\n    }\n  }, [validatePath]);\n  \n  // Subscribe to validation errors\n  useEffect(() => {\n    const unsubscribe = onValidationError((errors) => {\n      setValidationResult(errors);\n    });\n    \n    return unsubscribe;\n  }, [onValidationError]);\n  \n  return {\n    validationResult,\n    isValidating,\n    validateSettings,\n    validateSettingPath\n  };\n}\n\n/**\n * Hook for settings migration within provider context\n */\nexport function useSettingsMigration() {\n  const { migrate, onMigrationComplete } = useSettingsProvider();\n  const [migrationResult, setMigrationResult] = React.useState<MigrationResult | null>(null);\n  const [isMigrating, setIsMigrating] = React.useState(false);\n  \n  const migrateSettings = React.useCallback(async (targetVersion: string) => {\n    setIsMigrating(true);\n    try {\n      const result = await migrate(targetVersion);\n      setMigrationResult(result);\n      return result;\n    } finally {\n      setIsMigrating(false);\n    }\n  }, [migrate]);\n  \n  // Subscribe to migration completion\n  useEffect(() => {\n    const unsubscribe = onMigrationComplete((result) => {\n      setMigrationResult(result);\n    });\n    \n    return unsubscribe;\n  }, [onMigrationComplete]);\n  \n  return {\n    migrationResult,\n    isMigrating,\n    migrateSettings\n  };\n}\n\n/**\n * Hook for subscribing to setting changes within provider context\n */\nexport function useSettingChangeListener<T>(\n  path: SettingsPath,\n  callback: (value: T) => void,\n  dependencies: React.DependencyList = []\n) {\n  const { onSettingChange } = useSettingsProvider();\n  \n  useEffect(() => {\n    const unsubscribe = onSettingChange(path, callback);\n    return unsubscribe;\n  }, [path, callback, onSettingChange, ...dependencies]);\n}