import { SettingsPath } from '../types/generated/settings';
import { deepClone, getAllPaths, getValueByPath } from './settingsHelpers';
import { validateSettings, ValidationResult } from './settingsValidator';

/**
 * Settings export/import utilities for backup, sharing, and migration
 */

export interface ExportOptions {
  format: 'json' | 'yaml' | 'toml' | 'ini';
  includeMetadata?: boolean;
  includeSecrets?: boolean;
  compress?: boolean;
  prettify?: boolean;
  excludePaths?: SettingsPath[];
  includePaths?: SettingsPath[];
  addTimestamp?: boolean;
  addVersion?: boolean;
}

export interface ImportOptions {
  validateBeforeImport?: boolean;
  mergeMode?: 'replace' | 'merge' | 'selective';
  backupCurrent?: boolean;
  skipValidation?: boolean;
  allowPartialImport?: boolean;
}

export interface ExportResult {
  success: boolean;
  data?: string;
  blob?: Blob;
  metadata?: ExportMetadata;
  errors: string[];
  warnings: string[];
}

export interface ImportResult {
  success: boolean;
  importedSettings?: any;
  backup?: any;
  appliedPaths: SettingsPath[];
  skippedPaths: SettingsPath[];
  errors: string[];
  warnings: string[];
  validationResult?: ValidationResult;
}

export interface ExportMetadata {
  exportedAt: string;
  exportedBy?: string;
  version: string;
  format: string;
  pathCount: number;
  settingsVersion?: string;
  appVersion?: string;
}

/**
 * Default export options
 */
const DEFAULT_EXPORT_OPTIONS: ExportOptions = {
  format: 'json',
  includeMetadata: true,
  includeSecrets: false,
  compress: false,
  prettify: true,
  addTimestamp: true,
  addVersion: true
};

/**
 * Default import options
 */
const DEFAULT_IMPORT_OPTIONS: ImportOptions = {
  validateBeforeImport: true,
  mergeMode: 'replace',
  backupCurrent: true,
  allowPartialImport: false
};

/**
 * Sensitive paths that should not be exported by default
 */
const SENSITIVE_PATHS = [
  'system.apiKey',
  'system.secrets',
  'auth.tokens',
  'auth.credentials'
];

/**
 * Exports settings to specified format
 * @param settings - Settings object to export
 * @param options - Export options
 * @returns Export result
 */
export async function exportSettings(
  settings: any,
  options: Partial<ExportOptions> = {}
): Promise<ExportResult> {
  const opts = { ...DEFAULT_EXPORT_OPTIONS, ...options };
  const errors: string[] = [];
  const warnings: string[] = [];
  
  try {
    // Clone settings to avoid modifying original
    let exportData = deepClone(settings);
    
    // Filter out sensitive data if requested
    if (!opts.includeSecrets) {
      SENSITIVE_PATHS.forEach(path => {
        try {
          deletePath(exportData, path as SettingsPath);
        } catch {
          // Ignore if path doesn't exist
        }
      });
    }
    
    // Apply path filters
    if (opts.includePaths && opts.includePaths.length > 0) {
      exportData = filterIncludePaths(exportData, opts.includePaths);
    }
    
    if (opts.excludePaths && opts.excludePaths.length > 0) {
      exportData = filterExcludePaths(exportData, opts.excludePaths);
    }
    
    // Create metadata
    const metadata: ExportMetadata = {
      exportedAt: new Date().toISOString(),
      version: '1.0.0',
      format: opts.format,
      pathCount: getAllPaths(exportData).length,
      settingsVersion: getValueByPath(exportData, 'system.version' as SettingsPath),
      appVersion: getValueByPath(exportData, 'system.appVersion' as SettingsPath)
    };
    
    // Add metadata to export if requested
    if (opts.includeMetadata) {
      exportData._metadata = metadata;
    }
    
    // Convert to requested format
    let data: string;
    let blob: Blob;
    
    switch (opts.format) {
      case 'json':
        data = JSON.stringify(exportData, null, opts.prettify ? 2 : 0);
        blob = new Blob([data], { type: 'application/json' });
        break;
        
      case 'yaml':
        data = convertToYaml(exportData);
        blob = new Blob([data], { type: 'text/yaml' });
        break;
        
      case 'toml':
        data = convertToToml(exportData);
        blob = new Blob([data], { type: 'text/plain' });
        break;
        
      case 'ini':
        data = convertToIni(exportData);
        blob = new Blob([data], { type: 'text/plain' });
        break;
        
      default:
        throw new Error(`Unsupported format: ${opts.format}`);
    }
    
    // Compress if requested
    if (opts.compress) {
      // In a real implementation, you would use a compression library
      warnings.push('Compression not implemented, exporting uncompressed');
    }
    
    return {
      success: true,
      data,
      blob,
      metadata,
      errors,
      warnings
    };
    
  } catch (error) {
    return {
      success: false,
      errors: [`Export failed: ${error instanceof Error ? error.message : String(error)}`],
      warnings
    };
  }
}

/**
 * Imports settings from string data
 * @param data - Settings data to import
 * @param format - Data format
 * @param currentSettings - Current settings (for backup/merge)
 * @param options - Import options
 * @returns Import result
 */
export async function importSettings(
  data: string,
  format: 'json' | 'yaml' | 'toml' | 'ini',
  currentSettings: any,
  options: Partial<ImportOptions> = {}
): Promise<ImportResult> {
  const opts = { ...DEFAULT_IMPORT_OPTIONS, ...options };
  const errors: string[] = [];
  const warnings: string[] = [];
  const appliedPaths: SettingsPath[] = [];
  const skippedPaths: SettingsPath[] = [];
  
  try {
    // Backup current settings if requested
    const backup = opts.backupCurrent ? deepClone(currentSettings) : undefined;
    
    // Parse imported data
    let importedData: any;
    
    switch (format) {
      case 'json':
        importedData = JSON.parse(data);
        break;
        
      case 'yaml':
        importedData = parseYaml(data);
        break;
        
      case 'toml':
        importedData = parseToml(data);
        break;
        
      case 'ini':
        importedData = parseIni(data);
        break;
        
      default:
        throw new Error(`Unsupported format: ${format}`);
    }
    
    // Extract metadata if present
    const metadata = importedData._metadata;
    if (metadata) {
      warnings.push(`Importing settings from ${metadata.exportedAt} (${metadata.format})`);\n      delete importedData._metadata;\n    }\n    \n    // Validate imported settings if requested\n    let validationResult: ValidationResult | undefined;\n    if (opts.validateBeforeImport && !opts.skipValidation) {\n      validationResult = validateSettings(importedData);\n      \n      if (!validationResult.isValid && !opts.allowPartialImport) {\n        return {\n          success: false,\n          appliedPaths: [],\n          skippedPaths: [],\n          errors: [`Imported settings validation failed: ${validationResult.errors.length} errors`],\n          warnings,\n          validationResult\n        };\n      }\n    }\n    \n    // Apply settings based on merge mode\n    let finalSettings: any;\n    \n    switch (opts.mergeMode) {\n      case 'replace':\n        finalSettings = deepClone(importedData);\n        appliedPaths.push(...getAllPaths(importedData));\n        break;\n        \n      case 'merge':\n        finalSettings = deepMergeSettings(currentSettings, importedData);\n        appliedPaths.push(...getAllPaths(importedData));\n        break;\n        \n      case 'selective':\n        // In selective mode, only import valid settings\n        const { merged, applied, skipped } = selectiveImport(\n          currentSettings,\n          importedData,\n          validationResult\n        );\n        finalSettings = merged;\n        appliedPaths.push(...applied);\n        skippedPaths.push(...skipped);\n        break;\n        \n      default:\n        throw new Error(`Unknown merge mode: ${opts.mergeMode}`);\n    }\n    \n    return {\n      success: true,\n      importedSettings: finalSettings,\n      backup,\n      appliedPaths,\n      skippedPaths,\n      errors,\n      warnings,\n      validationResult\n    };\n    \n  } catch (error) {\n    return {\n      success: false,\n      appliedPaths: [],\n      skippedPaths: [],\n      errors: [`Import failed: ${error instanceof Error ? error.message : String(error)}`],\n      warnings\n    };\n  }\n}\n\n/**\n * Exports settings to a file\n * @param settings - Settings to export\n * @param filename - Export filename\n * @param options - Export options\n */\nexport async function exportSettingsToFile(\n  settings: any,\n  filename: string,\n  options: Partial<ExportOptions> = {}\n): Promise<void> {\n  const result = await exportSettings(settings, options);\n  \n  if (!result.success || !result.blob) {\n    throw new Error(`Export failed: ${result.errors.join(', ')}`);\n  }\n  \n  // Create download link and trigger download\n  const url = URL.createObjectURL(result.blob);\n  const link = document.createElement('a');\n  link.href = url;\n  link.download = filename;\n  \n  document.body.appendChild(link);\n  link.click();\n  document.body.removeChild(link);\n  \n  URL.revokeObjectURL(url);\n}\n\n/**\n * Imports settings from a file\n * @param file - File to import\n * @param currentSettings - Current settings\n * @param options - Import options\n * @returns Import result\n */\nexport async function importSettingsFromFile(\n  file: File,\n  currentSettings: any,\n  options: Partial<ImportOptions> = {}\n): Promise<ImportResult> {\n  return new Promise((resolve) => {\n    const reader = new FileReader();\n    \n    reader.onload = async (e) => {\n      if (!e.target?.result) {\n        resolve({\n          success: false,\n          appliedPaths: [],\n          skippedPaths: [],\n          errors: ['Failed to read file'],\n          warnings: []\n        });\n        return;\n      }\n      \n      const data = e.target.result as string;\n      const format = getFormatFromFilename(file.name) || 'json';\n      \n      const result = await importSettings(data, format, currentSettings, options);\n      resolve(result);\n    };\n    \n    reader.onerror = () => {\n      resolve({\n        success: false,\n        appliedPaths: [],\n        skippedPaths: [],\n        errors: ['Failed to read file'],\n        warnings: []\n      });\n    };\n    \n    reader.readAsText(file);\n  });\n}\n\n// Helper functions\n\nfunction deletePath(obj: any, path: SettingsPath): void {\n  const parts = path.split('.');\n  let current = obj;\n  \n  for (let i = 0; i < parts.length - 1; i++) {\n    if (!current[parts[i]]) return;\n    current = current[parts[i]];\n  }\n  \n  delete current[parts[parts.length - 1]];\n}\n\nfunction filterIncludePaths(settings: any, includePaths: SettingsPath[]): any {\n  const filtered: any = {};\n  \n  includePaths.forEach(path => {\n    const value = getValueByPath(settings, path);\n    if (value !== undefined) {\n      setNestedValue(filtered, path, value);\n    }\n  });\n  \n  return filtered;\n}\n\nfunction filterExcludePaths(settings: any, excludePaths: SettingsPath[]): any {\n  const filtered = deepClone(settings);\n  \n  excludePaths.forEach(path => {\n    deletePath(filtered, path);\n  });\n  \n  return filtered;\n}\n\nfunction setNestedValue(obj: any, path: SettingsPath, value: any): void {\n  const parts = path.split('.');\n  let current = obj;\n  \n  for (let i = 0; i < parts.length - 1; i++) {\n    if (!current[parts[i]]) {\n      current[parts[i]] = {};\n    }\n    current = current[parts[i]];\n  }\n  \n  current[parts[parts.length - 1]] = value;\n}\n\nfunction deepMergeSettings(target: any, source: any): any {\n  const result = deepClone(target);\n  \n  function merge(dest: any, src: any): void {\n    for (const [key, value] of Object.entries(src)) {\n      if (value && typeof value === 'object' && !Array.isArray(value)) {\n        if (!dest[key] || typeof dest[key] !== 'object') {\n          dest[key] = {};\n        }\n        merge(dest[key], value);\n      } else {\n        dest[key] = deepClone(value);\n      }\n    }\n  }\n  \n  merge(result, source);\n  return result;\n}\n\nfunction selectiveImport(\n  currentSettings: any,\n  importedData: any,\n  validationResult?: ValidationResult\n): { merged: any; applied: SettingsPath[]; skipped: SettingsPath[] } {\n  const merged = deepClone(currentSettings);\n  const applied: SettingsPath[] = [];\n  const skipped: SettingsPath[] = [];\n  \n  const allPaths = getAllPaths(importedData);\n  const invalidPaths = new Set(\n    validationResult?.errors.map(error => error.path) || []\n  );\n  \n  allPaths.forEach(path => {\n    if (invalidPaths.has(path)) {\n      skipped.push(path);\n    } else {\n      const value = getValueByPath(importedData, path);\n      setNestedValue(merged, path, value);\n      applied.push(path);\n    }\n  });\n  \n  return { merged, applied, skipped };\n}\n\nfunction getFormatFromFilename(filename: string): 'json' | 'yaml' | 'toml' | 'ini' | null {\n  const extension = filename.toLowerCase().split('.').pop();\n  \n  switch (extension) {\n    case 'json':\n      return 'json';\n    case 'yaml':\n    case 'yml':\n      return 'yaml';\n    case 'toml':\n      return 'toml';\n    case 'ini':\n    case 'cfg':\n      return 'ini';\n    default:\n      return null;\n  }\n}\n\n// Format conversion functions (simplified implementations)\nfunction convertToYaml(obj: any): string {\n  // Simplified YAML conversion - in production use a proper YAML library\n  return JSON.stringify(obj, null, 2)\n    .replace(/\"/g, '')\n    .replace(/: /g, ': ');\n}\n\nfunction convertToToml(obj: any): string {\n  // Simplified TOML conversion - in production use a proper TOML library\n  let toml = '';\n  \n  function convertValue(value: any, prefix = ''): string {\n    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {\n      let result = '';\n      for (const [key, val] of Object.entries(value)) {\n        const newPrefix = prefix ? `${prefix}.${key}` : key;\n        if (typeof val === 'object' && val !== null && !Array.isArray(val)) {\n          result += `\\n[${newPrefix}]\\n`;\n          result += convertValue(val, newPrefix);\n        } else {\n          result += `${key} = ${JSON.stringify(val)}\\n`;\n        }\n      }\n      return result;\n    }\n    return '';\n  }\n  \n  return convertValue(obj);\n}\n\nfunction convertToIni(obj: any): string {\n  // Simplified INI conversion\n  let ini = '';\n  \n  function convertSection(section: any, sectionName = ''): string {\n    let result = '';\n    \n    if (sectionName) {\n      result += `[${sectionName}]\\n`;\n    }\n    \n    for (const [key, value] of Object.entries(section)) {\n      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {\n        result += convertSection(value, sectionName ? `${sectionName}.${key}` : key);\n      } else {\n        result += `${key}=${JSON.stringify(value)}\\n`;\n      }\n    }\n    \n    result += '\\n';\n    return result;\n  }\n  \n  return convertSection(obj);\n}\n\nfunction parseYaml(data: string): any {\n  // Simplified YAML parsing - in production use a proper YAML library\n  try {\n    return JSON.parse(data.replace(/'/g, '\"'));\n  } catch {\n    throw new Error('Invalid YAML format');\n  }\n}\n\nfunction parseToml(data: string): any {\n  // Simplified TOML parsing - in production use a proper TOML library\n  throw new Error('TOML parsing not implemented in this demo');\n}\n\nfunction parseIni(data: string): any {\n  // Simplified INI parsing\n  const result: any = {};\n  const lines = data.split('\\n');\n  let currentSection: any = result;\n  \n  for (const line of lines) {\n    const trimmed = line.trim();\n    \n    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith(';')) {\n      continue;\n    }\n    \n    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {\n      const sectionName = trimmed.slice(1, -1);\n      const parts = sectionName.split('.');\n      \n      currentSection = result;\n      for (const part of parts) {\n        if (!currentSection[part]) {\n          currentSection[part] = {};\n        }\n        currentSection = currentSection[part];\n      }\n    } else if (trimmed.includes('=')) {\n      const [key, ...valueParts] = trimmed.split('=');\n      const value = valueParts.join('=');\n      \n      try {\n        currentSection[key.trim()] = JSON.parse(value.trim());\n      } catch {\n        currentSection[key.trim()] = value.trim();\n      }\n    }\n  }\n  \n  return result;\n}\n\n/**\n * Creates a settings template with default values\n * @param paths - Paths to include in template\n * @returns Template object\n */\nexport function createSettingsTemplate(paths?: SettingsPath[]): any {\n  const template: any = {};\n  \n  const defaultPaths = paths || [\n    'ui.theme',\n    'ui.fontSize',\n    'system.debug',\n    'system.performance.memoryOptimization',\n    'visualisation.quality',\n    'visualisation.camera.fov'\n  ];\n  \n  const defaults: Record<string, any> = {\n    'ui.theme': 'system',\n    'ui.fontSize': 'medium',\n    'system.debug': false,\n    'system.performance.memoryOptimization': true,\n    'visualisation.quality': 'medium',\n    'visualisation.camera.fov': 75\n  };\n  \n  defaultPaths.forEach(path => {\n    setNestedValue(template, path, defaults[path] || null);\n  });\n  \n  return template;\n}