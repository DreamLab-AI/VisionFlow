import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../design-system/components/Card';
import { Label } from '../../design-system/components/Label';
import { Switch } from '../../design-system/components/Switch';
import { Button } from '../../design-system/components/Button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../design-system/components/Select';
import { Textarea } from '../../design-system/components/Textarea';
import { Badge } from '../../design-system/components/Badge';
import { Download, Upload, FileJson, Copy, Check, AlertTriangle, RotateCcw, FileText } from 'lucide-react';
import { useSelectiveSetting, useSettingSetter } from '../../../hooks/useSelectiveSettingsStore';
import { settingsApi } from '../../../api/settingsApi';
import { toast } from '../../../utils/toast';
import { useSettingsStore } from '../../../store/settingsStore';

/**
 * ExportImportSettings Settings Panel
 * Provides settings backup, restore, and sharing functionality with selective access patterns
 */
export function ExportImportSettings() {
  const { set } = useSettingSetter();
  const [exportData, setExportData] = useState<string>('');
  const [importData, setImportData] = useState<string>('');
  const [copied, setCopied] = useState(false);
  const [lastExportTime, setLastExportTime] = useState<Date | null>(null);
  const [lastImportTime, setLastImportTime] = useState<Date | null>(null);
  
  const { partialSettings } = useSettingsStore();
  const { batchSet } = useSettingSetter();
  
  // Use selective settings access for export/import preferences
  const autoBackup = useSelectiveSetting<boolean>('system.backup.autoEnabled') ?? false;
  const backupInterval = useSelectiveSetting<number>('system.backup.interval') ?? 24; // hours
  const includePersonalData = useSelectiveSetting<boolean>('system.backup.includePersonalData') ?? false;
  const compressBackups = useSelectiveSetting<boolean>('system.backup.compress') ?? true;
  const maxBackups = useSelectiveSetting<number>('system.backup.maxBackups') ?? 10;
  const cloudBackup = useSelectiveSetting<boolean>('system.backup.cloudEnabled') ?? false;

  const handleSettingChange = async (path: string, value: any) => {
    await set(path, value);
  };

  const exportSettings = async (format: 'json' | 'yaml' | 'minimal' = 'json') => {
    try {
      let dataToExport: any;
      let filename: string;
      let mimeType: string;

      switch (format) {
        case 'minimal':
          // Export only essential settings
          dataToExport = {
            version: '1.0',
            timestamp: new Date().toISOString(),
            type: 'minimal',
            settings: {
              theme: partialSettings.ui?.theme,
              performance: partialSettings.system?.performance,
              visualization: {
                enabled: partialSettings.visualisation?.enabled,
                quality: partialSettings.visualisation?.quality,
                theme: partialSettings.visualisation?.theme
              }
            }
          };
          filename = `visionflow-settings-minimal-${new Date().toISOString().slice(0, 10)}.json`;
          mimeType = 'application/json';
          break;
        
        case 'yaml':
          // Convert to YAML format (simplified)
          const yamlData = convertToYaml(partialSettings);
          dataToExport = yamlData;
          filename = `visionflow-settings-${new Date().toISOString().slice(0, 10)}.yaml`;
          mimeType = 'text/yaml';
          break;
        
        case 'json':
        default:
          dataToExport = {
            version: '1.0',
            timestamp: new Date().toISOString(),
            type: 'full',
            settings: partialSettings,
            metadata: {
              userAgent: navigator.userAgent,
              platform: navigator.platform,
              includePersonalData
            }
          };
          filename = `visionflow-settings-${new Date().toISOString().slice(0, 10)}.json`;
          mimeType = 'application/json';
          break;
      }

      const content = typeof dataToExport === 'string' ? dataToExport : JSON.stringify(dataToExport, null, 2);
      setExportData(content);
      setLastExportTime(new Date());
      
      // Download file
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      toast.success(`Settings exported as ${format.toUpperCase()}`);
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export settings');
    }
  };

  const importSettings = async () => {
    if (!importData.trim()) {
      toast.error('Please paste settings data to import');
      return;
    }

    try {
      const parsed = JSON.parse(importData);
      
      // Validate the import data
      if (!parsed.settings) {
        throw new Error('Invalid settings file format');
      }
      
      // Check version compatibility
      if (parsed.version && parsed.version !== '1.0') {
        toast.warning(`Settings from version ${parsed.version} may not be fully compatible`);
      }
      
      // Import settings using the store's update method
      await useSettingsStore.getState().updateSettings((draft) => {
        Object.assign(draft, parsed.settings);
      });
      
      setLastImportTime(new Date());
      toast.success('Settings imported successfully');
      
      // Clear import data
      setImportData('');
    } catch (error) {
      console.error('Import failed:', error);
      toast.error('Failed to import settings. Please check the format.');
    }
  };

  const importFromFile = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,.yaml,.yml';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          setImportData(content);
          toast.info('File content loaded. Click Import to apply settings.');
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  const copyToClipboard = async () => {
    if (!exportData) {
      toast.error('No export data to copy');
      return;
    }
    
    try {
      await navigator.clipboard.writeText(exportData);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success('Settings copied to clipboard');
    } catch (error) {
      toast.error('Failed to copy to clipboard');
    }
  };

  const resetBackupSettings = async () => {
    await Promise.all([
      set('system.backup.autoEnabled', false),
      set('system.backup.interval', 24),
      set('system.backup.includePersonalData', false),
      set('system.backup.compress', true),
      set('system.backup.maxBackups', 10),
      set('system.backup.cloudEnabled', false)
    ]);
  };

  const convertToYaml = (obj: any, indent = 0): string => {
    const spaces = '  '.repeat(indent);
    let yaml = '';
    
    for (const key in obj) {
      const value = obj[key];
      yaml += `${spaces}${key}:`;
      
      if (value === null) {
        yaml += ' null\n';
      } else if (typeof value === 'boolean') {
        yaml += ` ${value}\n`;
      } else if (typeof value === 'number') {
        yaml += ` ${value}\n`;
      } else if (typeof value === 'string') {
        yaml += ` "${value}"\n`;
      } else if (Array.isArray(value)) {
        yaml += '\n';
        value.forEach(item => {
          yaml += `${spaces}  - ${typeof item === 'string' ? `"${item}"` : item}\n`;
        });
      } else if (typeof value === 'object') {
        yaml += '\n';
        yaml += convertToYaml(value, indent + 1);
      }
    }
    
    return yaml;
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <FileJson className="w-5 h-5" />
            <CardTitle>Export & Import Settings</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Export Section */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              <h3 className="text-sm font-medium">Export Settings</h3>
            </div>
            
            <div className="grid grid-cols-3 gap-2">
              <Button
                variant="outline"
                onClick={() => exportSettings('json')}
                className="flex items-center gap-2"
              >
                <FileJson className="w-4 h-4" />
                Full JSON
              </Button>
              <Button
                variant="outline"
                onClick={() => exportSettings('minimal')}
                className="flex items-center gap-2"
              >
                <FileText className="w-4 h-4" />
                Minimal
              </Button>
              <Button
                variant="outline"
                onClick={() => exportSettings('yaml')}
                className="flex items-center gap-2"
              >
                <FileText className="w-4 h-4" />
                YAML
              </Button>
            </div>
            
            {lastExportTime && (
              <div className="text-xs text-muted-foreground">
                Last exported: {lastExportTime.toLocaleString()}
              </div>
            )}
            
            {exportData && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Exported Data</Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyToClipboard}
                    className="h-6 px-2"
                  >
                    {copied ? (
                      <Check className="w-3 h-3 text-green-500" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </Button>
                </div>
                <Textarea
                  value={exportData}
                  readOnly
                  className="h-24 text-xs font-mono"
                  placeholder="Exported settings will appear here"
                />
              </div>
            )}
          </div>

          {/* Import Section */}
          <div className="space-y-4">
            <div className="border-t pt-4">
              <div className="flex items-center gap-2 mb-3">
                <Upload className="w-4 h-4" />
                <h3 className="text-sm font-medium">Import Settings</h3>
              </div>
              
              <div className="space-y-4">
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={importFromFile}
                    className="flex items-center gap-2"
                  >
                    <Upload className="w-4 h-4" />
                    Import from File
                  </Button>
                  <Button
                    variant="default"
                    onClick={importSettings}
                    disabled={!importData.trim()}
                  >
                    Apply Settings
                  </Button>
                </div>
                
                <div className="space-y-2">
                  <Label className="text-sm">Or paste settings data:</Label>
                  <Textarea
                    value={importData}
                    onChange={(e) => setImportData(e.target.value)}
                    className="h-24 text-xs font-mono"
                    placeholder="Paste exported settings JSON or YAML here..."
                  />
                </div>
                
                {lastImportTime && (
                  <div className="text-xs text-muted-foreground">
                    Last imported: {lastImportTime.toLocaleString()}
                  </div>
                )}
                
                {importData && (
                  <div className="flex items-center gap-2 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                    <AlertTriangle className="w-4 h-4 text-blue-600" />
                    <span className="text-sm text-blue-700 dark:text-blue-300">
                      Importing will overwrite current settings. Consider exporting current settings as backup.
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Auto Backup Settings */}
          <div className="space-y-4">
            <div className="border-t pt-4">
              <div className="flex items-center gap-2 mb-3">
                <RotateCcw className="w-4 h-4" />
                <h3 className="text-sm font-medium">Automatic Backup</h3>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-sm">Enable Auto Backup</Label>
                    <p className="text-xs text-muted-foreground">
                      Automatically backup settings periodically
                    </p>
                  </div>
                  <Switch
                    checked={autoBackup}
                    onCheckedChange={(checked) => handleSettingChange('system.backup.autoEnabled', checked)}
                  />
                </div>

                {autoBackup && (
                  <div className="space-y-4 pl-4 border-l-2 border-muted">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-sm">Backup Interval</Label>
                        <Select
                          value={backupInterval.toString()}
                          onValueChange={(value) => handleSettingChange('system.backup.interval', parseInt(value))}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1">Every Hour</SelectItem>
                            <SelectItem value="6">Every 6 Hours</SelectItem>
                            <SelectItem value="24">Daily</SelectItem>
                            <SelectItem value="168">Weekly</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-sm">Max Backups</Label>
                        <Select
                          value={maxBackups.toString()}
                          onValueChange={(value) => handleSettingChange('system.backup.maxBackups', parseInt(value))}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="5">5 backups</SelectItem>
                            <SelectItem value="10">10 backups</SelectItem>
                            <SelectItem value="25">25 backups</SelectItem>
                            <SelectItem value="50">50 backups</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Include Personal Data</Label>
                          <p className="text-xs text-muted-foreground">Include user-specific settings</p>
                        </div>
                        <Switch
                          checked={includePersonalData}
                          onCheckedChange={(checked) => handleSettingChange('system.backup.includePersonalData', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Compress Backups</Label>
                          <p className="text-xs text-muted-foreground">Reduce backup file size</p>
                        </div>
                        <Switch
                          checked={compressBackups}
                          onCheckedChange={(checked) => handleSettingChange('system.backup.compress', checked)}
                        />
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Cloud Backup</Label>
                        <p className="text-xs text-muted-foreground">
                          Store backups in cloud storage (requires authentication)
                        </p>
                      </div>
                      <Badge variant={cloudBackup ? 'default' : 'secondary'}>
                        {cloudBackup ? 'Enabled' : 'Local Only'}
                      </Badge>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="border-t pt-4">
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={async () => {
                  // Reset to defaults using the new resetSettingsSections method
                  try {
                    const defaultSettings = await settingsApi.resetSettingsSections(['essential']);
                    // Convert to batch updates
                    const updates = Object.entries(defaultSettings).map(([path, value]) => ({ path, value }));
                    await batchSet(updates);
                    toast.success('Settings reset to defaults');
                  } catch (error) {
                    toast.error('Failed to reset settings');
                  }
                }}
                className="flex-1"
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                Reset All Settings
              </Button>
              <Button
                variant="outline"
                onClick={resetBackupSettings}
              >
                Reset Backup Settings
              </Button>
            </div>
          </div>

          {/* Information */}
          <div className="bg-muted/20 p-4 rounded-lg space-y-2">
            <h4 className="text-sm font-medium">Export/Import Tips:</h4>
            <ul className="text-xs text-muted-foreground space-y-1">
              <li>• <strong>Full JSON:</strong> Complete settings backup with metadata</li>
              <li>• <strong>Minimal:</strong> Only essential settings for quick setup</li>
              <li>• <strong>YAML:</strong> Human-readable format for editing</li>
              <li>• Settings are validated before import to prevent corruption</li>
              <li>• Auto-backup creates local backups to prevent data loss</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ExportImportSettings;