import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Input } from '@/features/design-system/components/Input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Filter, Plus, X, Save, RotateCcw } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('FilterPanel');

interface FilterPanelProps {
  className?: string;
}

interface FilterRule {
  id: string;
  field: string;
  operator: 'equals' | 'contains' | 'startsWith' | 'endsWith' | 'greaterThan' | 'lessThan' | 'between';
  value: string | number | [number, number];
  enabled: boolean;
}

interface FilterGroup {
  id: string;
  name: string;
  rules: FilterRule[];
  logic: 'AND' | 'OR';
  enabled: boolean;
}

export const FilterPanel: React.FC<FilterPanelProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [activeTab, setActiveTab] = useState('rules');
  const [newRuleField, setNewRuleField] = useState('');
  const [newRuleOperator, setNewRuleOperator] = useState<FilterRule['operator']>('equals');
  const [newRuleValue, setNewRuleValue] = useState('');
  
  // Subscribe only to filter-related settings
  const filterSettings = useSelectiveSettings({
    enabled: 'filters.enabled',
    caseSensitive: 'filters.caseSensitive',
    autoApply: 'filters.autoApply',
    maxRules: 'filters.limits.maxRules',
    savePresets: 'filters.savePresets',
    showMatchCount: 'filters.showMatchCount',
    highlightMatches: 'filters.highlightMatches'
  });
  
  // Mock filter data - in real app this would come from store/API
  const [filterGroups, setFilterGroups] = useState<FilterGroup[]>([
    {
      id: '1',
      name: 'Active Users',
      rules: [
        {
          id: 'r1',
          field: 'status',
          operator: 'equals',
          value: 'active',
          enabled: true
        },
        {
          id: 'r2',
          field: 'lastLogin',
          operator: 'greaterThan',
          value: '2024-01-01',
          enabled: true
        }
      ],
      logic: 'AND',
      enabled: true
    },
    {
      id: '2',
      name: 'High Value Data',
      rules: [
        {
          id: 'r3',
          field: 'value',
          operator: 'greaterThan',
          value: 1000,
          enabled: true
        },
        {
          id: 'r4',
          field: 'category',
          operator: 'contains',
          value: 'premium',
          enabled: false
        }
      ],
      logic: 'OR',
      enabled: false
    }
  ]);
  
  const availableFields = useMemo(() => [
    'name', 'status', 'type', 'category', 'value', 'date', 'lastLogin', 'email', 'tags'
  ], []);
  
  const operators: Array<{ value: FilterRule['operator']; label: string }> = [
    { value: 'equals', label: 'Equals' },
    { value: 'contains', label: 'Contains' },
    { value: 'startsWith', label: 'Starts with' },
    { value: 'endsWith', label: 'Ends with' },
    { value: 'greaterThan', label: 'Greater than' },
    { value: 'lessThan', label: 'Less than' },
    { value: 'between', label: 'Between' }
  ];
  
  const addRule = (groupId: string) => {
    if (!newRuleField || !newRuleValue) return;
    
    const newRule: FilterRule = {
      id: `r${Date.now()}`,
      field: newRuleField,
      operator: newRuleOperator,
      value: newRuleValue,
      enabled: true
    };
    
    setFilterGroups(prev => prev.map(group => 
      group.id === groupId 
        ? { ...group, rules: [...group.rules, newRule] }
        : group
    ));
    
    setNewRuleField('');
    setNewRuleValue('');
    logger.info('Added filter rule', { groupId, rule: newRule });
  };
  
  const removeRule = (groupId: string, ruleId: string) => {
    setFilterGroups(prev => prev.map(group => 
      group.id === groupId 
        ? { ...group, rules: group.rules.filter(rule => rule.id !== ruleId) }
        : group
    ));
    logger.info('Removed filter rule', { groupId, ruleId });
  };
  
  const toggleRule = (groupId: string, ruleId: string) => {
    setFilterGroups(prev => prev.map(group => 
      group.id === groupId 
        ? {
            ...group,
            rules: group.rules.map(rule => 
              rule.id === ruleId ? { ...rule, enabled: !rule.enabled } : rule
            )
          }
        : group
    ));
  };
  
  const toggleGroup = (groupId: string) => {
    setFilterGroups(prev => prev.map(group => 
      group.id === groupId ? { ...group, enabled: !group.enabled } : group
    ));
  };
  
  const clearAllFilters = () => {
    setFilterGroups(prev => prev.map(group => ({ ...group, enabled: false })));
    logger.info('Cleared all filters');
  };
  
  const formatRuleValue = (rule: FilterRule): string => {
    if (Array.isArray(rule.value)) {
      return `${rule.value[0]} - ${rule.value[1]}`;
    }
    return String(rule.value);
  };
  
  const activeRulesCount = filterGroups
    .filter(group => group.enabled)
    .reduce((count, group) => count + group.rules.filter(rule => rule.enabled).length, 0);
  
  if (!filterSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter size={20} />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Filter size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">Filtering is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable filtering in settings to create and apply data filters
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Filter size={20} />
            Filter Panel
            {activeRulesCount > 0 && (
              <Badge className="bg-blue-100 text-blue-800">
                {activeRulesCount} active
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={clearAllFilters}>
              <RotateCcw size={16} className="mr-1" />
              Clear All
            </Button>
            {filterSettings.savePresets && (
              <Button size="sm">
                <Save size={16} className="mr-1" />
                Save Preset
              </Button>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="rules">Filter Rules</TabsTrigger>
            <TabsTrigger value="presets">Presets</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>
          
          <TabsContent value="rules" className="space-y-4">
            {/* Add New Rule */}
            <div className="border rounded-lg p-4">
              <h3 className="font-medium mb-3">Add New Rule</h3>
              <div className="grid grid-cols-4 gap-2">
                <Select value={newRuleField} onValueChange={setNewRuleField}>
                  <SelectTrigger>
                    <SelectValue placeholder="Field" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableFields.map(field => (
                      <SelectItem key={field} value={field}>
                        {field.charAt(0).toUpperCase() + field.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={newRuleOperator} onValueChange={setNewRuleOperator}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {operators.map(op => (
                      <SelectItem key={op.value} value={op.value}>
                        {op.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Input
                  value={newRuleValue}
                  onChange={(e) => setNewRuleValue(e.target.value)}
                  placeholder="Value"
                />
                <Button 
                  onClick={() => addRule(filterGroups[0]?.id)}
                  disabled={!newRuleField || !newRuleValue}
                >
                  <Plus size={16} />
                </Button>
              </div>
            </div>
            
            {/* Filter Groups */}
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {filterGroups.map((group) => (
                  <div key={group.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant={group.enabled ? 'default' : 'outline'}
                          onClick={() => toggleGroup(group.id)}
                        >
                          {group.enabled ? 'Enabled' : 'Disabled'}
                        </Button>
                        <span className="font-medium">{group.name}</span>
                        <Badge variant="outline">{group.logic}</Badge>
                        <Badge variant="secondary">
                          {group.rules.filter(rule => rule.enabled).length} rules
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      {group.rules.map((rule) => (
                        <div key={rule.id} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                          <Button
                            size="sm"
                            variant={rule.enabled ? 'default' : 'outline'}
                            onClick={() => toggleRule(group.id, rule.id)}
                            className="h-6 w-16 text-xs"
                          >
                            {rule.enabled ? 'ON' : 'OFF'}
                          </Button>
                          <span className="text-sm font-medium">{rule.field}</span>
                          <span className="text-sm text-muted-foreground">
                            {operators.find(op => op.value === rule.operator)?.label}
                          </span>
                          <span className="text-sm">{formatRuleValue(rule)}</span>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => removeRule(group.id, rule.id)}
                            className="h-6 w-6 p-0"
                          >
                            <X size={12} />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="presets" className="space-y-4">
            <div className="text-center py-8">
              <Save size={48} className="mx-auto mb-4 text-gray-400" />
              <p className="text-muted-foreground">No saved presets</p>
              <p className="text-sm text-muted-foreground mt-2">
                Save your current filter configuration as a preset
              </p>
              <Button className="mt-4">
                <Plus size={16} className="mr-1" />
                Create Preset
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Case Sensitive</span>
                <Button
                  variant={filterSettings.caseSensitive ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('filters.caseSensitive', !filterSettings.caseSensitive)}
                >
                  {filterSettings.caseSensitive ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Auto Apply</span>
                <Button
                  variant={filterSettings.autoApply ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('filters.autoApply', !filterSettings.autoApply)}
                >
                  {filterSettings.autoApply ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Show Match Count</span>
                <Button
                  variant={filterSettings.showMatchCount ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('filters.showMatchCount', !filterSettings.showMatchCount)}
                >
                  {filterSettings.showMatchCount ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Highlight Matches</span>
                <Button
                  variant={filterSettings.highlightMatches ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('filters.highlightMatches', !filterSettings.highlightMatches)}
                >
                  {filterSettings.highlightMatches ? 'Enabled' : 'Disabled'}
                </Button>
              </div>
            </div>
            
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">Limits</h3>
              <div className="text-sm text-muted-foreground">
                <p>Maximum rules per filter: {filterSettings.maxRules}</p>
                <p>Current active rules: {activeRulesCount}</p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default FilterPanel;