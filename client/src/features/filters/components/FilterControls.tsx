import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Input } from '@/features/design-system/components/Input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Popover, PopoverContent, PopoverTrigger } from '@/features/design-system/components/Dialog'; // Using Dialog as base
import { Filter, Search, X, Plus, RotateCcw } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('FilterControls');

interface FilterControlsProps {
  className?: string;
  onFiltersChange?: (filters: any[]) => void;
  compact?: boolean;
}

interface QuickFilter {
  id: string;
  label: string;
  field: string;
  value: string;
  active: boolean;
}

export const FilterControls: React.FC<FilterControlsProps> = ({ 
  className, 
  onFiltersChange,
  compact = false 
}) => {
  const { set } = useSettingSetter();
  const [searchTerm, setSearchTerm] = useState('');
  const [quickFilters, setQuickFilters] = useState<QuickFilter[]>([
    { id: '1', label: 'Active', field: 'status', value: 'active', active: false },
    { id: '2', label: 'Recent', field: 'date', value: 'last-7-days', active: false },
    { id: '3', label: 'High Priority', field: 'priority', value: 'high', active: false },
    { id: '4', label: 'Completed', field: 'status', value: 'completed', active: false }
  ]);
  
  // Memoize settings paths to prevent infinite loops
  const settingsPaths = useMemo(() => ({
    enabled: 'filters.enabled',
    showQuickFilters: 'filters.quickFilters.enabled',
    showSearch: 'filters.search.enabled',
    searchPlaceholder: 'filters.search.placeholder',
    caseSensitive: 'filters.caseSensitive',
    autoApply: 'filters.autoApply'
  }), []);
  
  // Subscribe only to filter control settings
  const filterSettings = useSelectiveSettings(settingsPaths);
  
  const activeFilters = quickFilters.filter(filter => filter.active);
  
  const toggleQuickFilter = (filterId: string) => {
    setQuickFilters(prev => 
      prev.map(filter => 
        filter.id === filterId 
          ? { ...filter, active: !filter.active }
          : filter
      )
    );
    
    const updatedFilters = quickFilters.map(filter => 
      filter.id === filterId 
        ? { ...filter, active: !filter.active }
        : filter
    );
    
    onFiltersChange?.(updatedFilters.filter(f => f.active));
    logger.info('Quick filter toggled', { filterId });
  };
  
  const clearAllFilters = () => {
    setQuickFilters(prev => prev.map(filter => ({ ...filter, active: false })));
    setSearchTerm('');
    onFiltersChange?.([]);
    logger.info('All filters cleared');
  };
  
  const handleSearchChange = (value: string) => {
    setSearchTerm(value);
    if (filterSettings.autoApply) {
      // In real app, trigger search/filter update
      logger.info('Search term changed', { searchTerm: value });
    }
  };
  
  if (!filterSettings.enabled) {
    return null;
  }
  
  if (compact) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        {filterSettings.showSearch && (
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
            <Input
              value={searchTerm}
              onChange={(e) => handleSearchChange(e.target.value)}
              placeholder={filterSettings.searchPlaceholder || "Search..."}
              className="pl-10 pr-8 h-8"
            />
            {searchTerm && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handleSearchChange('')}
                className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0"
              >
                <X size={12} />
              </Button>
            )}
          </div>
        )}
        
        {activeFilters.length > 0 && (
          <div className="flex items-center gap-1">
            <Badge className="bg-blue-100 text-blue-800">
              {activeFilters.length}
            </Badge>
            <Button size="sm" variant="ghost" onClick={clearAllFilters} className="h-6 w-6 p-0">
              <X size={12} />
            </Button>
          </div>
        )}
        
        <Popover>
          <PopoverTrigger asChild>
            <Button size="sm" variant="outline">
              <Filter size={14} />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-64">
            <div className="space-y-3">
              <h4 className="font-medium text-sm">Quick Filters</h4>
              <div className="space-y-2">
                {quickFilters.map((filter) => (
                  <div key={filter.id} className="flex items-center justify-between">
                    <span className="text-sm">{filter.label}</span>
                    <Button
                      size="sm"
                      variant={filter.active ? 'default' : 'outline'}
                      onClick={() => toggleQuickFilter(filter.id)}
                      className="h-6 text-xs"
                    >
                      {filter.active ? 'ON' : 'OFF'}
                    </Button>
                  </div>
                ))}
              </div>
              {activeFilters.length > 0 && (
                <Button size="sm" variant="outline" onClick={clearAllFilters} className="w-full">
                  Clear All
                </Button>
              )}
            </div>
          </PopoverContent>
        </Popover>
      </div>
    );
  }
  
  return (
    <div className={`space-y-4 ${className}`}>
      {/* Search Bar */}
      {filterSettings.showSearch && (
        <div className="relative">
          <Search size={20} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
          <Input
            value={searchTerm}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder={filterSettings.searchPlaceholder || "Search items..."}
            className="pl-10 pr-10"
          />
          {searchTerm && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => handleSearchChange('')}
              className="absolute right-1 top-1/2 transform -translate-y-1/2 h-8 w-8 p-0"
            >
              <X size={16} />
            </Button>
          )}
        </div>
      )}
      
      {/* Quick Filters */}
      {filterSettings.showQuickFilters && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Quick Filters</h3>
            {activeFilters.length > 0 && (
              <Button size="sm" variant="ghost" onClick={clearAllFilters}>
                <RotateCcw size={14} className="mr-1" />
                Clear ({activeFilters.length})
              </Button>
            )}
          </div>
          
          <div className="flex flex-wrap gap-2">
            {quickFilters.map((filter) => (
              <Button
                key={filter.id}
                size="sm"
                variant={filter.active ? 'default' : 'outline'}
                onClick={() => toggleQuickFilter(filter.id)}
                className="h-8"
              >
                {filter.label}
                {filter.active && <X size={12} className="ml-1" />}
              </Button>
            ))}
          </div>
        </div>
      )}
      
      {/* Active Filters Summary */}
      {(activeFilters.length > 0 || searchTerm) && (
        <div className="flex items-center gap-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <Filter size={16} className="text-blue-600" />
          <span className="text-sm text-blue-800 font-medium">
            {activeFilters.length + (searchTerm ? 1 : 0)} filter(s) active
          </span>
          <div className="flex items-center gap-1 ml-auto">
            {searchTerm && (
              <Badge variant="secondary" className="text-xs">
                Search: "{searchTerm}"
              </Badge>
            )}
            {activeFilters.map((filter) => (
              <Badge key={filter.id} variant="secondary" className="text-xs">
                {filter.label}
                <X 
                  size={10} 
                  className="ml-1 cursor-pointer" 
                  onClick={() => toggleQuickFilter(filter.id)}
                />
              </Badge>
            ))}
            <Button size="sm" variant="ghost" onClick={clearAllFilters} className="h-6 w-6 p-0">
              <X size={12} />
            </Button>
          </div>
        </div>
      )}
      
      {/* Filter Settings */}
      <div className="flex items-center gap-4 pt-2 border-t">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Case sensitive:</span>
          <Button
            size="sm"
            variant={filterSettings.caseSensitive ? 'default' : 'outline'}
            onClick={() => set('filters.caseSensitive', !filterSettings.caseSensitive)}
            className="h-6 text-xs"
          >
            {filterSettings.caseSensitive ? 'ON' : 'OFF'}
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Auto apply:</span>
          <Button
            size="sm"
            variant={filterSettings.autoApply ? 'default' : 'outline'}
            onClick={() => set('filters.autoApply', !filterSettings.autoApply)}
            className="h-6 text-xs"
          >
            {filterSettings.autoApply ? 'ON' : 'OFF'}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default FilterControls;