// Enhanced Control Center - Reorganized tabbed interface
import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Input } from '@/features/design-system/components/Input';
import { Button } from '@/features/design-system/components/Button';
import { 
  Monitor, 
  Palette, 
  Activity, 
  BarChart3, 
  Headphones,
  Gauge,
  Database,
  Code,
  Search,
  Settings
} from 'lucide-react';
import { useSettingsStore } from '@/store/settingsStore';

// Tab components
import {
  DashboardTab,
  VisualizationTab,
  PhysicsEngineTab,
  AnalyticsTab,
  XRTab,
  PerformanceTab,
  DataManagementTab,
  DeveloperTab
} from './tabs';

interface EnhancedControlCenterProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export const EnhancedControlCenter: React.FC<EnhancedControlCenterProps> = ({ 
  isOpen = true, 
  onClose 
}) => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [searchQuery, setSearchQuery] = useState('');
  const { flushPendingUpdates } = useSettingsStore();
  
  // Tab definitions following the reorganization plan
  const tabs = [
    { 
      id: 'dashboard', 
      label: 'Dashboard', 
      icon: Monitor, 
      description: 'System overview and quick actions',
      component: DashboardTab 
    },
    { 
      id: 'visualization', 
      label: 'Visualization', 
      icon: Palette, 
      description: 'Nodes, edges, effects & rendering',
      component: VisualizationTab 
    },
    { 
      id: 'physics', 
      label: 'Physics Engine', 
      icon: Activity, 
      description: 'GPU engine, forces & constraints',
      component: PhysicsEngineTab 
    },
    { 
      id: 'analytics', 
      label: 'Analytics', 
      icon: BarChart3, 
      description: 'Clustering, anomaly detection & ML',
      component: AnalyticsTab 
    },
    { 
      id: 'xr', 
      label: 'XR/AR', 
      icon: Headphones, 
      description: 'Quest 3, spatial computing',
      component: XRTab 
    },
    { 
      id: 'performance', 
      label: 'Performance', 
      icon: Gauge, 
      description: 'Monitoring, optimization & profiling',
      component: PerformanceTab 
    },
    { 
      id: 'data', 
      label: 'Data Management', 
      icon: Database, 
      description: 'Import/export, streaming & persistence',
      component: DataManagementTab 
    },
    { 
      id: 'developer', 
      label: 'Developer', 
      icon: Code, 
      description: 'Debug tools, API testing & features',
      component: DeveloperTab 
    },
  ];
  
  // Flush any pending updates when component unmounts or tab changes
  useEffect(() => {
    return () => {
      flushPendingUpdates();
    };
  }, [flushPendingUpdates]);
  
  useEffect(() => {
    // Flush pending updates when switching tabs
    flushPendingUpdates();
  }, [activeTab, flushPendingUpdates]);
  
  const currentTab = tabs.find(tab => tab.id === activeTab);
  const CurrentTabComponent = currentTab?.component;
  
  if (!isOpen) {
    return null;
  }
  
  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b bg-muted/20">
        <div className="flex items-center gap-3">
          <Settings className="w-6 h-6 text-primary" />
          <div>
            <h1 className="text-xl font-bold">Control Center</h1>
            <p className="text-sm text-muted-foreground">
              {currentTab?.description || 'Unified settings and configuration'}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Global search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search all settings..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 w-64"
            />
          </div>
          
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ×
            </Button>
          )}
        </div>
      </div>
      
      {/* Tab Navigation & Content */}
      <div className="flex-1 flex min-h-0">
        <Tabs 
          value={activeTab} 
          onValueChange={setActiveTab}
          className="flex-1 flex flex-col"
        >
          {/* Vertical Tab List */}
          <div className="flex border-b">
            <TabsList className="grid w-full grid-cols-4 lg:grid-cols-8 h-auto p-1">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <TabsTrigger
                    key={tab.id}
                    value={tab.id}
                    className="flex flex-col items-center gap-1 py-3 px-4 data-[state=active]:bg-primary/10"
                  >
                    <Icon className="w-4 h-4" />
                    <span className="text-xs font-medium">{tab.label}</span>
                  </TabsTrigger>
                );
              })}
            </TabsList>
          </div>
          
          {/* Tab Content */}
          <div className="flex-1 overflow-hidden">
            {tabs.map((tab) => {
              const Component = tab.component;
              return (
                <TabsContent 
                  key={tab.id}
                  value={tab.id}
                  className="h-full m-0 p-0 data-[state=active]:flex data-[state=active]:flex-col"
                >
                  <ScrollArea className="flex-1">
                    <div className="p-6">
                      <Component searchQuery={searchQuery} />
                    </div>
                  </ScrollArea>
                </TabsContent>
              );
            })}
          </div>
        </Tabs>
      </div>
    </div>
  );
};

export default EnhancedControlCenter;