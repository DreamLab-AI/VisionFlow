import React, { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { 
  Activity, 
  Eye, 
  Zap, 
  BarChart3, 
  Box, 
  Gauge, 
  Database,
  Code
} from 'lucide-react';

// Import existing components
import { DashboardPanel } from '../../dashboard/components/DashboardPanel';
import { PhysicsEngineControls } from '../../../physics/components/PhysicsEngineControls';
import { SettingsPanelProgrammatic } from './SettingsPanelProgrammatic';

export function SettingsPanelRedesign() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="h-full flex flex-col bg-background">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-col h-full">
        <TabsList className="grid grid-cols-4 lg:grid-cols-8 gap-1 p-1 h-auto">
          <TabsTrigger value="dashboard" className="flex flex-col gap-1 p-2">
            <Activity className="h-4 w-4" />
            <span className="text-xs">Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex flex-col gap-1 p-2">
            <Eye className="h-4 w-4" />
            <span className="text-xs">Visual</span>
          </TabsTrigger>
          <TabsTrigger value="physics" className="flex flex-col gap-1 p-2">
            <Zap className="h-4 w-4" />
            <span className="text-xs">Physics</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex flex-col gap-1 p-2">
            <BarChart3 className="h-4 w-4" />
            <span className="text-xs">Analytics</span>
          </TabsTrigger>
          <TabsTrigger value="xr" className="flex flex-col gap-1 p-2">
            <Box className="h-4 w-4" />
            <span className="text-xs">XR/AR</span>
          </TabsTrigger>
          <TabsTrigger value="performance" className="flex flex-col gap-1 p-2">
            <Gauge className="h-4 w-4" />
            <span className="text-xs">Perf</span>
          </TabsTrigger>
          <TabsTrigger value="data" className="flex flex-col gap-1 p-2">
            <Database className="h-4 w-4" />
            <span className="text-xs">Data</span>
          </TabsTrigger>
          <TabsTrigger value="developer" className="flex flex-col gap-1 p-2">
            <Code className="h-4 w-4" />
            <span className="text-xs">Dev</span>
          </TabsTrigger>
        </TabsList>

        <ScrollArea className="flex-1">
          <TabsContent value="dashboard" className="mt-0">
            <DashboardPanel />
          </TabsContent>

          <TabsContent value="visualization" className="mt-0">
            {/* Use existing visualization settings */}
            <SettingsPanelProgrammatic />
          </TabsContent>

          <TabsContent value="physics" className="mt-0">
            <PhysicsEngineControls />
          </TabsContent>

          <TabsContent value="analytics" className="mt-0">
            <div className="p-4">
              <h3 className="text-lg font-semibold mb-4">Analytics</h3>
              <p className="text-muted-foreground">Clustering, anomaly detection, and ML features coming soon...</p>
            </div>
          </TabsContent>

          <TabsContent value="xr" className="mt-0">
            <div className="p-4">
              <h3 className="text-lg font-semibold mb-4">XR/AR Settings</h3>
              <p className="text-muted-foreground">Quest 3 and immersive settings coming soon...</p>
            </div>
          </TabsContent>

          <TabsContent value="performance" className="mt-0">
            <div className="p-4">
              <h3 className="text-lg font-semibold mb-4">Performance</h3>
              <p className="text-muted-foreground">System monitoring and optimization coming soon...</p>
            </div>
          </TabsContent>

          <TabsContent value="data" className="mt-0">
            <div className="p-4">
              <h3 className="text-lg font-semibold mb-4">Data Management</h3>
              <p className="text-muted-foreground">Import/export and persistence coming soon...</p>
            </div>
          </TabsContent>

          <TabsContent value="developer" className="mt-0">
            <div className="p-4">
              <h3 className="text-lg font-semibold mb-4">Developer Tools</h3>
              <p className="text-muted-foreground">Debug tools and API testing coming soon...</p>
            </div>
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}