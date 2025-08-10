import React, { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { 
  Home, 
  Eye, 
  Zap, 
  BarChart3, 
  Headphones, 
  Activity, 
  Database,
  Code,
  Shield,
} from 'lucide-react';

// Import all panel components
import { PhysicsEngineControls } from '@/features/physics/components/PhysicsEngineControls';
import { VisualizationSettings } from '../VisualizationSettings';
import { AuthPanel } from '@/features/auth/components/AuthPanel';

// Dashboard component  
const Dashboard = () => (
  <div className="p-6 space-y-6">
    <div className="grid grid-cols-2 gap-4">
      <div className="p-4 border rounded-lg">
        <h3 className="font-semibold mb-2">Graph Status</h3>
        <p className="text-sm text-muted-foreground">Active nodes: 0</p>
        <p className="text-sm text-muted-foreground">Active edges: 0</p>
      </div>
      <div className="p-4 border rounded-lg">
        <h3 className="font-semibold mb-2">Performance</h3>
        <p className="text-sm text-muted-foreground">FPS: 60</p>
        <p className="text-sm text-muted-foreground">GPU Usage: 0%</p>
      </div>
    </div>
    
    <div className="p-4 border rounded-lg">
      <h3 className="font-semibold mb-2">Quick Actions</h3>
      <div className="flex gap-2">
        <button className="px-3 py-1 text-sm border rounded">Reset View</button>
        <button className="px-3 py-1 text-sm border rounded">Export Graph</button>
        <button className="px-3 py-1 text-sm border rounded">Import Data</button>
      </div>
    </div>
  </div>
);

// Analytics component
const Analytics = () => (
  <div className="p-6">
    <h3 className="font-semibold mb-4">Analytics Dashboard</h3>
    <p className="text-sm text-muted-foreground">Graph analytics and insights will appear here</p>
  </div>
);

// XR/AR component
const XRSettings = () => (
  <div className="p-6">
    <h3 className="font-semibold mb-4">XR/AR Configuration</h3>
    <p className="text-sm text-muted-foreground">Virtual and augmented reality settings</p>
  </div>
);

// Performance component
const Performance = () => (
  <div className="p-6">
    <h3 className="font-semibold mb-4">Performance Monitoring</h3>
    <p className="text-sm text-muted-foreground">System performance metrics and optimization</p>
  </div>
);

// Data component
const DataManagement = () => (
  <div className="p-6">
    <h3 className="font-semibold mb-4">Data Management</h3>
    <p className="text-sm text-muted-foreground">Import, export, and manage graph data</p>
  </div>
);

// Developer component
const Developer = () => (
  <div className="p-6">
    <h3 className="font-semibold mb-4">Developer Tools</h3>
    <p className="text-sm text-muted-foreground">Advanced debugging and development options</p>
  </div>
);

export function SettingsPanel() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="h-full flex flex-col">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabsList className="grid w-full grid-cols-9 shrink-0">
          <TabsTrigger value="dashboard" className="flex items-center gap-1">
            <Home className="h-3 w-3" />
            <span className="hidden lg:inline">Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex items-center gap-1">
            <Eye className="h-3 w-3" />
            <span className="hidden lg:inline">Visual</span>
          </TabsTrigger>
          <TabsTrigger value="physics" className="flex items-center gap-1">
            <Zap className="h-3 w-3" />
            <span className="hidden lg:inline">Physics</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-1">
            <BarChart3 className="h-3 w-3" />
            <span className="hidden lg:inline">Analytics</span>
          </TabsTrigger>
          <TabsTrigger value="xr" className="flex items-center gap-1">
            <Headphones className="h-3 w-3" />
            <span className="hidden lg:inline">XR/AR</span>
          </TabsTrigger>
          <TabsTrigger value="performance" className="flex items-center gap-1">
            <Activity className="h-3 w-3" />
            <span className="hidden lg:inline">Perf</span>
          </TabsTrigger>
          <TabsTrigger value="data" className="flex items-center gap-1">
            <Database className="h-3 w-3" />
            <span className="hidden lg:inline">Data</span>
          </TabsTrigger>
          <TabsTrigger value="developer" className="flex items-center gap-1">
            <Code className="h-3 w-3" />
            <span className="hidden lg:inline">Dev</span>
          </TabsTrigger>
          <TabsTrigger value="auth" className="flex items-center gap-1">
            <Shield className="h-3 w-3" />
            <span className="hidden lg:inline">Auth</span>
          </TabsTrigger>
        </TabsList>
        
        <ScrollArea className="flex-1">
          <TabsContent value="dashboard" className="mt-0">
            <Dashboard />
          </TabsContent>
          
          <TabsContent value="visualization" className="mt-0">
            <VisualizationSettings />
          </TabsContent>
          
          <TabsContent value="physics" className="mt-0">
            <PhysicsEngineControls />
          </TabsContent>
          
          <TabsContent value="analytics" className="mt-0">
            <Analytics />
          </TabsContent>
          
          <TabsContent value="xr" className="mt-0">
            <XRSettings />
          </TabsContent>
          
          <TabsContent value="performance" className="mt-0">
            <Performance />
          </TabsContent>
          
          <TabsContent value="data" className="mt-0">
            <DataManagement />
          </TabsContent>
          
          <TabsContent value="developer" className="mt-0">
            <Developer />
          </TabsContent>
          
          <TabsContent value="auth" className="mt-0">
            <AuthPanel />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}