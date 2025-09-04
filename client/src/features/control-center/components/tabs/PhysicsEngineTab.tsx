// Physics Engine Tab - GPU engine, forces, constraints and layout controls
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Separator } from '@/features/design-system/components/Separator';
import { Badge } from '@/features/design-system/components/Badge';
import { 
  Activity, 
  Zap, 
  Settings, 
  Target,
  Layers,
  Gauge,
  GitBranch
} from 'lucide-react';
import { useSettingsStore } from '@/store/settingsStore';
import { SettingControlComponent } from '@/features/settings/components/SettingControlComponent';

interface PhysicsEngineTabProps {
  searchQuery?: string;
}

export const PhysicsEngineTab: React.FC<PhysicsEngineTabProps> = ({ searchQuery = '' }) => {
  const { settings, setByPath } = useSettingsStore();
  const physics = settings?.visualisation?.physics;
  
  // GPU Engine Settings
  const gpuEngineSettings = [
    {
      path: 'visualisation.physics.enabled',
      label: 'Enable GPU Physics',
      type: 'toggle' as const,
      description: 'Enable GPU-accelerated physics simulation'
    },
    {
      path: 'visualisation.physics.computeMode',
      label: 'Compute Mode',
      type: 'select' as const,
      description: 'GPU kernel computation mode',
      options: [
        { value: '0', label: 'Basic' },
        { value: '1', label: 'Optimized' },
        { value: '2', label: 'Advanced' }
      ]
    },
    {
      path: 'visualisation.physics.gridCellSize',
      label: 'Grid Cell Size',
      type: 'slider' as const,
      min: 1.0,
      max: 100.0,
      step: 1.0,
      description: 'Spatial grid cell size for optimization'
    },
    {
      path: 'visualisation.physics.featureFlags',
      label: 'Feature Flags',
      type: 'numberInput' as const,
      min: 0,
      max: 255,
      description: 'GPU kernel feature flags bitmask'
    }
  ];
  
  // Force Dynamics Settings
  const forceDynamicsSettings = [
    {
      path: 'visualisation.physics.springK',
      label: 'Spring Force',
      type: 'slider' as const,
      min: 0.001,
      max: 10.0,
      step: 0.001,
      description: 'Spring constant for attractive forces'
    },
    {
      path: 'visualisation.physics.repelK',
      label: 'Repulsion Force',
      type: 'slider' as const,
      min: 0.001,
      max: 100.0,
      step: 0.001,
      description: 'Repulsion force strength'
    },
    {
      path: 'visualisation.physics.attractionK',
      label: 'Attraction Force',
      type: 'slider' as const,
      min: 0.0,
      max: 1.0,
      step: 0.001,
      description: 'Global attraction force strength'
    },
    {
      path: 'visualisation.physics.gravity',
      label: 'Gravity',
      type: 'slider' as const,
      min: -1.0,
      max: 1.0,
      step: 0.001,
      description: 'Central gravity force'
    },
    {
      path: 'visualisation.physics.damping',
      label: 'Damping',
      type: 'slider' as const,
      min: 0.0,
      max: 1.0,
      step: 0.001,
      description: 'Velocity damping factor'
    }
  ];
  
  // Advanced Parameters
  const advancedSettings = [
    {
      path: 'visualisation.physics.restLength',
      label: 'Rest Length',
      type: 'slider' as const,
      min: 0.1,
      max: 10.0,
      step: 0.1,
      description: 'Default spring rest length'
    },
    {
      path: 'visualisation.physics.repulsionCutoff',
      label: 'Repulsion Cutoff',
      type: 'slider' as const,
      min: 1.0,
      max: 1000.0,
      step: 1.0,
      description: 'Distance cutoff for repulsion forces'
    },
    {
      path: 'visualisation.physics.centerGravityK',
      label: 'Center Gravity',
      type: 'slider' as const,
      min: -1.0,
      max: 1.0,
      step: 0.001,
      description: 'Central gravity strength'
    },
    {
      path: 'visualisation.physics.maxForce',
      label: 'Max Force',
      type: 'numberInput' as const,
      min: 1,
      max: 1000,
      description: 'Maximum force magnitude cap'
    }
  ];
  
  // Boundary & Collision Settings
  const boundarySettings = [
    {
      path: 'visualisation.physics.enableBounds',
      label: 'Enable Boundaries',
      type: 'toggle' as const,
      description: 'Enable simulation boundaries'
    },
    {
      path: 'visualisation.physics.boundsSize',
      label: 'Boundary Size',
      type: 'slider' as const,
      min: 100,
      max: 2000,
      step: 10,
      description: 'Simulation boundary size'
    },
    {
      path: 'visualisation.physics.boundaryDamping',
      label: 'Boundary Damping',
      type: 'slider' as const,
      min: 0.0,
      max: 1.0,
      step: 0.01,
      description: 'Velocity damping at boundaries'
    },
    {
      path: 'visualisation.physics.separationRadius',
      label: 'Separation Radius',
      type: 'slider' as const,
      min: 0.1,
      max: 10.0,
      step: 0.1,
      description: 'Minimum separation between nodes'
    }
  ];
  
  const renderSettingSection = (title: string, icon: any, settings: any[], searchable: string) => {
    const Icon = icon;
    
    if (searchQuery && !searchable.toLowerCase().includes(searchQuery.toLowerCase())) {
      return null;
    }
    
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Icon className="w-5 h-5" />
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {settings.map((setting) => {
            const value = settings?.visualisation ? 
              getNestedValue(settings, setting.path.split('.')) : 
              undefined;
            
            return (
              <SettingControlComponent
                key={setting.path}
                path={setting.path}
                settingDef={setting}
                value={value}
                onChange={(newValue) => setByPath(setting.path, newValue)}
              />
            );
          })}
        </CardContent>
      </Card>
    );
  };
  
  // Helper function to get nested values
  const getNestedValue = (obj: any, path: string[]) => {
    return path.reduce((current, key) => current?.[key], obj);
  };
  
  const physicsEnabled = physics?.enabled || false;
  
  return (
    <div className="space-y-6">
      {/* Physics Engine Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Physics Engine Status
            </div>
            <Badge variant={physicsEnabled ? "default" : "secondary"}>
              {physicsEnabled ? "Active" : "Inactive"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {physics?.iterations || 0}
              </div>
              <div className="text-sm text-muted-foreground">Iterations/Frame</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {physics?.temperature?.toFixed(2) || '0.00'}
              </div>
              <div className="text-sm text-muted-foreground">Temperature</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {physics?.dt?.toFixed(4) || '0.0000'}
              </div>
              <div className="text-sm text-muted-foreground">Delta Time</div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* GPU Engine Settings */}
      {renderSettingSection(
        'GPU Engine', 
        Zap, 
        gpuEngineSettings, 
        'gpu engine compute mode grid cell size feature flags kernel'
      )}
      
      {/* Force Dynamics */}
      {renderSettingSection(
        'Force Dynamics', 
        Target, 
        forceDynamicsSettings, 
        'force dynamics spring repulsion attraction gravity damping'
      )}
      
      {/* Constraints & Layout */}
      {renderSettingSection(
        'Boundaries & Collision', 
        Layers, 
        boundarySettings, 
        'boundaries collision bounds damping separation radius'
      )}
      
      {/* Advanced Parameters */}
      {renderSettingSection(
        'Advanced Parameters', 
        Settings, 
        advancedSettings, 
        'advanced parameters rest length repulsion cutoff center gravity max force'
      )}
      
      {/* Performance Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Gauge className="w-5 h-5" />
            Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">GPU Memory Usage</span>
              <span className="text-sm font-mono">{(physics?.gpuMemory || 0).toFixed(1)} MB</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Compute Time</span>
              <span className="text-sm font-mono">{(physics?.computeTime || 0).toFixed(2)} ms</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Active Forces</span>
              <span className="text-sm font-mono">{physics?.activeForces || 0}</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PhysicsEngineTab;