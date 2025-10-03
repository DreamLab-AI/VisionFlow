/**
 * Graph Visualisation Tab Component
 * Controls advanced visualisation features with UK English localisation
 */

import React, { useState, useCallback } from 'react';
import {
  RefreshCw,
  Camera,
  MousePointer2,
  Zap,
  Eye,
  RotateCcw,
  AlertCircle,
  Palette,
  Play,
  Pause
} from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Switch } from '@/features/design-system/components/Switch';
import { Label } from '@/features/design-system/components/Label';
import { Badge } from '@/features/design-system/components/Badge';
import { Slider } from '@/features/design-system/components/Slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Separator } from '@/features/design-system/components/Separator';
import { toast } from '@/features/design-system/components/Toast';

interface GraphVisualisationTabProps {
  graphId?: string;
  onFeatureUpdate?: (feature: string, data: any) => void;
}

export const GraphVisualisationTab: React.FC<GraphVisualisationTabProps> = ({ 
  graphId = 'default',
  onFeatureUpdate
}) => {
  // Synchronisation states
  const [syncEnabled, setSyncEnabled] = useState(false);
  const [cameraSync, setCameraSync] = useState(true);
  const [selectionSync, setSelectionSync] = useState(true);
  const [zoomSync, setZoomSync] = useState(true);
  const [panSync, setPanSync] = useState(true);
  const [transitionDuration, setTransitionDuration] = useState([300]);

  // Animation states
  const [animationsEnabled, setAnimationsEnabled] = useState(true);
  const [transitionEffect, setTransitionEffect] = useState('smooth');
  const [nodeAnimations, setNodeAnimations] = useState(true);
  const [edgeAnimations, setEdgeAnimations] = useState(true);
  
  // Visual effects
  const [visualEffects, setVisualEffects] = useState({
    bloom: false,
    glow: true,
    particles: false,
    trails: false
  });

  const handleSyncToggle = useCallback((enabled: boolean) => {
    setSyncEnabled(enabled);
    onFeatureUpdate?.('synchronisation', { 
      enabled,
      options: {
        enableCameraSync: cameraSync,
        enableSelectionSync: selectionSync,
        enableZoomSync: zoomSync,
        enablePanSync: panSync,
        transitionDuration: transitionDuration[0]
      }
    });
    
    toast({
      title: enabled ? "Graph Synchronisation Enabled" : "Graph Synchronisation Disabled",
      description: enabled 
        ? "Both graphs will now move in synchronisation" 
        : "Graphs can now be navigated independently"
    });
  }, [cameraSync, selectionSync, zoomSync, panSync, transitionDuration, onFeatureUpdate]);

  const handleAnimationsToggle = useCallback((enabled: boolean) => {
    setAnimationsEnabled(enabled);
    onFeatureUpdate?.('animations', { enabled });
    
    toast({
      title: enabled ? "Animations Enabled" : "Animations Disabled",
      description: enabled 
        ? "Graph transitions will be animated smoothly" 
        : "Graph updates will be instantaneous"
    });
  }, [onFeatureUpdate]);

  const handleVisualEffectToggle = useCallback((effect: keyof typeof visualEffects, enabled: boolean) => {
    setVisualEffects(prev => ({ ...prev, [effect]: enabled }));
    onFeatureUpdate?.('visualEffects', { ...visualEffects, [effect]: enabled });
    
    toast({
      title: `${effect.charAt(0).toUpperCase() + effect.slice(1)} ${enabled ? 'Enabled' : 'Disabled'}`,
      description: `Visual ${effect} effect has been ${enabled ? 'activated' : 'deactivated'}`
    });
  }, [visualEffects, onFeatureUpdate]);

  const resetCameraPosition = useCallback(() => {
    onFeatureUpdate?.('camera', { action: 'reset' });
    toast({
      title: "Camera Reset",
      description: "Camera position restored to default view"
    });
  }, [onFeatureUpdate]);

  const createCameraBookmark = useCallback(() => {
    onFeatureUpdate?.('camera', { action: 'bookmark' });
    toast({
      title: "Camera Bookmark Created",
      description: "Current camera position saved for quick access"
    });
  }, [onFeatureUpdate]);

  return (
    <div className="space-y-4">
      {/* Graph Synchronisation */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <RefreshCw className="h-4 w-4" />
            Graph Synchronisation
            <Badge variant="secondary" className="text-xs">
              <AlertCircle className="h-3 w-3 mr-1" />
              Partial
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="sync-toggle">Enable Synchronisation</Label>
            <Switch
              id="sync-toggle"
              checked={syncEnabled}
              onCheckedChange={handleSyncToggle}
            />
          </div>
          
          {syncEnabled && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="grid grid-cols-2 gap-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Camera Sync</Label>
                  <Switch
                    checked={cameraSync}
                    onCheckedChange={setCameraSync}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Selection Sync</Label>
                  <Switch
                    checked={selectionSync}
                    onCheckedChange={setSelectionSync}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Zoom Sync</Label>
                  <Switch
                    checked={zoomSync}
                    onCheckedChange={setZoomSync}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Pan Sync</Label>
                  <Switch
                    checked={panSync}
                    onCheckedChange={setPanSync}
                  />
                </div>
              </div>
              
              <div className="space-y-1">
                <Label className="text-xs">Transition Duration (ms)</Label>
                <Slider
                  value={transitionDuration}
                  onValueChange={setTransitionDuration}
                  min={0}
                  max={1000}
                  step={50}
                  className="w-full"
                />
                <span className="text-xs text-muted-foreground">{transitionDuration[0]}ms</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Animation System */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Animation System
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="animations-toggle">Enable Animations</Label>
            <Switch
              id="animations-toggle"
              checked={animationsEnabled}
              onCheckedChange={handleAnimationsToggle}
            />
          </div>
          
          {animationsEnabled && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="grid grid-cols-2 gap-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Node Animations</Label>
                  <Switch
                    checked={nodeAnimations}
                    onCheckedChange={setNodeAnimations}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Edge Animations</Label>
                  <Switch
                    checked={edgeAnimations}
                    onCheckedChange={setEdgeAnimations}
                  />
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-xs">Active Animations:</span>
                <span className="text-xs font-mono">
                  {(nodeAnimations ? 1 : 0) + (edgeAnimations ? 1 : 0)} types
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Visual Effects */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Visual Effects
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(visualEffects).map(([effect, enabled]) => (
              <div key={effect} className="flex items-center justify-between">
                <Label className="text-xs capitalize">{effect}</Label>
                <Switch
                  checked={enabled}
                  onCheckedChange={(checked) => handleVisualEffectToggle(effect as keyof typeof visualEffects, checked)}
                />
              </div>
            ))}
          </div>
          
          <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
            <strong>Note:</strong> Some visual effects may impact performance on lower-end devices.
          </div>
        </CardContent>
      </Card>

      {/* Camera Controls */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Camera className="h-4 w-4" />
            Camera Controls
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={resetCameraPosition}
              className="w-full"
            >
              <RotateCcw className="h-3 w-3 mr-1" />
              Reset View
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={createCameraBookmark}
              className="w-full"
            >
              <Eye className="h-3 w-3 mr-1" />
              Bookmark
            </Button>
          </div>
          
          <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
            Camera bookmarking and advanced positioning controls are under development.
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GraphVisualisationTab;