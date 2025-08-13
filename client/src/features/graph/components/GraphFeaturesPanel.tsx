/**
 * GraphFeaturesPanel Component
 * Provides UI controls for innovative graph features
 */

import React, { useState, useCallback } from 'react';
import { 
  GitCompare, 
  Zap, 
  Brain, 
  Clock, 
  Users, 
  Glasses,
  Settings,
  Play,
  Pause,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Switch } from '@/features/design-system/components/Switch';
import { Label } from '@/features/design-system/components/Label';
import { Slider } from '@/features/design-system/components/Slider';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/features/design-system/components/Select';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { useGraphStore } from '../../../store/graphStore';
import { toast } from '@/features/design-system/components/Toast';

interface GraphFeaturesPanelProps {
  graphId?: string;
  compact?: boolean;
}

export const GraphFeaturesPanel: React.FC<GraphFeaturesPanelProps> = ({ 
  graphId = 'default',
  compact = false 
}) => {
  // Feature states
  const [syncEnabled, setSyncEnabled] = useState(false);
  const [comparisonEnabled, setComparisonEnabled] = useState(false);
  const [animationsEnabled, setAnimationsEnabled] = useState(true);
  const [aiInsightsEnabled, setAiInsightsEnabled] = useState(false);
  const [autoOptimize, setAutoOptimize] = useState(false);
  
  // Sync options
  const [cameraSync, setCameraSync] = useState(true);
  const [selectionSync, setSelectionSync] = useState(true);
  const [zoomSync, setZoomSync] = useState(true);
  const [transitionDuration, setTransitionDuration] = useState(300);
  
  // Interaction modes
  const [timeTravelActive, setTimeTravelActive] = useState(false);
  const [collaborationActive, setCollaborationActive] = useState(false);
  const [vrModeActive, setVrModeActive] = useState(false);

  // Get graph data from store
  const graphData = useGraphStore(state => state.graphs[graphId]);

  // Feature control handlers
  const handleSyncToggle = useCallback((enabled: boolean) => {
    setSyncEnabled(enabled);
    toast({
      title: enabled ? "Graph Synchronisation Enabled" : "Graph Synchronisation Disabled",
      description: enabled 
        ? "Both graphs will now move in sync" 
        : "Graphs can now be navigated independently"
    });
  }, []);

  const handleComparisonToggle = useCallback((enabled: boolean) => {
    setComparisonEnabled(enabled);
    if (enabled) {
      toast({
        title: "Graph Comparison Activated",
        description: "Analysing similarities and differences between graphs..."
      });
    }
  }, []);

  const handleAnimationsToggle = useCallback((enabled: boolean) => {
    setAnimationsEnabled(enabled);
    toast({
      title: enabled ? "Animations Enabled" : "Animations Disabled",
      description: enabled 
        ? "Graph transitions will be animated" 
        : "Graph updates will be instant"
    });
  }, []);

  const handleAiInsightsToggle = useCallback((enabled: boolean) => {
    setAiInsightsEnabled(enabled);
    if (enabled) {
      toast({
        title: "AI Insights Activated",
        description: "Analysing graph structure for optimisations..."
      });
    }
  }, []);

  const handleAutoOptimizeToggle = useCallback((enabled: boolean) => {
    setAutoOptimize(enabled);
    if (enabled) {
      toast({
        title: "Auto-Optimisation Enabled",
        description: "Graph layout will continuously optimise based on AI recommendations"
      });
    }
  }, []);

  const handleTimeTravelToggle = useCallback(() => {
    setTimeTravelActive(!timeTravelActive);
    toast({
      title: timeTravelActive ? "Time Travel Mode Deactivated" : "Time Travel Mode Activated",
      description: timeTravelActive 
        ? "Returned to current graph state" 
        : "Navigate through graph history with arrow keys"
    });
  }, [timeTravelActive]);

  const handleCollaborationToggle = useCallback(() => {
    setCollaborationActive(!collaborationActive);
    toast({
      title: collaborationActive ? "Collaboration Ended" : "Collaboration Started",
      description: collaborationActive 
        ? "Collaboration session closed" 
        : "Sharing session link with participants..."
    });
  }, [collaborationActive]);

  const handleVrModeToggle = useCallback(() => {
    setVrModeActive(!vrModeActive);
    toast({
      title: vrModeActive ? "VR Mode Deactivated" : "VR Mode Activated",
      description: vrModeActive 
        ? "Returned to standard view" 
        : "Entering immersive VR environment..."
    });
  }, [vrModeActive]);

  const runClustering = useCallback(() => {
    toast({
      title: "Running Clustering Analysis",
      description: "Identifying node clusters and communities..."
    });
  }, []);

  const optimiseLayout = useCallback(() => {
    toast({
      title: "Optimising Graph Layout",
      description: "Applying AI-recommended positions..."
    });
  }, []);

  if (compact) {
    // Compact view for integration in other panels
    return (
      <div className="space-y-2 p-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Graph Sync</Label>
          <Switch
            checked={syncEnabled}
            onCheckedChange={handleSyncToggle}
          />
        </div>
        <div className="flex items-center justify-between">
          <Label className="text-xs">Comparison</Label>
          <Switch
            checked={comparisonEnabled}
            onCheckedChange={handleComparisonToggle}
          />
        </div>
        <div className="flex items-center justify-between">
          <Label className="text-xs">AI Insights</Label>
          <Switch
            checked={aiInsightsEnabled}
            onCheckedChange={handleAiInsightsToggle}
          />
        </div>
        <div className="flex items-center justify-between">
          <Label className="text-xs">Animations</Label>
          <Switch
            checked={animationsEnabled}
            onCheckedChange={handleAnimationsToggle}
          />
        </div>
      </div>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold flex items-center gap-2">
          <Zap className="h-4 w-4" />
          Graph Features
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Tabs defaultValue="sync" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="sync" className="text-xs">Sync</TabsTrigger>
            <TabsTrigger value="compare" className="text-xs">Compare</TabsTrigger>
            <TabsTrigger value="ai" className="text-xs">AI</TabsTrigger>
            <TabsTrigger value="interact" className="text-xs">Interact</TabsTrigger>
          </TabsList>

          {/* Synchronisation Tab */}
          <TabsContent value="sync" className="space-y-3 mt-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="sync-toggle">Enable Synchronisation</Label>
              <Switch
                id="sync-toggle"
                checked={syncEnabled}
                onCheckedChange={handleSyncToggle}
              />
            </div>
            
            {syncEnabled && (
              <div className="space-y-2 pl-4 border-l-2 border-muted">
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
                <div className="space-y-1">
                  <Label className="text-xs">Transition Duration (ms)</Label>
                  <Slider
                    value={[transitionDuration]}
                    onValueChange={([value]) => setTransitionDuration(value)}
                    min={0}
                    max={1000}
                    step={50}
                    className="w-full"
                  />
                  <span className="text-xs text-muted-foreground">{transitionDuration}ms</span>
                </div>
              </div>
            )}
          </TabsContent>

          {/* Comparison Tab */}
          <TabsContent value="compare" className="space-y-3 mt-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="compare-toggle">Enable Comparison</Label>
              <Switch
                id="compare-toggle"
                checked={comparisonEnabled}
                onCheckedChange={handleComparisonToggle}
              />
            </div>
            
            {comparisonEnabled && (
              <div className="space-y-2">
                <Select defaultValue="structural">
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Comparison Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="structural">Structural Similarity</SelectItem>
                    <SelectItem value="semantic">Semantic Similarity</SelectItem>
                    <SelectItem value="both">Both</SelectItem>
                  </SelectContent>
                </Select>
                
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full"
                  onClick={() => toast({ title: "Running comparison analysis..." })}
                >
                  <GitCompare className="h-3 w-3 mr-1" />
                  Run Analysis
                </Button>
                
                <div className="text-xs space-y-1 p-2 bg-muted rounded">
                  <div className="flex justify-between">
                    <span>Similarity:</span>
                    <span className="font-mono">---%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Matches:</span>
                    <span className="font-mono">---</span>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>

          {/* AI Insights Tab */}
          <TabsContent value="ai" className="space-y-3 mt-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="ai-toggle">Enable AI Insights</Label>
              <Switch
                id="ai-toggle"
                checked={aiInsightsEnabled}
                onCheckedChange={handleAiInsightsToggle}
              />
            </div>
            
            {aiInsightsEnabled && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Auto-Optimise</Label>
                  <Switch
                    checked={autoOptimize}
                    onCheckedChange={handleAutoOptimizeToggle}
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={runClustering}
                  >
                    <Brain className="h-3 w-3 mr-1" />
                    Clustering
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={optimiseLayout}
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Optimise
                  </Button>
                </div>
                
                <div className="text-xs space-y-1 p-2 bg-muted rounded">
                  <div className="flex justify-between">
                    <span>Algorithm:</span>
                    <span className="font-mono">---</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Confidence:</span>
                    <span className="font-mono">---%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Clusters:</span>
                    <span className="font-mono">---</span>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>

          {/* Interaction Tab */}
          <TabsContent value="interact" className="space-y-3 mt-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="animations-toggle">Animations</Label>
              <Switch
                id="animations-toggle"
                checked={animationsEnabled}
                onCheckedChange={handleAnimationsToggle}
              />
            </div>
            
            <div className="space-y-2">
              <Button 
                variant={timeTravelActive ? "default" : "outline"}
                size="sm" 
                className="w-full"
                onClick={handleTimeTravelToggle}
              >
                <Clock className="h-3 w-3 mr-1" />
                {timeTravelActive ? "Exit Time Travel" : "Time Travel Mode"}
              </Button>
              
              <Button 
                variant={collaborationActive ? "default" : "outline"}
                size="sm" 
                className="w-full"
                onClick={handleCollaborationToggle}
              >
                <Users className="h-3 w-3 mr-1" />
                {collaborationActive ? "End Collaboration" : "Start Collaboration"}
              </Button>
              
              <Button 
                variant={vrModeActive ? "default" : "outline"}
                size="sm" 
                className="w-full"
                onClick={handleVrModeToggle}
              >
                <Glasses className="h-3 w-3 mr-1" />
                {vrModeActive ? "Exit VR Mode" : "Enter VR Mode"}
              </Button>
            </div>
            
            {(timeTravelActive || collaborationActive || vrModeActive) && (
              <div className="text-xs space-y-1 p-2 bg-muted rounded">
                {timeTravelActive && (
                  <div className="flex justify-between">
                    <span>Time Step:</span>
                    <span className="font-mono">---/---</span>
                  </div>
                )}
                {collaborationActive && (
                  <div className="flex justify-between">
                    <span>Participants:</span>
                    <span className="font-mono">1</span>
                  </div>
                )}
                {vrModeActive && (
                  <div className="flex justify-between">
                    <span>VR Status:</span>
                    <span className="font-mono text-green-500">Active</span>
                  </div>
                )}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default GraphFeaturesPanel;