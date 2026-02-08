/**
 * Physics Control Panel
 *
 * Provides comprehensive controls for the physics simulation engine via Phase 5 Physics API:
 * - Start/stop/pause/step simulation
 * - Adjust simulation parameters in real-time
 * - Optimize layout with different algorithms
 * - Monitor GPU status and statistics
 * - Reset simulation state
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Slider } from '@/features/design-system/components/Slider';
import { Label } from '@/features/design-system/components/Label';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { useToast } from '@/features/design-system/components/Toast';
import { TooltipRoot, TooltipContent, TooltipProvider, TooltipTrigger } from '@/features/design-system/components/Tooltip';
import { Play, Square, RefreshCw, Zap, Activity, Cpu, Info, AlertCircle, CheckCircle } from 'lucide-react';
import { usePhysicsService, SimulationParameters } from '../hooks/usePhysicsService';

interface PhysicsControlPanelProps {
  className?: string;
}

export function PhysicsControlPanel({ className }: PhysicsControlPanelProps) {
  const { toast } = useToast();
  const {
    status,
    loading,
    error,
    startSimulation,
    stopSimulation,
    updateParameters,
    performStep,
    resetSimulation,
    optimizeLayout,
  } = usePhysicsService();

  // Local parameter state for UI controls
  const [localParams, setLocalParams] = useState<SimulationParameters>({
    time_step: 0.016,
    damping: 0.8,
    spring_constant: 1.0,
    repulsion_strength: 1.5,
    attraction_strength: 1.0,
    max_velocity: 100.0,
    convergence_threshold: 0.01,
    max_iterations: 1000,
    auto_stop_on_convergence: false,
  });

  const [optimizationAlgorithm, setOptimizationAlgorithm] = useState('force_directed');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const handleParameterChange = (key: keyof SimulationParameters, value: number | boolean) => {
    setLocalParams((prev) => ({ ...prev, [key]: value }));
    setHasUnsavedChanges(true);
  };

  const handleApplyParameters = async () => {
    try {
      await updateParameters(localParams);
      setHasUnsavedChanges(false);
      toast({
        title: 'Parameters Updated',
        description: 'Physics simulation parameters have been updated.',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Update Failed',
        description: err.message || 'Failed to update parameters',
        variant: 'destructive',
      });
    }
  };

  const handleStart = async () => {
    try {
      await startSimulation(localParams);
      toast({
        title: 'Simulation Started',
        description: 'Physics simulation is now running.',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Start Failed',
        description: err.message || 'Failed to start simulation',
        variant: 'destructive',
      });
    }
  };

  const handleStop = async () => {
    try {
      await stopSimulation();
      toast({
        title: 'Simulation Stopped',
        description: 'Physics simulation has been stopped.',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Stop Failed',
        description: err.message || 'Failed to stop simulation',
        variant: 'destructive',
      });
    }
  };

  const handleStep = async () => {
    try {
      await performStep();
      toast({
        title: 'Step Completed',
        description: 'Performed single simulation step.',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Step Failed',
        description: err.message || 'Failed to perform step',
        variant: 'destructive',
      });
    }
  };

  const handleReset = async () => {
    try {
      await resetSimulation();
      toast({
        title: 'Simulation Reset',
        description: 'Physics simulation has been reset to initial state.',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Reset Failed',
        description: err.message || 'Failed to reset simulation',
        variant: 'destructive',
      });
    }
  };

  const handleOptimize = async () => {
    try {
      const result = await optimizeLayout({
        algorithm: optimizationAlgorithm,
        max_iterations: 1000,
        target_energy: 0.01,
      });
      toast({
        title: 'Layout Optimized',
        description: `Updated ${result.nodes_updated} nodes (score: ${result.optimization_score.toFixed(4)})`,
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Optimization Failed',
        description: err.message || 'Failed to optimize layout',
        variant: 'destructive',
      });
    }
  };

  const StatusIndicator = () => {
    const running = status?.running ?? false;
    return (
      <div className="flex items-center gap-2">
        {running ? (
          <>
            <Activity className="h-4 w-4 text-green-500 animate-pulse" />
            <Badge variant="default" className="bg-green-500">
              Running
            </Badge>
          </>
        ) : (
          <>
            <Square className="h-4 w-4 text-gray-400" />
            <Badge variant="secondary">Stopped</Badge>
          </>
        )}
      </div>
    );
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Physics Simulation Control
            </CardTitle>
            <CardDescription>
              Control and monitor the physics simulation engine
            </CardDescription>
          </div>
          <StatusIndicator />
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <Tabs defaultValue="controls" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="controls">Controls</TabsTrigger>
            <TabsTrigger value="parameters">Parameters</TabsTrigger>
            <TabsTrigger value="status">Status</TabsTrigger>
          </TabsList>

          {/* Controls Tab */}
          <TabsContent value="controls" className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Button
                onClick={handleStart}
                disabled={status?.running || loading}
                className="w-full"
              >
                <Play className="mr-2 h-4 w-4" />
                Start
              </Button>
              <Button
                onClick={handleStop}
                disabled={!status?.running || loading}
                variant="destructive"
                className="w-full"
              >
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <TooltipProvider>
                <TooltipRoot>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={handleStep}
                      disabled={loading}
                      variant="outline"
                      className="w-full"
                    >
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Step
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Perform single simulation step for debugging</p>
                  </TooltipContent>
                </TooltipRoot>
              </TooltipProvider>

              <TooltipProvider>
                <TooltipRoot>
                  <TooltipTrigger asChild>
                    <Button
                      onClick={handleReset}
                      disabled={loading}
                      variant="outline"
                      className="w-full"
                    >
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Reset
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Reset simulation to initial state</p>
                  </TooltipContent>
                </TooltipRoot>
              </TooltipProvider>
            </div>

            <div className="border-t pt-4 space-y-3">
              <Label className="text-sm font-medium">Layout Optimization</Label>
              <Select value={optimizationAlgorithm} onValueChange={setOptimizationAlgorithm}>
                <SelectTrigger>
                  <SelectValue placeholder="Select algorithm" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="force_directed">Force-Directed</SelectItem>
                  <SelectItem value="spring_embedder">Spring Embedder</SelectItem>
                  <SelectItem value="fruchterman_reingold">Fruchterman-Reingold</SelectItem>
                  <SelectItem value="kamada_kawai">Kamada-Kawai</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={handleOptimize} disabled={loading} className="w-full">
                <Zap className="mr-2 h-4 w-4" />
                Optimize Layout
              </Button>
            </div>
          </TabsContent>

          {/* Parameters Tab */}
          <TabsContent value="parameters" className="space-y-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="spring">Spring Constant</Label>
                  <span className="text-sm text-muted-foreground">
                    {localParams.spring_constant?.toFixed(2)}
                  </span>
                </div>
                <Slider
                  id="spring"
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  value={[localParams.spring_constant || 1.0]}
                  onValueChange={([v]) => handleParameterChange('spring_constant', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="damping">Damping</Label>
                  <span className="text-sm text-muted-foreground">
                    {localParams.damping?.toFixed(2)}
                  </span>
                </div>
                <Slider
                  id="damping"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={[localParams.damping || 0.8]}
                  onValueChange={([v]) => handleParameterChange('damping', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="repulsion">Repulsion Strength</Label>
                  <span className="text-sm text-muted-foreground">
                    {localParams.repulsion_strength?.toFixed(2)}
                  </span>
                </div>
                <Slider
                  id="repulsion"
                  min={0.5}
                  max={3.0}
                  step={0.1}
                  value={[localParams.repulsion_strength || 1.5]}
                  onValueChange={([v]) => handleParameterChange('repulsion_strength', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="attraction">Attraction Strength</Label>
                  <span className="text-sm text-muted-foreground">
                    {localParams.attraction_strength?.toFixed(2)}
                  </span>
                </div>
                <Slider
                  id="attraction"
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  value={[localParams.attraction_strength || 1.0]}
                  onValueChange={([v]) => handleParameterChange('attraction_strength', v)}
                />
              </div>

              <Button
                onClick={handleApplyParameters}
                disabled={!hasUnsavedChanges || loading}
                className="w-full mt-4"
              >
                {hasUnsavedChanges ? (
                  <>
                    <AlertCircle className="mr-2 h-4 w-4" />
                    Apply Changes
                  </>
                ) : (
                  <>
                    <CheckCircle className="mr-2 h-4 w-4" />
                    No Changes
                  </>
                )}
              </Button>
            </div>
          </TabsContent>

          {/* Status Tab */}
          <TabsContent value="status" className="space-y-4">
            {status?.gpu_status && (
              <div className="space-y-3">
                <Label className="text-sm font-medium flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  GPU Status
                </Label>
                <div className="rounded-lg border p-3 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Device:</span>
                    <span className="font-medium">{status.gpu_status.device_name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Compute Capability:</span>
                    <span className="font-medium">{status.gpu_status.compute_capability}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Memory:</span>
                    <span className="font-medium">{status.gpu_status.total_memory_mb} MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Free Memory:</span>
                    <span className="font-medium">{status.gpu_status.free_memory_mb} MB</span>
                  </div>
                </div>
              </div>
            )}

            {status?.statistics && (
              <div className="space-y-3">
                <Label className="text-sm font-medium flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Simulation Statistics
                </Label>
                <div className="rounded-lg border p-3 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Steps:</span>
                    <span className="font-medium">{status.statistics.total_steps}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Avg Step Time:</span>
                    <span className="font-medium">
                      {status.statistics.average_step_time_ms.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Avg Energy:</span>
                    <span className="font-medium">
                      {status.statistics.average_energy.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">GPU Memory Used:</span>
                    <span className="font-medium">
                      {status.statistics.gpu_memory_used_mb.toFixed(1)} MB
                    </span>
                  </div>
                </div>
              </div>
            )}

            {!status && (
              <div className="flex items-center justify-center p-8 text-muted-foreground">
                <Info className="mr-2 h-4 w-4" />
                <span>No status information available</span>
              </div>
            )}
          </TabsContent>
        </Tabs>

        {error && (
          <div className="rounded-lg border border-destructive bg-destructive/10 p-3 flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-destructive">Error</p>
              <p className="text-sm text-destructive/90">{error}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
