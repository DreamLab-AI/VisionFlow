

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { 
  Brain, 
  Cpu,
  RefreshCw,
  Target,
  Layers,
  Sparkles,
  TrendingUp,
  Zap,
  AlertCircle,
  Settings,
  Play,
  Pause
} from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Switch } from '@/features/design-system/components/Switch';
import { Label } from '@/features/design-system/components/Label';
import { Badge } from '@/features/design-system/components/Badge';
import { Slider } from '@/features/design-system/components/Slider';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/features/design-system/components/Select';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Progress } from '@/features/design-system/components/Progress';
import { toast } from '@/features/design-system/components/Toast';
import {
  optimizationApi,
  optimizationWebSocket,
  OptimizationTask,
  OptimizationResult,
  OptimizationParams,
  GraphData,
  OptimizationWebSocketEvent
} from '@/api/optimizationApi';
import { createLogger } from '../../../../utils/loggerConfig';

const logger = createLogger('GraphOptimisationTab');

interface GraphOptimisationTabProps {
  graphId?: string;
  graphData?: GraphData;
  onFeatureUpdate?: (feature: string, data: any) => void;
  onOptimizationComplete?: (result: OptimizationResult) => void;
}

export const GraphOptimisationTab: React.FC<GraphOptimisationTabProps> = ({
  graphId = 'default',
  graphData,
  onFeatureUpdate,
  onOptimizationComplete
}) => {
  
  const [aiInsightsEnabled, setAiInsightsEnabled] = useState(false);
  const [autoOptimise, setAutoOptimise] = useState(false);
  const [optimisationLevel, setOptimisationLevel] = useState([3]);
  const [isOptimising, setIsOptimising] = useState(false);
  const [optimisationProgress, setOptimisationProgress] = useState(0);
  
  
  const [performanceMode, setPerformanceMode] = useState('balanced');
  const [clusteringEnabled, setClusteringEnabled] = useState(false);
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('force-directed');

  
  const [optimisationResults, setOptimisationResults] = useState<OptimizationResult | null>(null);
  const [currentTask, setCurrentTask] = useState<OptimizationTask | null>(null);
  const [gpuStatus, setGpuStatus] = useState<{ available: boolean; utilization: number }>({ available: false, utilization: 0 });
  const [availableAlgorithms, setAvailableAlgorithms] = useState<string[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [canCancel, setCanCancel] = useState(false);


  const wsConnectedRef = useRef(false);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const clusterPollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const currentTaskRef = useRef<OptimizationTask | null>(null);
  const onOptimizationCompleteRef = useRef(onOptimizationComplete);

  // Keep refs in sync so WS handlers always see latest values
  useEffect(() => {
    currentTaskRef.current = currentTask;
  }, [currentTask]);

  useEffect(() => {
    onOptimizationCompleteRef.current = onOptimizationComplete;
  }, [onOptimizationComplete]);


  useEffect(() => {
    const initialize = async () => {
      try {
        
        if (!wsConnectedRef.current) {
          await optimizationWebSocket.connect();
          wsConnectedRef.current = true;

          
          optimizationWebSocket.addEventListener('optimization_progress', handleOptimizationProgress);
          optimizationWebSocket.addEventListener('optimization_complete', handleOptimizationComplete);
          optimizationWebSocket.addEventListener('optimization_error', handleOptimizationError);
          optimizationWebSocket.addEventListener('gpu_status', handleGpuStatusUpdate);
        }

        
        const [gpuStatusData, algorithms, metrics] = await Promise.all([
          optimizationApi.getGpuStatus(),
          optimizationApi.getAvailableAlgorithms(),
          optimizationApi.getPerformanceMetrics()
        ]);

        setGpuStatus({
          available: gpuStatusData.available,
          utilization: gpuStatusData.utilization
        });
        setAvailableAlgorithms(algorithms);
        setPerformanceMetrics(metrics);

      } catch (error) {
        logger.error('Failed to initialize optimization system:', error);
        toast({
          title: "Initialization Error",
          description: "Failed to connect to GPU optimization backend",
          variant: "destructive"
        });
      }
    };

    initialize();

    return () => {
      
      if (wsConnectedRef.current) {
        optimizationWebSocket.removeEventListener('optimization_progress', handleOptimizationProgress);
        optimizationWebSocket.removeEventListener('optimization_complete', handleOptimizationComplete);
        optimizationWebSocket.removeEventListener('optimization_error', handleOptimizationError);
        optimizationWebSocket.removeEventListener('gpu_status', handleGpuStatusUpdate);
        optimizationWebSocket.disconnect();
        wsConnectedRef.current = false;
      }

      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }

      if (clusterPollTimeoutRef.current) {
        clearTimeout(clusterPollTimeoutRef.current);
        clusterPollTimeoutRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  
  // WS handlers read from refs to avoid stale closures
  const handleOptimizationProgress = useCallback((event: OptimizationWebSocketEvent) => {
    if (event.taskId === currentTaskRef.current?.taskId) {
      setOptimisationProgress(event.data.progress || 0);
      setCurrentTask(prev => prev ? { ...prev, progress: event.data.progress || 0 } : null);
    }
  }, []);

  const handleOptimizationComplete = useCallback((event: OptimizationWebSocketEvent) => {
    if (event.taskId === currentTaskRef.current?.taskId) {
      setIsOptimising(false);
      setCanCancel(false);
      setOptimisationProgress(100);


      optimizationApi.getOptimizationResults(event.taskId!).then(result => {
        setOptimisationResults(result);
        onOptimizationCompleteRef.current?.(result);

        toast({
          title: "Optimization Complete",
          description: `Layout improved with ${(result.performanceGain * 100).toFixed(0)}% performance gain using ${result.algorithm}`
        });
      });
    }
  }, []);

  const handleOptimizationError = useCallback((event: OptimizationWebSocketEvent) => {
    if (event.taskId === currentTaskRef.current?.taskId) {
      setIsOptimising(false);
      setCanCancel(false);

      toast({
        title: "Optimization Failed",
        description: event.data.error || "GPU optimization encountered an error",
        variant: "destructive"
      });
    }
  }, []);

  const handleGpuStatusUpdate = useCallback((event: OptimizationWebSocketEvent) => {
    setGpuStatus({
      available: event.data.available,
      utilization: event.data.utilization
    });
  }, []);

  const handleAiInsightsToggle = useCallback((enabled: boolean) => {
    setAiInsightsEnabled(enabled);
    onFeatureUpdate?.('aiInsights', { enabled, autoOptimise });
    
    if (enabled) {
      toast({
        title: "AI Insights Activated",
        description: "Analysing graph structure for optimisation opportunities..."
      });
    } else {
      toast({
        title: "AI Insights Deactivated",
        description: "Automatic optimisation disabled"
      });
    }
  }, [autoOptimise, onFeatureUpdate]);

  const handleAutoOptimiseToggle = useCallback((enabled: boolean) => {
    setAutoOptimise(enabled);
    onFeatureUpdate?.('aiInsights', { enabled: aiInsightsEnabled, autoOptimise: enabled });
    
    if (enabled) {
      toast({
        title: "Auto-Optimisation Enabled",
        description: "Graph layout will continuously optimise based on AI recommendations"
      });
    }
  }, [aiInsightsEnabled, onFeatureUpdate]);

  const runLayoutOptimisation = useCallback(async () => {
    if (!graphData) {
      toast({
        title: "No Graph Data",
        description: "Please load a graph before running optimization",
        variant: "destructive"
      });
      return;
    }

    if (!gpuStatus.available) {
      toast({
        title: "GPU Unavailable",
        description: "GPU acceleration is required for optimization",
        variant: "destructive"
      });
      return;
    }

    setIsOptimising(true);
    setOptimisationProgress(0);
    setCanCancel(true);

    toast({
      title: "Starting GPU Optimization",
      description: "Initializing stress majorization algorithm..."
    });

    try {
      const optimizationParams: OptimizationParams = {
        algorithm: layoutAlgorithm as OptimizationParams['algorithm'],
        optimizationLevel: optimisationLevel[0],
        clusteringEnabled,
        gpuAcceleration: true,
        performanceMode: performanceMode as OptimizationParams['performanceMode'],
        maxIterations: 1000,
        convergenceThreshold: 0.01
      };

      const task = await optimizationApi.triggerStressMajorization(graphData, optimizationParams);
      setCurrentTask(task);

      
      if (!optimizationWebSocket.isConnected()) {
        pollIntervalRef.current = setInterval(async () => {
          try {
            const status = await optimizationApi.getOptimizationStatus(task.taskId);
            setOptimisationProgress(status.progress);
            setCurrentTask(status);

            if (status.status === 'completed') {
              clearInterval(pollIntervalRef.current!);
              const result = await optimizationApi.getOptimizationResults(task.taskId);
              setOptimisationResults(result);
              setIsOptimising(false);
              setCanCancel(false);
              onOptimizationComplete?.(result);
            } else if (status.status === 'failed') {
              clearInterval(pollIntervalRef.current!);
              setIsOptimising(false);
              setCanCancel(false);
              throw new Error(status.error || 'Optimization failed');
            }
          } catch (error) {
            clearInterval(pollIntervalRef.current!);
            throw error;
          }
        }, 1000);
      }

    } catch (error) {
      setIsOptimising(false);
      setCanCancel(false);
      toast({
        title: "Optimization Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      });
    }
  }, [graphData, gpuStatus.available, layoutAlgorithm, optimisationLevel, clusteringEnabled, performanceMode, onOptimizationComplete]);

  const runClusterAnalysis = useCallback(async () => {
    if (!graphData) {
      toast({
        title: "No Graph Data",
        description: "Please load a graph before running cluster analysis",
        variant: "destructive"
      });
      return;
    }

    setIsOptimising(true);
    toast({
      title: "Running GPU Cluster Analysis",
      description: "Identifying node communities using GPU acceleration..."
    });

    try {
      const task = await optimizationApi.runClusteringAnalysis(graphData, 'louvain');
      setCurrentTask(task);


      const pollClustering = async () => {
        try {
          const status = await optimizationApi.getOptimizationStatus(task.taskId);

          if (status.status === 'completed') {
            clusterPollTimeoutRef.current = null;
            const result = await optimizationApi.getOptimizationResults(task.taskId);
            setOptimisationResults(result);
            setIsOptimising(false);

            toast({
              title: "Clustering Analysis Complete",
              description: `Identified ${result.clusters} optimal clusters with ${(result.metrics.clustering.modularity * 100).toFixed(1)}% modularity`
            });
          } else if (status.status === 'failed') {
            clusterPollTimeoutRef.current = null;
            throw new Error(status.error || 'Clustering failed');
          } else {
            clusterPollTimeoutRef.current = setTimeout(pollClustering, 1000);
          }
        } catch (error) {
          clusterPollTimeoutRef.current = null;
          setIsOptimising(false);
          toast({
            title: "Clustering Failed",
            description: error instanceof Error ? error.message : "Unknown error occurred",
            variant: "destructive"
          });
        }
      };

      clusterPollTimeoutRef.current = setTimeout(pollClustering, 1000);

    } catch (error) {
      setIsOptimising(false);
      toast({
        title: "Clustering Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      });
    }
  }, [graphData]);

  const cancelOptimization = useCallback(async () => {
    if (currentTask) {
      try {
        await optimizationApi.cancelOptimization(currentTask.taskId);
        setIsOptimising(false);
        setCanCancel(false);
        setCurrentTask(null);

        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
        }

        toast({
          title: "Optimization Cancelled",
          description: "GPU optimization task has been cancelled"
        });
      } catch (error) {
        toast({
          title: "Cancellation Failed",
          description: "Could not cancel optimization task",
          variant: "destructive"
        });
      }
    }
  }, [currentTask]);

  const applyRecommendation = useCallback((recommendation: any) => {
    toast({
      title: "Applying Recommendation",
      description: recommendation.description
    });
    
    onFeatureUpdate?.('recommendation', recommendation);
  }, [onFeatureUpdate]);

  return (
    <div className="space-y-4">
      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Brain className="h-4 w-4" />
            AI-Powered Optimisation
            <Badge variant={gpuStatus.available ? "default" : "destructive"} className="text-xs">
              <AlertCircle className="h-3 w-3 mr-1" />
              {gpuStatus.available ? `GPU ${gpuStatus.utilization.toFixed(0)}%` : "GPU Unavailable"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="ai-insights-toggle">Enable AI Insights</Label>
            <Switch
              id="ai-insights-toggle"
              checked={aiInsightsEnabled}
              onCheckedChange={handleAiInsightsToggle}
            />
          </div>
          
          {aiInsightsEnabled && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Auto-Optimise</Label>
                <Switch
                  checked={autoOptimise}
                  onCheckedChange={handleAutoOptimiseToggle}
                />
              </div>
              
              <div className="space-y-1">
                <Label className="text-xs">Optimisation Level</Label>
                <Slider
                  value={optimisationLevel}
                  onValueChange={setOptimisationLevel}
                  min={1}
                  max={5}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Conservative</span>
                  <span>Level {optimisationLevel[0]}</span>
                  <span>Aggressive</span>
                </div>
              </div>

              <Select value={layoutAlgorithm} onValueChange={setLayoutAlgorithm}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select Algorithm" />
                </SelectTrigger>
                <SelectContent>
                  {availableAlgorithms.length > 0 ? (
                    availableAlgorithms.map(algorithm => (
                      <SelectItem key={algorithm} value={algorithm}>
                        {algorithm.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                      </SelectItem>
                    ))
                  ) : (
                    <>
                      <SelectItem value="stress-majorization">Stress Majorization (GPU)</SelectItem>
                      <SelectItem value="force-directed">Force-Directed</SelectItem>
                      <SelectItem value="hierarchical">Hierarchical</SelectItem>
                      <SelectItem value="adaptive">Adaptive (AI)</SelectItem>
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>
          )}
        </CardContent>
      </Card>

      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <RefreshCw className="h-4 w-4" />
            Optimisation Tools
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {isOptimising && (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <Label className="text-xs">
                  {currentTask?.algorithm || 'Optimization'} Progress
                </Label>
                <div className="flex items-center gap-2">
                  <span className="text-xs">{optimisationProgress.toFixed(1)}%</span>
                  {canCancel && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={cancelOptimization}
                      className="h-5 px-2 text-xs"
                    >
                      <Pause className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              </div>
              <Progress value={optimisationProgress} className="w-full" />
              {currentTask?.estimatedCompletion && (
                <div className="text-xs text-muted-foreground">
                  Est. completion: {new Date(currentTask.estimatedCompletion).toLocaleTimeString()}
                </div>
              )}
            </div>
          )}
          
          <div className="grid grid-cols-2 gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={runLayoutOptimisation}
              disabled={isOptimising || !gpuStatus.available || !graphData}
              className="w-full"
            >
              <Target className="h-3 w-3 mr-1" />
              {isOptimising ? "Optimising..." : "Layout"}
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={runClusterAnalysis}
              disabled={isOptimising || !gpuStatus.available || !graphData}
              className="w-full"
            >
              <Layers className="h-3 w-3 mr-1" />
              Clustering
            </Button>
          </div>

          <div className="flex items-center justify-between">
            <Label className="text-xs">Enable Clustering</Label>
            <Switch
              checked={clusteringEnabled}
              onCheckedChange={setClusteringEnabled}
            />
          </div>

          {optimisationResults && (
            <div className="text-xs space-y-3 p-3 bg-muted rounded-md">
              <div className="font-semibold text-primary flex items-center gap-2">
                Optimization Results
                <Badge variant="secondary" className="text-xs">
                  GPU Accelerated
                </Badge>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="flex justify-between">
                  <span>Algorithm:</span>
                  <span className="font-mono text-xs">{optimisationResults.algorithm}</span>
                </div>
                <div className="flex justify-between">
                  <span>Confidence:</span>
                  <span className="font-mono text-green-600">
                    {(optimisationResults.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Performance:</span>
                  <span className="font-mono text-blue-600">
                    +{(optimisationResults.performanceGain * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Clusters:</span>
                  <span className="font-mono">{optimisationResults.clusters}</span>
                </div>
                <div className="flex justify-between">
                  <span>Iterations:</span>
                  <span className="font-mono">{optimisationResults.iterations}</span>
                </div>
                <div className="flex justify-between">
                  <span>GPU Usage:</span>
                  <span className="font-mono text-purple-600">
                    {optimisationResults.gpuUtilization.toFixed(1)}%
                  </span>
                </div>
              </div>
              {optimisationResults.metrics && (
                <div className="space-y-2 pt-2 border-t">
                  <div className="font-semibold">Metrics</div>
                  <div className="grid grid-cols-2 gap-1 text-xs">
                    <div>Stress Reduction: {(optimisationResults.metrics.stressMajorization.stressReduction * 100).toFixed(1)}%</div>
                    <div>Modularity: {(optimisationResults.metrics.clustering.modularity * 100).toFixed(1)}%</div>
                    <div>Compute Time: {optimisationResults.metrics.performance.computeTime.toFixed(2)}s</div>
                    <div>Efficiency: {(optimisationResults.metrics.performance.efficiency * 100).toFixed(1)}%</div>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {}
      {optimisationResults && optimisationResults.recommendations && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              AI Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {optimisationResults.recommendations.map((rec, index: number) => (
              <div key={index} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={rec.priority === 'critical' ? 'destructive' : rec.priority === 'high' ? 'destructive' : rec.priority === 'medium' ? 'default' : 'secondary'}
                      className="text-xs"
                    >
                      {rec.priority}
                    </Badge>
                    <span className="text-xs font-semibold capitalize">{rec.type}</span>
                    {rec.confidence && (
                      <span className="text-xs text-muted-foreground">({(rec.confidence * 100).toFixed(0)}%)</span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{rec.description}</p>
                </div>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => applyRecommendation(rec)}
                  className="ml-2"
                >
                  Apply
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Cpu className="h-4 w-4" />
            Performance Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label className="text-xs">Performance Mode</Label>
            <Select value={performanceMode} onValueChange={setPerformanceMode}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select Mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="battery">Battery Saver</SelectItem>
                <SelectItem value="balanced">Balanced</SelectItem>
                <SelectItem value="performance">High Performance</SelectItem>
                <SelectItem value="extreme">Extreme Performance</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {performanceMetrics && (
            <div className="text-xs space-y-2 p-2 bg-muted/50 rounded">
              <div className="font-semibold">System Status</div>
              <div className="grid grid-cols-2 gap-2">
                <div>GPU Utilization: {(performanceMetrics.gpu?.utilization as number | undefined)?.toFixed(1) || 'N/A'}%</div>
                <div>Active Tasks: {(performanceMetrics.optimization?.activeTasksCount as number | undefined) || 0}</div>
                <div>Success Rate: {(performanceMetrics.optimization?.successRate as number | undefined) ? ((performanceMetrics.optimization.successRate as number) * 100).toFixed(1) : 'N/A'}%</div>
                <div>Avg Execution: {(performanceMetrics.optimization?.averageExecutionTime as number | undefined)?.toFixed(1) || 'N/A'}s</div>
              </div>
            </div>
          )}

          <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
            <strong>GPU Acceleration:</strong> {gpuStatus.available ? 'Available and active' : 'Unavailable - using CPU fallback'}.
            Real-time optimization with WebSocket progress tracking enabled.
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GraphOptimisationTab;