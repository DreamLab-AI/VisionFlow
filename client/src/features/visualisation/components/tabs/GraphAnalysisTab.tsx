

import React, { useState, useCallback, useEffect } from 'react';
import {
  GitCompare,
  Brain,
  TrendingUp,
  BarChart3,
  Network,
  Target,
  AlertCircle,
  Cpu,
  Activity,
  Zap,
  Clock
} from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Switch } from '@/features/design-system/components/Switch';
import { Label } from '@/features/design-system/components/Label';
import { Badge } from '@/features/design-system/components/Badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/features/design-system/components/Select';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Separator } from '@/features/design-system/components/Separator';
import { toast } from '@/features/design-system/components/Toast';
import { ShortestPathControls } from '@/features/analytics/components/ShortestPathControls';
import { useAnalytics } from '@/hooks/useAnalytics';
import type { GraphNode, GraphEdge } from '@/features/graph/types/graphTypes';

interface GraphAnalysisTabProps {
  graphId?: string;
  graphData?: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  otherGraphData?: any;
}

export const GraphAnalysisTab: React.FC<GraphAnalysisTabProps> = ({
  graphId = 'default',
  graphData,
  otherGraphData
}) => {
  
  const {
    structuralAnalysis,
    semanticAnalysis,
    clusteringResults,
    performanceStats,
    isGPUEnabled,
    isAnalyzing,
    error,
    runAnalysis,
    cancelTask,
    hasActiveTasks,
    activeTasks,
    refresh
  } = useAnalytics({
    autoRefreshStats: true,
    enableWebSocket: true
  });

  
  const [comparisonEnabled, setComparisonEnabled] = useState(false);
  const [analysisType, setAnalysisType] = useState<'structural' | 'semantic' | 'both'>('both');
  const [metricsEnabled, setMetricsEnabled] = useState(true);
  const [autoAnalysis, setAutoAnalysis] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  const handleComparisonToggle = useCallback((enabled: boolean) => {
    setComparisonEnabled(enabled);
    if (enabled && (!graphData || !otherGraphData)) {
      toast({
        title: "Insufficient Data",
        description: "Two graphs are required for comparison analysis",
        variant: "destructive"
      });
      setComparisonEnabled(false);
      return;
    }

    if (enabled) {
      toast({
        title: "Graph Comparison Activated",
        description: "Ready to analyse similarities and differences between graphs"
      });
    } else {
      toast({
        title: "Graph Comparison Deactivated",
        description: "Comparison analysis stopped"
      });
    }
  }, [graphData, otherGraphData]);

  const runStructuralAnalysis = useCallback(async () => {
    if (!graphData?.nodes || !graphData?.edges) {
      toast({
        title: "No Graph Data",
        description: "Please load graph data first",
        variant: "destructive"
      });
      return;
    }

    try {
      toast({
        title: "Running Structural Analysis",
        description: "Analysing graph topology with GPU acceleration..."
      });

      const taskId = await runAnalysis({
        type: 'structural',
        graphData,
        options: {
          include_centrality: true,
          include_clustering: true,
          include_connectivity: true,
          cluster_resolution: 1.0
        }
      });

      setCurrentTaskId(taskId);

      toast({
        title: "Analysis Started",
        description: `Task ID: ${taskId}. Progress will be displayed below.`
      });
    } catch (error: any) {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to start structural analysis",
        variant: "destructive"
      });
    }
  }, [graphData, runAnalysis]);

  const runSemanticAnalysis = useCallback(async () => {
    if (!graphData?.nodes || !graphData?.edges) {
      toast({
        title: "No Graph Data",
        description: "Please load graph data first",
        variant: "destructive"
      });
      return;
    }

    try {
      toast({
        title: "Running Semantic Analysis",
        description: "Analysing node content and semantic relationships..."
      });

      const taskId = await runAnalysis({
        type: 'semantic',
        graphData,
        options: {
          similarity_threshold: 0.7,
          topic_count: 10,
          embedding_model: 'default'
        }
      });

      setCurrentTaskId(taskId);

      toast({
        title: "Semantic Analysis Started",
        description: `Task ID: ${taskId}. Processing node content...`
      });
    } catch (error: any) {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to start semantic analysis",
        variant: "destructive"
      });
    }
  }, [graphData, runAnalysis]);

  const exportAnalysisResults = useCallback(() => {
    const results = {
      structural: structuralAnalysis,
      semantic: semanticAnalysis,
      clustering: clusteringResults,
      performance: performanceStats,
      timestamp: new Date().toISOString(),
      graphId
    };

    if (!results.structural && !results.semantic && !results.clustering) {
      toast({
        title: "No Results Available",
        description: "Please run an analysis first",
        variant: "destructive"
      });
      return;
    }

    
    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `graph-analysis-${graphId}-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast({
      title: "Analysis Results Exported",
      description: "Downloaded analysis report as JSON"
    });
  }, [structuralAnalysis, semanticAnalysis, clusteringResults, performanceStats, graphId]);

  return (
    <div className="space-y-4">
      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <GitCompare className="h-4 w-4" />
            Graph Comparison
            <Badge variant="secondary" className="text-xs">
              <AlertCircle className="h-3 w-3 mr-1" />
              Partial
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="comparison-toggle">Enable Comparison</Label>
            <Switch
              id="comparison-toggle"
              checked={comparisonEnabled}
              onCheckedChange={handleComparisonToggle}
            />
          </div>
          
          {comparisonEnabled && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <Select value={analysisType} onValueChange={(value) => setAnalysisType(value as 'structural' | 'semantic' | 'both')}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Comparison Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="structural">Structural Similarity</SelectItem>
                  <SelectItem value="semantic">Semantic Similarity</SelectItem>
                  <SelectItem value="both">Comprehensive Analysis</SelectItem>
                </SelectContent>
              </Select>
              
              <div className="flex items-center justify-between">
                <Label className="text-xs">Automatic Analysis</Label>
                <Switch
                  checked={autoAnalysis}
                  onCheckedChange={setAutoAnalysis}
                />
              </div>

              {(structuralAnalysis || semanticAnalysis) && (
                <div className="text-xs space-y-2 p-3 bg-muted rounded-md">
                  <div className="font-semibold text-primary">Analysis Results</div>
                  <div className="grid grid-cols-2 gap-2">
                    {structuralAnalysis && (
                      <>
                        <div className="flex justify-between">
                          <span>Clusters Found:</span>
                          <span className="font-mono text-blue-600">
                            {structuralAnalysis.clusters?.length || 0}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Modularity:</span>
                          <span className="font-mono">
                            {(structuralAnalysis.modularity || 0).toFixed(3)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Avg Centrality:</span>
                          <span className="font-mono">
                            {(structuralAnalysis.centrality?.average || 0).toFixed(3)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Connected Comps:</span>
                          <span className="font-mono text-purple-600">
                            {structuralAnalysis.connected_components || 0}
                          </span>
                        </div>
                      </>
                    )}
                    {semanticAnalysis && (
                      <>
                        <div className="flex justify-between">
                          <span>Topics Found:</span>
                          <span className="font-mono text-green-600">
                            {semanticAnalysis.topics?.length || 0}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Avg Similarity:</span>
                          <span className="font-mono">
                            {(semanticAnalysis.average_similarity || 0).toFixed(3)}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Advanced Analytics
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Real-time Metrics</Label>
            <Switch
              checked={metricsEnabled}
              onCheckedChange={setMetricsEnabled}
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={runStructuralAnalysis}
              disabled={isAnalyzing || !graphData}
              className="w-full"
            >
              <Network className="h-3 w-3 mr-1" />
              {isAnalyzing ? "Analysing..." : "Structural"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={runSemanticAnalysis}
              disabled={isAnalyzing || !graphData}
              className="w-full"
            >
              <Target className="h-3 w-3 mr-1" />
              {isAnalyzing ? "Processing..." : "Semantic"}
            </Button>
          </div>

          {metricsEnabled && (
            <div className="text-xs space-y-2 p-3 bg-muted rounded-md">
              <div className="font-semibold text-primary flex items-center gap-2">
                Network Metrics
                {isGPUEnabled && (
                  <Badge variant="secondary" className="text-xs">
                    <Zap className="h-2 w-2 mr-1" />
                    GPU
                  </Badge>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="flex justify-between">
                  <span>Nodes:</span>
                  <span className="font-mono">{graphData?.nodes?.length || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Edges:</span>
                  <span className="font-mono">{graphData?.edges?.length || 0}</span>
                </div>
                {performanceStats && (
                  <>
                    <div className="flex justify-between">
                      <span>Iterations:</span>
                      <span className="font-mono">{performanceStats.iteration_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Kinetic Energy:</span>
                      <span className="font-mono">{performanceStats.kinetic_energy.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total Forces:</span>
                      <span className="font-mono">{performanceStats.total_forces.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>GPU Failures:</span>
                      <span className="font-mono text-orange-600">{performanceStats.gpu_failure_count}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {}
      {graphData?.nodes && graphData?.edges && graphData.nodes.length > 0 && (
        <ShortestPathControls 
          nodes={graphData.nodes}
          edges={graphData.edges}
        />
      )}

      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Analysis Actions
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button 
            variant="outline" 
            size="sm" 
            className="w-full"
            onClick={exportAnalysisResults}
          >
            <TrendingUp className="h-3 w-3 mr-1" />
            Export Analysis Report
          </Button>
          
          {}
          {hasActiveTasks && (
            <div className="text-xs space-y-2 p-3 bg-blue-50 dark:bg-blue-950 rounded-md border border-blue-200 dark:border-blue-800">
              <div className="font-semibold text-blue-700 dark:text-blue-300 flex items-center gap-2">
                <Activity className="h-3 w-3 animate-pulse" />
                Active Tasks
              </div>
              {Array.from(activeTasks.entries()).map(([taskId, task]) => (
                <div key={taskId} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs capitalize">
                      {task.task_type}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {task.progress}%
                    </span>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-5 w-5 p-0 text-red-500 hover:text-red-700"
                    onClick={() => cancelTask(taskId)}
                  >
                    ×
                  </Button>
                </div>
              ))}
            </div>
          )}

          {}
          {error && (
            <div className="text-xs text-red-600 dark:text-red-400 p-2 bg-red-50 dark:bg-red-950 rounded border border-red-200 dark:border-red-800">
              <strong>Error:</strong> {error}
            </div>
          )}

          {}
          {performanceStats && (
            <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="h-3 w-3" />
                <span>GPU Acceleration: {isGPUEnabled ? 'Enabled' : 'Disabled'}</span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="h-3 w-3" />
                <span>Mode: {performanceStats.compute_mode}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default GraphAnalysisTab;