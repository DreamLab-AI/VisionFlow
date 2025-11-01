import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Input } from '@/features/design-system/components/Input';
import { Label } from '@/features/design-system/components/Label';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Progress } from '@/features/design-system/components/Progress';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { 
  Route, 
  Target,
  RefreshCw,
  Play,
  Trash2,
  AlertCircle,
  Clock,
  TrendingUp,
  Navigation
} from 'lucide-react';
import { useToast } from '@/features/design-system/components/Toast';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { 
  useAnalyticsStore, 
  useCurrentSSSPResult, 
  useSSSPLoading, 
  useSSSPError,
  useSSSPMetrics
} from '../store/analyticsStore';
import type { GraphNode, GraphEdge } from '../../graph/types/graphTypes';

interface ShortestPathControlsProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  className?: string;
}

export function ShortestPathControls({ nodes, edges, className }: ShortestPathControlsProps) {
  const { toast } = useToast();
  
  
  const [sourceNodeId, setSourceNodeId] = useState<string>('');
  const [algorithm, setAlgorithm] = useState<'dijkstra' | 'bellman-ford' | 'floyd-warshall'>('dijkstra');
  const [showNormalized, setShowNormalized] = useState(false);
  const [progress, setProgress] = useState(0);
  
  
  const computeSSSP = useAnalyticsStore(state => state.computeSSSP);
  const clearResults = useAnalyticsStore(state => state.clearResults);
  const normalizeDistances = useAnalyticsStore(state => state.normalizeDistances);
  const getUnreachableNodes = useAnalyticsStore(state => state.getUnreachableNodes);
  
  
  const currentResult = useCurrentSSSPResult();
  const loading = useSSSPLoading();
  const error = useSSSPError();
  const metrics = useSSSPMetrics();

  const algorithms = [
    {
      id: 'dijkstra',
      name: 'Dijkstra',
      description: 'Optimal for graphs with non-negative weights',
      complexity: 'O((V + E) log V)',
      bestFor: 'Dense graphs, single source'
    },
    {
      id: 'bellman-ford',
      name: 'Bellman-Ford',
      description: 'Handles negative weights, detects negative cycles',
      complexity: 'O(VE)',
      bestFor: 'Graphs with negative edges'
    },
    {
      id: 'floyd-warshall',
      name: 'Floyd-Warshall',
      description: 'All-pairs shortest paths',
      complexity: 'O(V³)',
      bestFor: 'Small dense graphs, all pairs'
    }
  ];

  
  useEffect(() => {
    if (nodes.length > 0 && !sourceNodeId) {
      setSourceNodeId(nodes[0].id);
    }
  }, [nodes, sourceNodeId]);

  
  useEffect(() => {
    if (loading) {
      const timer = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);
      return () => clearInterval(timer);
    } else {
      setProgress(0);
    }
  }, [loading]);

  const handleCalculate = useCallback(async () => {
    if (!sourceNodeId || nodes.length === 0) {
      toast({
        title: 'Invalid Input',
        description: 'Please select a source node',
        variant: 'destructive',
      });
      return;
    }

    setProgress(0);
    
    try {
      await computeSSSP(nodes, edges, sourceNodeId, algorithm);
      
      setProgress(100);
      
      toast({
        title: 'Calculation Complete',
        description: `Shortest paths computed using ${algorithm}`,
      });
    } catch (error) {
      toast({
        title: 'Calculation Failed',
        description: error instanceof Error ? error.message : 'Unknown error during computation',
        variant: 'destructive',
      });
    }
  }, [computeSSSP, nodes, edges, sourceNodeId, algorithm, toast]);

  const handleClear = useCallback(() => {
    clearResults();
    setShowNormalized(false);
    setProgress(0);
    
    toast({
      title: 'Results Cleared',
      description: 'Shortest path results have been cleared',
    });
  }, [clearResults, toast]);

  const formatDistance = (distance: number): string => {
    if (!isFinite(distance)) return '∞';
    return showNormalized ? distance.toFixed(3) : distance.toFixed(2);
  };

  const getDistancesToDisplay = (): Record<string, number> => {
    if (!currentResult) return {};
    return showNormalized ? normalizeDistances(currentResult) : currentResult.distances;
  };

  const unreachableNodes = currentResult ? getUnreachableNodes(currentResult) : [];
  const reachableCount = currentResult ? Object.keys(currentResult.distances).length - unreachableNodes.length : 0;

  return (
    <div className={`space-y-4 ${className}`}>
      {}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Route className="h-5 w-5" />
            Shortest Path Analysis
          </CardTitle>
          <CardDescription>
            Single-source shortest path computation with multiple algorithms
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="config">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="config">Configuration</TabsTrigger>
              <TabsTrigger value="results">Results</TabsTrigger>
            </TabsList>
            
            <TabsContent value="config" className="space-y-4">
              <div className="space-y-2">
                <Label>Source Node</Label>
                <Input
                  type="text"
                  placeholder="Enter source node ID"
                  value={sourceNodeId}
                  onChange={(e) => setSourceNodeId(e.target.value)}
                  disabled={loading}
                />
                <p className="text-xs text-muted-foreground">
                  {nodes.length > 0 ? `Available nodes: ${nodes.map(n => n.label || n.id).join(', ')}` : 'No nodes available'}
                </p>
              </div>
              
              <div className="space-y-2">
                <Label>Algorithm</Label>
                <Select value={algorithm} onValueChange={(value) => setAlgorithm(value as typeof algorithm)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {algorithms.map(alg => (
                      <SelectItem key={alg.id} value={alg.id}>
                        <div className="flex items-center gap-2">
                          <span>{alg.name}</span>
                          <Badge variant="secondary" className="text-xs">{alg.complexity}</Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {algorithms.find(a => a.id === algorithm)?.description}
                </p>
              </div>
              
              <div className="flex items-center justify-between">
                <Label>Show Normalized Distances (0-1)</Label>
                <input
                  type="checkbox"
                  checked={showNormalized}
                  onChange={(e) => setShowNormalized(e.target.checked)}
                  className="h-4 w-4"
                  disabled={!currentResult}
                />
              </div>
              
              <div className="flex gap-2">
                <Button 
                  onClick={handleCalculate} 
                  disabled={loading || !sourceNodeId || nodes.length === 0}
                  className="flex-1"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Calculating...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Calculate
                    </>
                  )}
                </Button>
                
                <Button 
                  variant="outline" 
                  onClick={handleClear}
                  disabled={loading || !currentResult}
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Clear
                </Button>
              </div>
              
              {loading && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Computing shortest paths...</span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="results" className="space-y-4">
              {error && (
                <div className="flex items-center gap-2 p-3 rounded-lg bg-red-50 border border-red-200">
                  <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
                  <div className="text-sm text-red-700">
                    <div className="font-medium">Error</div>
                    <div>{error}</div>
                  </div>
                </div>
              )}
              
              {currentResult ? (
                <div className="space-y-4">
                  {}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3 rounded-lg bg-blue-50 border border-blue-200">
                      <div className="flex items-center gap-2">
                        <Target className="h-4 w-4 text-blue-500" />
                        <span className="text-sm font-medium text-blue-700">Source</span>
                      </div>
                      <div className="text-lg font-bold text-blue-900 mt-1">
                        {nodes.find(n => n.id === currentResult.sourceNodeId)?.label || currentResult.sourceNodeId}
                      </div>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-green-50 border border-green-200">
                      <div className="flex items-center gap-2">
                        <Navigation className="h-4 w-4 text-green-500" />
                        <span className="text-sm font-medium text-green-700">Reachable</span>
                      </div>
                      <div className="text-lg font-bold text-green-900 mt-1">{reachableCount}</div>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-orange-50 border border-orange-200">
                      <div className="flex items-center gap-2">
                        <AlertCircle className="h-4 w-4 text-orange-500" />
                        <span className="text-sm font-medium text-orange-700">Unreachable</span>
                      </div>
                      <div className="text-lg font-bold text-orange-900 mt-1">{unreachableNodes.length}</div>
                    </div>
                    
                    <div className="p-3 rounded-lg bg-purple-50 border border-purple-200">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-purple-500" />
                        <span className="text-sm font-medium text-purple-700">Time</span>
                      </div>
                      <div className="text-lg font-bold text-purple-900 mt-1">
                        {currentResult.computationTime.toFixed(2)}ms
                      </div>
                    </div>
                  </div>

                  {}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <Label className="font-medium">Shortest Distances</Label>
                      <Badge variant="outline" className="text-xs">
                        {algorithm.toUpperCase()} - {showNormalized ? 'Normalized' : 'Actual'}
                      </Badge>
                    </div>
                    
                    <ScrollArea className="h-[300px]">
                      <div className="border rounded-lg">
                        <table className="w-full text-sm">
                          <thead className="bg-muted/50 sticky top-0">
                            <tr>
                              <th className="text-left p-3 border-b font-medium">Node</th>
                              <th className="text-right p-3 border-b font-medium">Distance</th>
                              <th className="text-left p-3 border-b font-medium">Via</th>
                              <th className="text-center p-3 border-b font-medium">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(getDistancesToDisplay()).map(([nodeId, distance]) => {
                              const node = nodes.find(n => n.id === nodeId);
                              const predecessor = currentResult.predecessors[nodeId];
                              const predecessorNode = predecessor ? nodes.find(n => n.id === predecessor) : null;
                              const isUnreachable = !isFinite(distance);
                              const isSource = nodeId === currentResult.sourceNodeId;
                              
                              return (
                                <tr 
                                  key={nodeId} 
                                  className={`border-b hover:bg-muted/50 ${isSource ? 'bg-blue-50' : ''}`}
                                >
                                  <td className="p-3 font-medium">
                                    {node?.label || nodeId}
                                    {isSource && <Badge variant="outline" className="ml-2 text-xs">Source</Badge>}
                                  </td>
                                  <td className="p-3 text-right font-mono">
                                    {formatDistance(distance)}
                                  </td>
                                  <td className="p-3 text-muted-foreground">
                                    {predecessor ? (predecessorNode?.label || predecessor) : isSource ? '-' : 'N/A'}
                                  </td>
                                  <td className="p-3 text-center">
                                    <Badge 
                                      variant={isUnreachable ? 'destructive' : isSource ? 'default' : 'secondary'}
                                      className="text-xs"
                                    >
                                      {isUnreachable ? 'Unreachable' : isSource ? 'Source' : 'Reachable'}
                                    </Badge>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </ScrollArea>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-[200px] text-center">
                  <Route className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    No results available. Configure and run shortest path analysis.
                  </p>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {}
      {metrics.totalComputations > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Performance Metrics
            </CardTitle>
            <CardDescription>
              Computation statistics and cache performance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="font-medium text-muted-foreground">Total Computations</div>
                <div className="text-lg font-bold mt-1">{metrics.totalComputations}</div>
              </div>
              <div className="p-3 rounded-lg bg-green-50 border border-green-200">
                <div className="font-medium text-green-700">Cache Hits</div>
                <div className="text-lg font-bold text-green-900 mt-1">{metrics.cacheHits}</div>
              </div>
              <div className="p-3 rounded-lg bg-orange-50 border border-orange-200">
                <div className="font-medium text-orange-700">Cache Misses</div>
                <div className="text-lg font-bold text-orange-900 mt-1">{metrics.cacheMisses}</div>
              </div>
              <div className="p-3 rounded-lg bg-blue-50 border border-blue-200">
                <div className="font-medium text-blue-700">Avg Time</div>
                <div className="text-lg font-bold text-blue-900 mt-1">
                  {metrics.averageComputationTime.toFixed(2)}ms
                </div>
              </div>
              <div className="p-3 rounded-lg bg-purple-50 border border-purple-200">
                <div className="font-medium text-purple-700">Hit Rate</div>
                <div className="text-lg font-bold text-purple-900 mt-1">
                  {metrics.totalComputations > 0 
                    ? ((metrics.cacheHits / metrics.totalComputations) * 100).toFixed(1)
                    : 0
                  }%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}