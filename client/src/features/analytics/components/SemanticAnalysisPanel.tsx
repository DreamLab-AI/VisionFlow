/**
 * Semantic Analysis Panel
 *
 * Provides advanced graph analytics via Phase 5 Semantic API:
 * - Community detection (Louvain, Label Propagation, Hierarchical, Connected Components)
 * - Centrality computation (PageRank, Betweenness, Closeness)
 * - Shortest path finding with path reconstruction
 * - Semantic constraint generation
 * - Performance statistics monitoring
 */

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Label } from '@/features/design-system/components/Label';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { Input } from '@/features/design-system/components/Input';
import { Slider } from '@/features/design-system/components/Slider';
import { Switch } from '@/features/design-system/components/Switch';
import { useToast } from '@/features/design-system/components/Toast';
import { Tooltip, TooltipProvider } from '@/features/design-system/components/Tooltip';
import {
  Network,
  TrendingUp,
  Route,
  Settings,
  Activity,
  AlertCircle,
  Trash2
} from 'lucide-react';
import {
  useSemanticService,
  CommunitiesResponse,
  CentralityResponse,
  ShortestPathResponse,
} from '../hooks/useSemanticService';

interface SemanticAnalysisPanelProps {
  className?: string;
}

export function SemanticAnalysisPanel({ className }: SemanticAnalysisPanelProps) {
  const { toast } = useToast();
  const {
    statistics,
    loading,
    error,
    detectCommunities,
    computeCentrality,
    computeShortestPath,
    generateConstraints,
    invalidateCache,
  } = useSemanticService();

  const [activeTab, setActiveTab] = useState('communities');

  // Community detection state
  const [communityAlgorithm, setCommunityAlgorithm] = useState<
    'louvain' | 'label_propagation' | 'connected_components' | 'hierarchical'
  >('louvain');
  const [minClusterSize, setMinClusterSize] = useState(5);
  const [communityResults, setCommunityResults] = useState<CommunitiesResponse | null>(null);

  // Centrality state
  const [centralityAlgorithm, setCentralityAlgorithm] = useState<'pagerank' | 'betweenness' | 'closeness'>(
    'pagerank'
  );
  const [topK, setTopK] = useState(10);
  const [centralityResults, setCentralityResults] = useState<CentralityResponse | null>(null);

  // Shortest path state
  const [sourceNode, setSourceNode] = useState('');
  const [targetNode, setTargetNode] = useState('');
  const [pathResults, setPathResults] = useState<ShortestPathResponse | null>(null);

  // Constraints state
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [enableClustering, setEnableClustering] = useState(true);
  const [enableImportance, setEnableImportance] = useState(true);
  const [enableTopic, setEnableTopic] = useState(false);
  const [maxConstraints, setMaxConstraints] = useState(1000);
  const [constraintsResult, setConstraintsResult] = useState<{ constraint_count: number; status: string } | null>(
    null
  );

  const handleDetectCommunities = async () => {
    try {
      const result = await detectCommunities({
        algorithm: communityAlgorithm,
        min_cluster_size: communityAlgorithm === 'hierarchical' ? minClusterSize : undefined,
      });
      setCommunityResults(result);
      toast({
        title: 'Communities Detected',
        description: `Found ${Object.keys(result.cluster_sizes).length} clusters (modularity: ${result.modularity.toFixed(4)})`,
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Detection Failed',
        description: err.message || 'Failed to detect communities',
        variant: 'destructive',
      });
    }
  };

  const handleComputeCentrality = async () => {
    try {
      const result = await computeCentrality({
        algorithm: centralityAlgorithm,
        top_k: topK,
      });
      setCentralityResults(result);
      toast({
        title: 'Centrality Computed',
        description: `Computed ${centralityAlgorithm} scores for ${Object.keys(result.scores).length} nodes`,
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Computation Failed',
        description: err.message || 'Failed to compute centrality',
        variant: 'destructive',
      });
    }
  };

  const handleComputeShortestPath = async () => {
    if (!sourceNode) {
      toast({
        title: 'Invalid Input',
        description: 'Please enter a source node ID',
        variant: 'destructive',
      });
      return;
    }

    try {
      const result = await computeShortestPath({
        source_node_id: parseInt(sourceNode),
        target_node_id: targetNode ? parseInt(targetNode) : undefined,
        include_path: true,
      });
      setPathResults(result);
      toast({
        title: 'Path Computed',
        description: `Computed distances to ${Object.keys(result.distances).length} nodes`,
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Computation Failed',
        description: err.message || 'Failed to compute shortest path',
        variant: 'destructive',
      });
    }
  };

  const handleGenerateConstraints = async () => {
    try {
      const result = await generateConstraints({
        similarity_threshold: similarityThreshold,
        enable_clustering: enableClustering,
        enable_importance: enableImportance,
        enable_topic: enableTopic,
        max_constraints: maxConstraints,
      });
      setConstraintsResult(result);
      toast({
        title: 'Constraints Generated',
        description: `Generated ${result.constraint_count} semantic constraints`,
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Generation Failed',
        description: err.message || 'Failed to generate constraints',
        variant: 'destructive',
      });
    }
  };

  const handleInvalidateCache = async () => {
    try {
      await invalidateCache();
      toast({
        title: 'Cache Cleared',
        description: 'Semantic analysis cache has been invalidated',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Clear Failed',
        description: err.message || 'Failed to invalidate cache',
        variant: 'destructive',
      });
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5" />
              Semantic Analysis
            </CardTitle>
            <CardDescription>Advanced graph analytics and community detection</CardDescription>
          </div>
          <TooltipProvider>
            <Tooltip content="Clear analysis cache">
              <Button variant="outline" size="sm" onClick={handleInvalidateCache} disabled={loading}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </Tooltip>
          </TooltipProvider>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="communities">
              <Network className="h-4 w-4 mr-1" />
              Communities
            </TabsTrigger>
            <TabsTrigger value="centrality">
              <TrendingUp className="h-4 w-4 mr-1" />
              Centrality
            </TabsTrigger>
            <TabsTrigger value="paths">
              <Route className="h-4 w-4 mr-1" />
              Paths
            </TabsTrigger>
            <TabsTrigger value="constraints">
              <Settings className="h-4 w-4 mr-1" />
              Constraints
            </TabsTrigger>
          </TabsList>

          {/* Communities Tab */}
          <TabsContent value="communities" className="space-y-4">
            <div className="space-y-3">
              <div>
                <Label>Algorithm</Label>
                <Select value={communityAlgorithm} onValueChange={(v: any) => setCommunityAlgorithm(v)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="louvain">Louvain</SelectItem>
                    <SelectItem value="label_propagation">Label Propagation</SelectItem>
                    <SelectItem value="connected_components">Connected Components</SelectItem>
                    <SelectItem value="hierarchical">Hierarchical Clustering</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {communityAlgorithm === 'hierarchical' && (
                <div>
                  <Label>Min Cluster Size</Label>
                  <Input
                    type="number"
                    value={minClusterSize}
                    onChange={(e) => setMinClusterSize(parseInt(e.target.value))}
                    min={1}
                  />
                </div>
              )}

              <Button onClick={handleDetectCommunities} disabled={loading} className="w-full">
                <Network className="mr-2 h-4 w-4" />
                Detect Communities
              </Button>
            </div>

            {communityResults && (
              <div className="rounded-lg border p-4 space-y-3">
                <h4 className="font-medium flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Results
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Clusters:</span>
                    <Badge>{Object.keys(communityResults.cluster_sizes).length}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Modularity:</span>
                    <span className="font-medium">{communityResults.modularity.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Computation Time:</span>
                    <span className="font-medium">{communityResults.computation_time_ms.toFixed(1)} ms</span>
                  </div>
                </div>

                <div className="max-h-48 overflow-y-auto space-y-1">
                  <Label className="text-sm">Cluster Sizes:</Label>
                  {Object.entries(communityResults.cluster_sizes)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .map(([clusterId, size]) => (
                      <div key={clusterId} className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Cluster {clusterId}:</span>
                        <span>{size as number} nodes</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </TabsContent>

          {/* Centrality Tab */}
          <TabsContent value="centrality" className="space-y-4">
            <div className="space-y-3">
              <div>
                <Label>Algorithm</Label>
                <Select value={centralityAlgorithm} onValueChange={(v: any) => setCentralityAlgorithm(v)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pagerank">PageRank</SelectItem>
                    <SelectItem value="betweenness">Betweenness</SelectItem>
                    <SelectItem value="closeness">Closeness</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Top K Nodes</Label>
                <Input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  min={1}
                  max={100}
                />
              </div>

              <Button onClick={handleComputeCentrality} disabled={loading} className="w-full">
                <TrendingUp className="mr-2 h-4 w-4" />
                Compute Centrality
              </Button>
            </div>

            {centralityResults && (
              <div className="rounded-lg border p-4 space-y-3">
                <h4 className="font-medium flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Top {centralityResults.top_nodes.length} Nodes
                </h4>
                <div className="max-h-64 overflow-y-auto space-y-1">
                  {centralityResults.top_nodes.map(([nodeId, score], idx) => (
                    <div key={nodeId} className="flex justify-between text-sm">
                      <span className="text-muted-foreground">
                        #{idx + 1} Node {nodeId}:
                      </span>
                      <span className="font-medium">{score.toFixed(6)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>

          {/* Shortest Paths Tab */}
          <TabsContent value="paths" className="space-y-4">
            <div className="space-y-3">
              <div>
                <Label>Source Node ID</Label>
                <Input
                  type="number"
                  placeholder="Enter source node ID"
                  value={sourceNode}
                  onChange={(e) => setSourceNode(e.target.value)}
                />
              </div>

              <div>
                <Label>Target Node ID (optional)</Label>
                <Input
                  type="number"
                  placeholder="Leave empty for all paths"
                  value={targetNode}
                  onChange={(e) => setTargetNode(e.target.value)}
                />
              </div>

              <Button onClick={handleComputeShortestPath} disabled={loading || !sourceNode} className="w-full">
                <Route className="mr-2 h-4 w-4" />
                Find Shortest Paths
              </Button>
            </div>

            {pathResults && (
              <div className="rounded-lg border p-4 space-y-3">
                <h4 className="font-medium flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Results
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Source Node:</span>
                    <Badge>{pathResults.source_node}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Reachable Nodes:</span>
                    <span className="font-medium">{Object.keys(pathResults.distances).length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Computation Time:</span>
                    <span className="font-medium">{pathResults.computation_time_ms.toFixed(1)} ms</span>
                  </div>
                </div>

                {Object.keys(pathResults.paths).length > 0 && (
                  <div className="max-h-48 overflow-y-auto space-y-1">
                    <Label className="text-sm">Sample Paths:</Label>
                    {Object.entries(pathResults.paths)
                      .slice(0, 10)
                      .map(([nodeId, path]) => (
                        <div key={nodeId} className="text-sm">
                          <span className="text-muted-foreground">To node {nodeId}:</span>
                          <div className="font-mono text-xs ml-2">{path.join(' → ')}</div>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            )}
          </TabsContent>

          {/* Constraints Tab */}
          <TabsContent value="constraints" className="space-y-4">
            <div className="space-y-3">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Similarity Threshold</Label>
                  <span className="text-sm text-muted-foreground">{similarityThreshold.toFixed(2)}</span>
                </div>
                <Slider
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={[similarityThreshold]}
                  onValueChange={([v]) => setSimilarityThreshold(v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Enable Clustering Constraints</Label>
                  <Switch checked={enableClustering} onCheckedChange={setEnableClustering} />
                </div>
                <div className="flex items-center justify-between">
                  <Label>Enable Importance Constraints</Label>
                  <Switch checked={enableImportance} onCheckedChange={setEnableImportance} />
                </div>
                <div className="flex items-center justify-between">
                  <Label>Enable Topic Constraints</Label>
                  <Switch checked={enableTopic} onCheckedChange={setEnableTopic} />
                </div>
              </div>

              <div>
                <Label>Max Constraints</Label>
                <Input
                  type="number"
                  value={maxConstraints}
                  onChange={(e) => setMaxConstraints(parseInt(e.target.value))}
                  min={10}
                  max={10000}
                />
              </div>

              <Button onClick={handleGenerateConstraints} disabled={loading} className="w-full">
                <Settings className="mr-2 h-4 w-4" />
                Generate Constraints
              </Button>
            </div>

            {constraintsResult && (
              <div className="rounded-lg border p-4 space-y-2">
                <h4 className="font-medium flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Generation Complete
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Constraints Generated:</span>
                    <Badge>{constraintsResult.constraint_count}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Status:</span>
                    <span className="font-medium">{constraintsResult.status}</span>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Statistics Footer */}
        {statistics && (
          <div className="border-t pt-4">
            <Label className="text-sm font-medium flex items-center gap-2 mb-3">
              <Activity className="h-4 w-4" />
              Performance Statistics
            </Label>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Analyses:</span>
                <span className="font-medium">{statistics.total_analyses}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Cache Hit Rate:</span>
                <span className="font-medium">{(statistics.cache_hit_rate * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg Clustering:</span>
                <span className="font-medium">{statistics.average_clustering_time_ms.toFixed(1)} ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg Pathfinding:</span>
                <span className="font-medium">{statistics.average_pathfinding_time_ms.toFixed(1)} ms</span>
              </div>
            </div>
          </div>
        )}

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
