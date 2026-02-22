import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Slider } from '@/features/design-system/components/Slider';
import { Switch } from '@/features/design-system/components/Switch';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('SemanticClusteringControls');
import { Label } from '@/features/design-system/components/Label';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Progress } from '@/features/design-system/components/Progress';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { 
  Brain, 
  Sparkles, 
  Network, 
  AlertTriangle, 
  TrendingUp,
  BarChart3,
  GitBranch,
  Layers,
  RefreshCw,
  Download,
  Play,
  Pause
} from 'lucide-react';
import { useToast } from '@/features/design-system/components/Toast';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { unifiedApiClient } from '../../../services/api';

interface ClusteringMethod {
  id: string;
  name: string;
  description: string;
  params: Record<string, any>;
  gpuAccelerated: boolean;
}

interface Cluster {
  id: string;
  label: string;
  nodeCount: number;
  coherence: number;
  color: string;
  keywords: string[];
}

interface Anomaly {
  id: string;
  nodeId: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  score: number;
  description: string;
  timestamp: number;
}

export function SemanticClusteringControls() {
  const { toast } = useToast();
  
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [autoUpdate, setAutoUpdate] = useState(false);
  
  const [clusteringMethod, setClusteringMethod] = useState<string>('spectral');
  const [clusteringParams, setClusteringParams] = useState({
    numClusters: 8,
    minClusterSize: 5,
    similarity: 'cosine',
    convergenceThreshold: 0.001,
    maxIterations: 100,
  });
  
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);
  
  const [anomalyDetection, setAnomalyDetection] = useState({
    enabled: false,
    method: 'isolation_forest',
    sensitivity: 0.5,
    windowSize: 100,
    updateInterval: 5000,
  });
  
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [anomalyStats, setAnomalyStats] = useState({
    total: 0,
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  });

  const clusteringMethods: ClusteringMethod[] = [
    {
      id: 'spectral',
      name: 'Spectral Clustering',
      description: 'Graph-based clustering using eigendecomposition',
      params: ['similarity', 'numClusters'],
      gpuAccelerated: true,
    },
    {
      id: 'hierarchical',
      name: 'Hierarchical Clustering',
      description: 'Tree-based hierarchical decomposition',
      params: ['linkage', 'distanceThreshold'],
      gpuAccelerated: true,
    },
    {
      id: 'dbscan',
      name: 'DBSCAN',
      description: 'Density-based spatial clustering',
      params: ['eps', 'minSamples'],
      gpuAccelerated: true,
    },
    {
      id: 'kmeans',
      name: 'K-Means++',
      description: 'Centroid-based partitioning',
      params: ['numClusters', 'maxIterations'],
      gpuAccelerated: true,
    },
    {
      id: 'louvain',
      name: 'Louvain',
      description: 'Community detection via modularity optimization',
      params: ['resolution', 'randomState'],
      gpuAccelerated: true,
    },
    {
      id: 'affinity',
      name: 'Affinity Propagation',
      description: 'Message passing between data points',
      params: ['damping', 'preference'],
      gpuAccelerated: false,
    },
  ];

  const anomalyMethods = [
    { id: 'isolation_forest', name: 'Isolation Forest', description: 'Tree-based anomaly isolation' },
    { id: 'lof', name: 'Local Outlier Factor', description: 'Density-based local outliers' },
    { id: 'autoencoder', name: 'Autoencoder', description: 'Neural reconstruction error' },
    { id: 'statistical', name: 'Statistical', description: 'Z-score and IQR based' },
    { id: 'temporal', name: 'Temporal', description: 'Time-series anomalies' },
  ];

  
  const handleRunClustering = useCallback(async () => {
    setIsProcessing(true);
    setProgress(0);

    try {
      const response = await unifiedApiClient.post('/api/analytics/clustering/run', {
        method: clusteringMethod,
        params: clusteringParams,
      });

      setClusters(response.data.clusters);

      toast({
        title: 'Clustering Complete',
        description: `Found ${response.data.clusters.length} clusters`,
      });
    } catch (error) {
      toast({
        title: 'Clustering Failed',
        description: 'Unable to perform clustering analysis',
        variant: 'destructive',
      });
    } finally {
      setIsProcessing(false);
      setProgress(100);
    }
  }, [clusteringMethod, clusteringParams, toast]);

  const handleClusterSelection = useCallback(async (clusterId: string) => {
    setSelectedCluster(clusterId);

    try {
      await unifiedApiClient.post('/api/analytics/clustering/focus', { clusterId });
    } catch (error) {
      logger.error('Failed to focus cluster:', error);
    }
  }, []);

  const handleAnomalyToggle = useCallback(async (enabled: boolean) => {
    setAnomalyDetection(prev => ({ ...prev, enabled }));

    try {
      const { enabled: _existingEnabled, ...restAnomalyDetection } = anomalyDetection;
      await unifiedApiClient.post('/api/analytics/anomaly/toggle', { enabled, ...restAnomalyDetection });

      if (enabled) {
        toast({
          title: 'Anomaly Detection Enabled',
          description: 'Monitoring for unusual patterns',
        });
      }
    } catch (error) {
      logger.error('Failed to toggle anomaly detection:', error);
    }
  }, [anomalyDetection, toast]);

  
  useEffect(() => {
    if (!anomalyDetection.enabled) return;
    
    const fetchAnomalies = async () => {
      try {
        const response = await unifiedApiClient.get('/api/analytics/anomaly/current');
        setAnomalies(response.data.anomalies);
        setAnomalyStats(response.data.stats);
      } catch (error) {
        logger.error('Failed to fetch anomalies:', error);
      }
    };
    
    fetchAnomalies();
    const interval = setInterval(fetchAnomalies, anomalyDetection.updateInterval);
    return () => clearInterval(interval);
  }, [anomalyDetection.enabled, anomalyDetection.updateInterval]);

  
  useEffect(() => {
    if (isProcessing) {
      const timer = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);
      return () => clearInterval(timer);
    }
  }, [isProcessing]);

  return (
    <div className="space-y-4">
      {}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            Semantic Clustering
          </CardTitle>
          <CardDescription>
            GPU-accelerated graph clustering algorithms
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
                <Label>Clustering Method</Label>
                <Select value={clusteringMethod} onValueChange={setClusteringMethod}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {clusteringMethods.map(method => (
                      <SelectItem key={method.id} value={method.id}>
                        <div className="flex items-center gap-2">
                          <span>{method.name}</span>
                          {method.gpuAccelerated && (
                            <Badge variant="secondary" className="text-xs">GPU</Badge>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {clusteringMethods.find(m => m.id === clusteringMethod)?.description}
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Number of Clusters</Label>
                  <span className="text-sm text-muted-foreground">{clusteringParams.numClusters}</span>
                </div>
                <Slider
                  min={2}
                  max={20}
                  step={1}
                  value={[clusteringParams.numClusters]}
                  onValueChange={([v]) => setClusteringParams(prev => ({ ...prev, numClusters: v }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label>Similarity Metric</Label>
                <Select 
                  value={clusteringParams.similarity} 
                  onValueChange={(v) => setClusteringParams(prev => ({ ...prev, similarity: v }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cosine">Cosine Similarity</SelectItem>
                    <SelectItem value="euclidean">Euclidean Distance</SelectItem>
                    <SelectItem value="manhattan">Manhattan Distance</SelectItem>
                    <SelectItem value="jaccard">Jaccard Index</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-center justify-between">
                <Label>Auto-update Clusters</Label>
                <Switch
                  checked={autoUpdate}
                  onCheckedChange={setAutoUpdate}
                />
              </div>
              
              <Button 
                onClick={handleRunClustering} 
                disabled={isProcessing}
                className="w-full"
              >
                {isProcessing ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Run Clustering
                  </>
                )}
              </Button>
              
              {isProcessing && (
                <Progress value={progress} className="w-full" />
              )}
            </TabsContent>
            
            <TabsContent value="results" className="space-y-4">
              {clusters.length > 0 ? (
                <ScrollArea className="h-[300px]">
                  <div className="space-y-2">
                    {clusters.map(cluster => (
                      <div
                        key={cluster.id}
                        className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                          selectedCluster === cluster.id ? 'border-primary bg-primary/10' : 'hover:bg-muted'
                        }`}
                        onClick={() => handleClusterSelection(cluster.id)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div 
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: cluster.color }}
                            />
                            <span className="font-medium">{cluster.label}</span>
                          </div>
                          <Badge variant="outline">{cluster.nodeCount} nodes</Badge>
                        </div>
                        <div className="mt-2">
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span>Coherence: {(cluster.coherence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {cluster.keywords.slice(0, 3).map(keyword => (
                              <Badge key={keyword} variant="secondary" className="text-xs">
                                {keyword}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="flex flex-col items-center justify-center h-[200px] text-center">
                  <Network className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    No clusters found. Run clustering analysis to see results.
                  </p>
                </div>
              )}
              
              {clusters.length > 0 && (
                <Button variant="outline" className="w-full">
                  <Download className="mr-2 h-4 w-4" />
                  Export Clusters
                </Button>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            Anomaly Detection
          </CardTitle>
          <CardDescription>
            Real-time outlier and anomaly identification
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Enable Anomaly Detection</Label>
              <Switch
                checked={anomalyDetection.enabled}
                onCheckedChange={handleAnomalyToggle}
              />
            </div>
            
            {anomalyDetection.enabled && (
              <>
                <div className="space-y-2">
                  <Label>Detection Method</Label>
                  <Select 
                    value={anomalyDetection.method}
                    onValueChange={(v) => setAnomalyDetection(prev => ({ ...prev, method: v }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {anomalyMethods.map(method => (
                        <SelectItem key={method.id} value={method.id}>
                          <div className="flex flex-col">
                            <span>{method.name}</span>
                            <span className="text-xs text-muted-foreground">{method.description}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Sensitivity</Label>
                    <span className="text-sm text-muted-foreground">
                      {(anomalyDetection.sensitivity * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Slider
                    min={0}
                    max={1}
                    step={0.1}
                    value={[anomalyDetection.sensitivity]}
                    onValueChange={([v]) => setAnomalyDetection(prev => ({ ...prev, sensitivity: v }))}
                  />
                </div>
                
                {}
                <div className="p-4 rounded-lg bg-muted/50">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium">Detected Anomalies</span>
                    <Badge variant="outline">{anomalyStats.total}</Badge>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    <div className="text-center">
                      <div className="font-medium text-red-500">{anomalyStats.critical}</div>
                      <div className="text-muted-foreground">Critical</div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-orange-500">{anomalyStats.high}</div>
                      <div className="text-muted-foreground">High</div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-yellow-500">{anomalyStats.medium}</div>
                      <div className="text-muted-foreground">Medium</div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-blue-500">{anomalyStats.low}</div>
                      <div className="text-muted-foreground">Low</div>
                    </div>
                  </div>
                </div>
                
                {}
                {anomalies.length > 0 && (
                  <div>
                    <Label className="mb-2">Recent Anomalies</Label>
                    <ScrollArea className="h-[150px]">
                      <div className="space-y-2">
                        {anomalies.slice(0, 5).map(anomaly => (
                          <div key={anomaly.id} className="flex items-center gap-2 p-2 rounded border">
                            <Badge 
                              variant={
                                anomaly.severity === 'critical' ? 'destructive' :
                                anomaly.severity === 'high' ? 'destructive' :
                                anomaly.severity === 'medium' ? 'secondary' :
                                'outline'
                              }
                              className="text-xs"
                            >
                              {anomaly.severity}
                            </Badge>
                            <div className="flex-1 text-xs">
                              <div className="font-medium">Node {anomaly.nodeId}</div>
                              <div className="text-muted-foreground">{anomaly.description}</div>
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {new Date(anomaly.timestamp).toLocaleTimeString()}
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                )}
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Advanced Analytics
          </CardTitle>
          <CardDescription>
            GPU-powered graph analysis algorithms
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" size="sm">
              <Sparkles className="mr-2 h-4 w-4" />
              UMAP Reduction
            </Button>
            <Button variant="outline" size="sm">
              <GitBranch className="mr-2 h-4 w-4" />
              Graph Wavelets
            </Button>
            <Button variant="outline" size="sm">
              <Layers className="mr-2 h-4 w-4" />
              Hyperbolic Embed
            </Button>
            <Button variant="outline" size="sm">
              <BarChart3 className="mr-2 h-4 w-4" />
              Persistent Homology
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}