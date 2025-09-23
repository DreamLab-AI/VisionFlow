// Analytics Tab - Clustering, anomaly detection & ML features
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { Badge } from '@/features/design-system/components/Badge';
import { BarChart3, Brain, AlertTriangle, TrendingUp, Route, Network, Activity } from 'lucide-react';
import { useToast } from '@/features/design-system/components/Toast';
import { SemanticClusteringControls } from '@/features/analytics/components/SemanticClusteringControls';
import { ShortestPathControls } from '@/features/analytics/components/ShortestPathControls';
import { SSSPAnalysisPanel } from '@/features/analytics/components/SSSPAnalysisPanel';
import { useGraphStore } from '@/store/graphStore';
import type { GraphNode, GraphEdge } from '@/features/graph/types/graphTypes';

interface AnalyticsTabProps {
  searchQuery?: string;
}

export const AnalyticsTab: React.FC<AnalyticsTabProps> = ({ searchQuery = '' }) => {
  const { toast } = useToast();
  const { nodes, edges, isConnected } = useGraphStore();
  const [activeTab, setActiveTab] = useState('clustering');
  
  // Use actual graph data from store
  const graphNodes = nodes;
  const graphEdges = edges;
  
  useEffect(() => {
    if (graphNodes.length === 0) {
      toast({
        title: 'No Graph Data',
        description: 'Using demo data. Connect to a graph source for live analysis.',
        variant: 'default',
      });
    }
  }, [graphNodes.length, toast]);
  
  // Filter tabs based on search query
  const shouldShowTab = (tabContent: string) => {
    if (!searchQuery) return true;
    return tabContent.toLowerCase().includes(searchQuery.toLowerCase());
  };
  
  const availableTabs = [
    {
      id: 'clustering',
      label: 'Clustering',
      icon: Network,
      searchContent: 'clustering semantic machine learning gpu accelerated spectral hierarchical dbscan kmeans louvain anomaly detection',
      show: shouldShowTab('clustering semantic machine learning')
    },
    {
      id: 'paths',
      label: 'Shortest Paths',
      icon: Route,
      searchContent: 'shortest path dijkstra bellman ford floyd warshall graph algorithms',
      show: shouldShowTab('shortest path dijkstra algorithm')
    },
    {
      id: 'analysis',
      label: 'Graph Analysis',
      icon: BarChart3,
      searchContent: 'analysis sssp centrality metrics topology',
      show: shouldShowTab('analysis metrics topology')
    }
  ];
  
  const visibleTabs = availableTabs.filter(tab => tab.show);
  
  // Set first visible tab as active if current tab is not visible
  useEffect(() => {
    if (visibleTabs.length > 0 && !visibleTabs.some(tab => tab.id === activeTab)) {
      setActiveTab(visibleTabs[0].id);
    }
  }, [visibleTabs, activeTab]);
  
  if (visibleTabs.length === 0) {
    return (
      <div className="text-center text-muted-foreground py-8">
        No analytics features match your search.
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* Analytics Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Analytics & ML Features
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={isConnected ? 'default' : 'secondary'}>
                {isConnected ? 'Live Data' : 'Demo Mode'}
              </Badge>
              <Badge variant="outline">
                {graphNodes.length} nodes, {graphEdges.length} edges
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 border rounded-lg">
              <Network className="w-8 h-8 text-blue-500 mb-2" />
              <h4 className="font-semibold">GPU Clustering</h4>
              <p className="text-sm text-muted-foreground">Spectral, DBSCAN, K-means, Louvain algorithms</p>
            </div>
            <div className="p-4 border rounded-lg">
              <AlertTriangle className="w-8 h-8 text-red-500 mb-2" />
              <h4 className="font-semibold">Anomaly Detection</h4>
              <p className="text-sm text-muted-foreground">Real-time outlier identification</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Route className="w-8 h-8 text-green-500 mb-2" />
              <h4 className="font-semibold">Path Analysis</h4>
              <p className="text-sm text-muted-foreground">Dijkstra, Bellman-Ford, Floyd-Warshall</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Brain className="w-8 h-8 text-purple-500 mb-2" />
              <h4 className="font-semibold">ML Insights</h4>
              <p className="text-sm text-muted-foreground">UMAP, t-SNE, topological analysis</p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Analytics Controls */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          {visibleTabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <TabsTrigger key={tab.id} value={tab.id} className="flex items-center gap-2">
                <Icon className="w-4 h-4" />
                {tab.label}
              </TabsTrigger>
            );
          })}
        </TabsList>
        
        <TabsContent value="clustering" className="mt-6">
          <SemanticClusteringControls />
        </TabsContent>
        
        <TabsContent value="paths" className="mt-6">
          <ShortestPathControls 
            nodes={graphNodes} 
            edges={graphEdges}
            className="space-y-4"
          />
        </TabsContent>
        
        <TabsContent value="analysis" className="mt-6">
          <SSSPAnalysisPanel 
            nodes={graphNodes}
            edges={graphEdges}
            className="space-y-4"
          />
        </TabsContent>
      </Tabs>
      
      {/* Performance Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Analytics Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">GPU</div>
              <div className="text-sm text-muted-foreground">Accelerated</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{graphNodes.length}</div>
              <div className="text-sm text-muted-foreground">Nodes Analyzed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{graphEdges.length}</div>
              <div className="text-sm text-muted-foreground">Edges Processed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{isConnected ? 'Live' : 'Demo'}</div>
              <div className="text-sm text-muted-foreground">Data Mode</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AnalyticsTab;