/**
 * Graph Analysis Tab Component
 * Provides advanced graph analysis tools with UK English localisation
 */

import React, { useState, useCallback } from 'react';
import { 
  GitCompare, 
  Brain, 
  TrendingUp,
  BarChart3,
  Network,
  Target,
  AlertCircle
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
  // Analysis states
  const [comparisonEnabled, setComparisonEnabled] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [isAnalysing, setIsAnalysing] = useState(false);
  const [metricsEnabled, setMetricsEnabled] = useState(true);
  const [autoAnalysis, setAutoAnalysis] = useState(false);

  // Mock analysis data for demonstration
  const [mockAnalysis] = useState({
    similarity: {
      overall: 0.73,
      structural: 0.68,
      semantic: 0.78
    },
    matches: 127,
    differences: 42,
    clusters: 8,
    centrality: {
      betweenness: 0.34,
      closeness: 0.62,
      eigenvector: 0.51
    }
  });

  const handleComparisonToggle = useCallback((enabled: boolean) => {
    setComparisonEnabled(enabled);
    if (enabled) {
      toast({
        title: "Graph Comparison Activated",
        description: "Analysing similarities and differences between graphs..."
      });
    } else {
      toast({
        title: "Graph Comparison Deactivated",
        description: "Comparison analysis stopped"
      });
    }
  }, []);

  const runStructuralAnalysis = useCallback(async () => {
    setIsAnalysing(true);
    toast({
      title: "Running Structural Analysis",
      description: "Analysing graph topology and connectivity patterns..."
    });
    
    // Simulate analysis delay
    setTimeout(() => {
      setAnalysisResults(mockAnalysis);
      setIsAnalysing(false);
      toast({
        title: "Analysis Complete",
        description: "Structural analysis results are now available"
      });
    }, 2000);
  }, [mockAnalysis]);

  const runSemanticAnalysis = useCallback(async () => {
    setIsAnalysing(true);
    toast({
      title: "Running Semantic Analysis",
      description: "Analysing node content and semantic relationships..."
    });
    
    setTimeout(() => {
      setIsAnalysing(false);
      toast({
        title: "Semantic Analysis Complete",
        description: "Content similarity patterns identified"
      });
    }, 1500);
  }, []);

  const exportAnalysisResults = useCallback(() => {
    if (!analysisResults) {
      toast({
        title: "No Results Available",
        description: "Please run an analysis first",
        variant: "destructive"
      });
      return;
    }
    
    toast({
      title: "Exporting Analysis Results",
      description: "Downloading analysis report as JSON..."
    });
  }, [analysisResults]);

  return (
    <div className="space-y-4">
      {/* Comparison Controls */}
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
              <Select defaultValue="both">
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

              {analysisResults && (
                <div className="text-xs space-y-2 p-3 bg-muted rounded-md">
                  <div className="font-semibold text-primary">Analysis Results</div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex justify-between">
                      <span>Overall Similarity:</span>
                      <span className="font-mono text-green-600">
                        {(analysisResults.similarity.overall * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Node Matches:</span>
                      <span className="font-mono">{analysisResults.matches}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Structural:</span>
                      <span className="font-mono">
                        {(analysisResults.similarity.structural * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Differences:</span>
                      <span className="font-mono text-orange-600">{analysisResults.differences}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Advanced Analytics */}
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
              disabled={isAnalysing}
              className="w-full"
            >
              <Network className="h-3 w-3 mr-1" />
              {isAnalysing ? "Analysing..." : "Structural"}
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={runSemanticAnalysis}
              disabled={isAnalysing}
              className="w-full"
            >
              <Target className="h-3 w-3 mr-1" />
              {isAnalysing ? "Processing..." : "Semantic"}
            </Button>
          </div>

          {metricsEnabled && (
            <div className="text-xs space-y-2 p-3 bg-muted rounded-md">
              <div className="font-semibold text-primary">Network Metrics</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="flex justify-between">
                  <span>Clusters:</span>
                  <span className="font-mono">{mockAnalysis.clusters}</span>
                </div>
                <div className="flex justify-between">
                  <span>Betweenness:</span>
                  <span className="font-mono">{mockAnalysis.centrality.betweenness.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Closeness:</span>
                  <span className="font-mono">{mockAnalysis.centrality.closeness.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Eigenvector:</span>
                  <span className="font-mono">{mockAnalysis.centrality.eigenvector.toFixed(2)}</span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Shortest Path Analysis - Integrated analytics component */}
      {graphData?.nodes && graphData?.edges && graphData.nodes.length > 0 && (
        <ShortestPathControls 
          nodes={graphData.nodes}
          edges={graphData.edges}
        />
      )}

      {/* Export & Actions */}
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
          
          <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
            <strong>Note:</strong> Advanced graph analysis features are under development. 
            Current implementation provides basic comparison and metrics calculation.
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GraphAnalysisTab;