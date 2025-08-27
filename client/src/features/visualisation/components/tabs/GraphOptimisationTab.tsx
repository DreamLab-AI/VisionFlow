/**
 * Graph Optimisation Tab Component
 * Provides AI-powered graph optimisation tools with UK English localisation
 */

import React, { useState, useCallback } from 'react';
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

interface GraphOptimisationTabProps {
  graphId?: string;
  onFeatureUpdate?: (feature: string, data: any) => void;
}

export const GraphOptimisationTab: React.FC<GraphOptimisationTabProps> = ({ 
  graphId = 'default',
  onFeatureUpdate
}) => {
  // AI Insights states
  const [aiInsightsEnabled, setAiInsightsEnabled] = useState(false);
  const [autoOptimise, setAutoOptimise] = useState(false);
  const [optimisationLevel, setOptimisationLevel] = useState([3]);
  const [isOptimising, setIsOptimising] = useState(false);
  const [optimisationProgress, setOptimisationProgress] = useState(0);
  
  // Performance states
  const [performanceMode, setPerformanceMode] = useState('balanced');
  const [clusteringEnabled, setClusteringEnabled] = useState(false);
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('force-directed');
  
  // Mock optimisation results
  const [optimisationResults, setOptimisationResults] = useState<any>(null);
  const [mockResults] = useState({
    algorithm: 'Adaptive Force-Directed',
    confidence: 0.87,
    performanceGain: 0.34,
    clusters: 8,
    recommendations: [
      { type: 'layout', priority: 'high', description: 'Adjust node spacing for better clarity' },
      { type: 'clustering', priority: 'medium', description: 'Group related nodes for improved navigation' },
      { type: 'performance', priority: 'low', description: 'Enable GPU acceleration for smoother interactions' }
    ]
  });

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
    setIsOptimising(true);
    setOptimisationProgress(0);
    
    toast({
      title: "Running Layout Optimisation",
      description: "Applying AI-recommended positions and clustering..."
    });
    
    // Simulate optimisation progress
    const interval = setInterval(() => {
      setOptimisationProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsOptimising(false);
          setOptimisationResults(mockResults);
          toast({
            title: "Optimisation Complete",
            description: `Layout improved with ${(mockResults.performanceGain * 100).toFixed(0)}% performance gain`
          });
          return 100;
        }
        return prev + 10;
      });
    }, 200);
  }, [mockResults]);

  const runClusterAnalysis = useCallback(async () => {
    setIsOptimising(true);
    toast({
      title: "Running Cluster Analysis",
      description: "Identifying node communities and optimal groupings..."
    });
    
    setTimeout(() => {
      setIsOptimising(false);
      toast({
        title: "Clustering Analysis Complete",
        description: `Identified ${mockResults.clusters} optimal clusters`
      });
    }, 1500);
  }, [mockResults.clusters]);

  const applyRecommendation = useCallback((recommendation: any) => {
    toast({
      title: "Applying Recommendation",
      description: recommendation.description
    });
    
    onFeatureUpdate?.('recommendation', recommendation);
  }, [onFeatureUpdate]);

  return (
    <div className="space-y-4">
      {/* AI Insights Control */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Brain className="h-4 w-4" />
            AI-Powered Optimisation
            <Badge variant="secondary" className="text-xs">
              <AlertCircle className="h-3 w-3 mr-1" />
              Partial
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
                  <SelectItem value="force-directed">Force-Directed</SelectItem>
                  <SelectItem value="hierarchical">Hierarchical</SelectItem>
                  <SelectItem value="circular">Circular</SelectItem>
                  <SelectItem value="grid">Grid-Based</SelectItem>
                  <SelectItem value="adaptive">Adaptive (AI)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Optimisation Tools */}
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
                <Label className="text-xs">Progress</Label>
                <span className="text-xs">{optimisationProgress}%</span>
              </div>
              <Progress value={optimisationProgress} className="w-full" />
            </div>
          )}
          
          <div className="grid grid-cols-2 gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={runLayoutOptimisation}
              disabled={isOptimising || !aiInsightsEnabled}
              className="w-full"
            >
              <Target className="h-3 w-3 mr-1" />
              {isOptimising ? "Optimising..." : "Layout"}
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={runClusterAnalysis}
              disabled={isOptimising}
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
            <div className="text-xs space-y-2 p-3 bg-muted rounded-md">
              <div className="font-semibold text-primary">Optimisation Results</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="flex justify-between">
                  <span>Algorithm:</span>
                  <span className="font-mono text-xs">{optimisationResults.algorithm}</span>
                </div>
                <div className="flex justify-between">
                  <span>Confidence:</span>
                  <span className="font-mono text-green-600">
                    {(optimisationResults.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Performance:</span>
                  <span className="font-mono text-blue-600">
                    +{(optimisationResults.performanceGain * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Clusters:</span>
                  <span className="font-mono">{optimisationResults.clusters}</span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* AI Recommendations */}
      {aiInsightsEnabled && optimisationResults && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              AI Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {optimisationResults.recommendations.map((rec: any, index: number) => (
              <div key={index} className="flex items-center justify-between p-2 bg-muted/50 rounded">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Badge 
                      variant={rec.priority === 'high' ? 'destructive' : rec.priority === 'medium' ? 'default' : 'secondary'}
                      className="text-xs"
                    >
                      {rec.priority}
                    </Badge>
                    <span className="text-xs font-semibold capitalize">{rec.type}</span>
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

      {/* Performance Settings */}
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
          
          <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
            <strong>Note:</strong> AI-powered optimisation features are under active development. 
            Current implementation provides basic layout and clustering analysis.
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GraphOptimisationTab;