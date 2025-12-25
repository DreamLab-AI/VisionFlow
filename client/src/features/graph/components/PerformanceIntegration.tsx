

import { useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { graphPerformanceMonitor } from "../../../utils/graphPerformanceMonitor';
import { graphOptimizer } from "../../../utils/graphOptimizations';
import { debugState } from '../../../utils/clientDebugState';

interface PerformanceIntegrationProps {
  logseqNodeCount: number;
  logseqEdgeCount: number;
  visionflowNodeCount: number;
  visionflowEdgeCount: number;
  onPerformanceUpdate?: (metrics: any) => void;
}

export const PerformanceIntegration: React.FC<PerformanceIntegrationProps> = ({
  logseqNodeCount,
  logseqEdgeCount,
  visionflowNodeCount,
  visionflowEdgeCount,
  onPerformanceUpdate
}) => {
  const { camera, gl } = useThree();

  
  useEffect(() => {
    if (debugState.isEnabled()) {
      graphPerformanceMonitor.initializeWebGL(gl);
      graphOptimizer.initializeOptimizations(camera, gl);
    }
  }, [camera, gl]);

  
  useEffect(() => {
    graphPerformanceMonitor.mark('logseq-update');
    graphPerformanceMonitor.updateGraphMetrics('logseq', {
      nodeCount: logseqNodeCount,
      edgeCount: logseqEdgeCount,
      updateTime: graphPerformanceMonitor.measure('logseq-update'),
      instancedRendering: true
    });
  }, [logseqNodeCount, logseqEdgeCount]);

  useEffect(() => {
    graphPerformanceMonitor.mark('visionflow-update');
    graphPerformanceMonitor.updateGraphMetrics('visionflow', {
      nodeCount: visionflowNodeCount,
      edgeCount: visionflowEdgeCount,
      updateTime: graphPerformanceMonitor.measure('visionflow-update'),
      instancedRendering: visionflowNodeCount > 50
    });
  }, [visionflowNodeCount, visionflowEdgeCount]);

  
  useFrame((state, delta) => {
    if (!debugState.isEnabled()) return;

    
    graphPerformanceMonitor.beginFrame();
    
    
    graphOptimizer.optimizeFrame(camera);
    
    
    graphPerformanceMonitor.endFrame(gl);
    
    
    if (onPerformanceUpdate) {
      const metrics = graphPerformanceMonitor.getMetrics();
      onPerformanceUpdate(metrics);
    }
  });

  return null; 
};