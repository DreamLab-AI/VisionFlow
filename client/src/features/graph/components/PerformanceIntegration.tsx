

import { useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { dualGraphPerformanceMonitor } from '../../../utils/dualGraphPerformanceMonitor';
import { dualGraphOptimizer } from '../../../utils/dualGraphOptimizations';
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
      dualGraphPerformanceMonitor.initializeWebGL(gl);
      dualGraphOptimizer.initializeOptimizations(camera, gl);
    }
  }, [camera, gl]);

  
  useEffect(() => {
    dualGraphPerformanceMonitor.mark('logseq-update');
    dualGraphPerformanceMonitor.updateGraphMetrics('logseq', {
      nodeCount: logseqNodeCount,
      edgeCount: logseqEdgeCount,
      updateTime: dualGraphPerformanceMonitor.measure('logseq-update'),
      instancedRendering: true
    });
  }, [logseqNodeCount, logseqEdgeCount]);

  useEffect(() => {
    dualGraphPerformanceMonitor.mark('visionflow-update');
    dualGraphPerformanceMonitor.updateGraphMetrics('visionflow', {
      nodeCount: visionflowNodeCount,
      edgeCount: visionflowEdgeCount,
      updateTime: dualGraphPerformanceMonitor.measure('visionflow-update'),
      instancedRendering: visionflowNodeCount > 50
    });
  }, [visionflowNodeCount, visionflowEdgeCount]);

  
  useFrame((state, delta) => {
    if (!debugState.isEnabled()) return;

    
    dualGraphPerformanceMonitor.beginFrame();
    
    
    dualGraphOptimizer.optimizeFrame(camera);
    
    
    dualGraphPerformanceMonitor.endFrame(gl);
    
    
    if (onPerformanceUpdate) {
      const metrics = dualGraphPerformanceMonitor.getMetrics();
      onPerformanceUpdate(metrics);
    }
  });

  return null; 
};