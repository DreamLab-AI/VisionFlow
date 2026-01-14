import React, { Suspense, useState, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { createXRStore, XR } from '@react-three/xr';
import GraphManager from '../../features/graph/components/GraphManager';
import { GraphData } from '../../features/graph/managers/graphDataManager';
import { VRAgentActionScene } from './VRAgentActionScene';

interface VRGraphCanvasProps {
  graphData: GraphData;
  onDragStateChange?: (isDragging: boolean) => void;
  /** Enable agent action visualization */
  enableAgentActions?: boolean;
  /** Show VR performance stats */
  showStats?: boolean;
}

// Create XR store outside component to persist across renders
const xrStore = createXRStore({
  hand: true,
  controller: true,
});

export function VRGraphCanvas({
  graphData,
  onDragStateChange,
  enableAgentActions = true,
  showStats = false,
}: VRGraphCanvasProps) {
  const [isVRSupported, setIsVRSupported] = useState<boolean | null>(null);

  // Extract agent nodes for VR targeting
  const agentNodes = useMemo(() => {
    if (!graphData?.nodes) return [];
    return graphData.nodes
      .filter(node => node.metadata?.type === 'agent')
      .map(node => ({
        id: node.id,
        type: (node.metadata?.agentType as string) || 'unknown',
        position: node.position,
        status: (node.metadata?.status as 'active' | 'idle' | 'error' | 'warning') || 'idle',
      }));
  }, [graphData?.nodes]);

  // Check VR support on mount
  React.useEffect(() => {
    if (navigator.xr) {
      navigator.xr.isSessionSupported('immersive-vr').then(setIsVRSupported);
    } else {
      setIsVRSupported(false);
    }
  }, []);

  return (
    <>
      {isVRSupported && (
        <button
          onClick={() => xrStore.enterVR()}
          style={{
            position: 'absolute',
            bottom: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '12px 24px',
            fontSize: '16px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            zIndex: 1000,
          }}
        >
          Enter VR
        </button>
      )}
      <Canvas
        gl={{ antialias: true, alpha: false }}
        camera={{ position: [0, 1.6, 3], fov: 70 }}
      >
        <XR store={xrStore}>
          <Suspense fallback={null}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <GraphManager onDragStateChange={onDragStateChange} />

            {/* Agent Action Visualization Layer */}
            {enableAgentActions && (
              <VRAgentActionScene
                agents={agentNodes}
                maxConnections={20}
                baseDuration={500}
                enableHandTracking={true}
                showStats={showStats}
                debug={false}
              />
            )}
          </Suspense>
        </XR>
      </Canvas>
    </>
  );
}

export default VRGraphCanvas;
