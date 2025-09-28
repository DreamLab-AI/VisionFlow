import React, { useEffect, useRef, useState } from 'react';
import { BabylonScene } from '../babylon/BabylonScene';
import { useImmersiveData } from '../hooks/useImmersiveData';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('ImmersiveApp');

export interface ImmersiveAppProps {
  onExit?: () => void;
  initialData?: any;
}

export const ImmersiveApp: React.FC<ImmersiveAppProps> = ({ onExit, initialData }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [babylonScene, setBabylonScene] = useState<BabylonScene | null>(null);

  // Use the immersive data hook
  const {
    graphData,
    nodePositions,
    isLoading: dataLoading,
    error: dataError,
    updateNodePosition,
    selectNode,
    selectedNode
  } = useImmersiveData(initialData);

  useEffect(() => {
    const initializeImmersiveEnvironment = () => {
      try {
        if (!canvasRef.current) {
          throw new Error('Canvas reference not available');
        }

        logger.info('Initializing immersive environment...');

        // Initialize Babylon.js scene - using the simpler API
        const scene = new BabylonScene(canvasRef.current);
        setBabylonScene(scene);

        // The scene constructor already creates XRManager, GraphRenderer, and XRUI
        // They are initialized internally in BabylonScene

        // Start the render loop
        scene.run();

        setIsInitialized(true);
        logger.info('Immersive environment initialized successfully');

      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown initialization error';
        logger.error('Failed to initialize immersive environment:', errorMessage);
        setError(errorMessage);
      }
    };

    initializeImmersiveEnvironment();

    // Cleanup function
    return () => {
      cleanup();
    };
  }, []);

  // Update graph data when it changes
  useEffect(() => {
    if (babylonScene && graphData && !dataLoading) {
      babylonScene.setBotsData({
        graphData: graphData, // Pass the full graph data object
        nodePositions: nodePositions,
        nodes: graphData.nodes || [],
        edges: graphData.edges || []
      });
    }
  }, [babylonScene, graphData, nodePositions, dataLoading]);

  const cleanup = () => {
    logger.info('Cleaning up immersive environment...');

    babylonScene?.dispose();

    setBabylonScene(null);
    setIsInitialized(false);
  };

  const handleRetry = () => {
    setError(null);
    setIsInitialized(false);
    // Re-trigger initialization
    if (canvasRef.current) {
      // Force re-render
      const canvas = canvasRef.current;
      canvas.style.display = 'none';
      setTimeout(() => {
        canvas.style.display = 'block';
      }, 100);
    }
  };

  if (error) {
    return (
      <div className="immersive-error">
        <h2>Immersive Environment Error</h2>
        <p>{error}</p>
        <div className="error-actions">
          <button onClick={handleRetry}>Retry</button>
          {onExit && <button onClick={onExit}>Exit</button>}
        </div>
      </div>
    );
  }

  if (dataError) {
    return (
      <div className="immersive-error">
        <h2>Data Loading Error</h2>
        <p>{dataError}</p>
        <div className="error-actions">
          <button onClick={() => window.location.reload()}>Reload</button>
          {onExit && <button onClick={onExit}>Exit</button>}
        </div>
      </div>
    );
  }

  return (
    <div className="immersive-app">
      <canvas
        ref={canvasRef}
        className="immersive-canvas"
        style={{
          width: '100vw',
          height: '100vh',
          display: 'block',
          position: 'fixed',
          top: 0,
          left: 0,
          zIndex: 1000
        }}
      />

      {!isInitialized && (
        <div className="immersive-loading">
          <div className="loading-content">
            <h2>Initializing Immersive Environment</h2>
            <p>Setting up Babylon.js, XR, and graph visualization...</p>
            <div className="loading-spinner" />
          </div>
        </div>
      )}

      {isInitialized && (
        <div className="immersive-overlay">
          <div className="immersive-controls">
            <button
              className="enter-xr-button"
              onClick={async (event) => {
                // This is a user interaction event, required for WebXR
                event.preventDefault();
                console.log('Enter AR button clicked - user interaction event active');

                // Enter WebXR AR/VR mode
                if (babylonScene && babylonScene.xrManager) {
                  try {
                    console.log('Requesting immersive session...');
                    await babylonScene.xrManager.enterXR();
                    console.log('Successfully entered immersive mode');
                  } catch (err) {
                    console.error('Failed to enter XR:', err);
                    alert('Failed to enter immersive mode. Please ensure WebXR permissions are granted.');
                  }
                } else {
                  console.error('XRManager not available');
                }
              }}
              title="Enter VR/AR Mode"
              style={{
                marginRight: '10px',
                padding: '10px 20px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Enter AR
            </button>
            <button
              className="exit-button"
              onClick={onExit}
              title="Exit Immersive Mode"
              style={{
                padding: '10px 20px',
                backgroundColor: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Exit
            </button>
          </div>

          {selectedNode && (
            <div className="immersive-info">
              <h3>Selected Node: {selectedNode}</h3>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImmersiveApp;