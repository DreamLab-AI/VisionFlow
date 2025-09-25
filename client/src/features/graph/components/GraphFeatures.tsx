import React, { useState, useEffect, useCallback } from 'react';
import styles from './GraphFeatures.module.css';
import { innovationManager } from '../innovations';
import {
  graphSynchronization,
  graphComparison,
  graphAnimations,
  aiInsights,
  advancedInteractionModes
} from '../innovations';

interface GraphFeaturesProps {
  onFeatureToggle?: (featureId: string, enabled: boolean) => void;
  className?: string;
}

const GraphFeatures: React.FC<GraphFeaturesProps> = ({
  onFeatureToggle,
  className = ''
}) => {
  // Feature states
  const [syncEnabled, setSyncEnabled] = useState(false);
  const [comparisonEnabled, setComparisonEnabled] = useState(false);
  const [animationsEnabled, setAnimationsEnabled] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(false);
  const [interactionsEnabled, setInteractionsEnabled] = useState(false);

  // Sync specific states
  const [syncedGraphs, setSyncedGraphs] = useState<string[]>([]);
  const [syncMode, setSyncMode] = useState<'full' | 'partial'>('full');

  // Comparison results
  const [comparisonResults, setComparisonResults] = useState<any>(null);
  const [similarities, setSimilarities] = useState<number>(0);

  // Animation states
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [morphingActive, setMorphingActive] = useState(false);

  // AI recommendations
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [anomalies, setAnomalies] = useState<any[]>([]);

  // Advanced interaction modes
  const [timeTravelActive, setTimeTravelActive] = useState(false);
  const [explorationMode, setExplorationMode] = useState(false);
  const [vrMode, setVrMode] = useState(false);

  // Initialize innovation manager
  useEffect(() => {
    const initializeFeatures = async () => {
      const status = innovationManager.getStatus();
      if (!status.isInitialized) {
        await innovationManager.initialize({
          enableSync: true,
          enableComparison: true,
          enableAnimations: true,
          enableAI: true,
          enableAdvancedInteractions: true,
          performanceMode: 'balanced'
        });
      }

      // Update states based on active features
      const newStatus = innovationManager.getStatus();
      setSyncEnabled(newStatus.capabilities.synchronization);
      setComparisonEnabled(newStatus.capabilities.comparison);
      setAnimationsEnabled(newStatus.capabilities.animations);
      setAiEnabled(newStatus.capabilities.aiInsights);
      setInteractionsEnabled(newStatus.capabilities.advancedInteractions);
    };

    initializeFeatures();
  }, []);

  // Handle synchronization toggle
  const handleSyncToggle = useCallback(async () => {
    const newState = !syncEnabled;
    setSyncEnabled(newState);

    if (newState) {
      // Start synchronization
      graphSynchronization.startSync({
        mode: syncMode,
        preserveLayout: true,
        syncInterval: 100
      });
      setSyncedGraphs(['graph1', 'graph2']);
    } else {
      // Stop synchronization
      graphSynchronization.stopSync();
      setSyncedGraphs([]);
    }

    onFeatureToggle?.('synchronization', newState);
  }, [syncEnabled, syncMode, onFeatureToggle]);

  // Handle comparison toggle
  const handleComparisonToggle = useCallback(async () => {
    const newState = !comparisonEnabled;
    setComparisonEnabled(newState);

    if (newState) {
      // Perform comparison
      const results = await graphComparison.performComprehensiveComparison('graph1', 'graph2');
      setComparisonResults(results);
      setSimilarities(results?.overallSimilarity || 0);
    } else {
      setComparisonResults(null);
      setSimilarities(0);
    }

    onFeatureToggle?.('comparison', newState);
  }, [comparisonEnabled, onFeatureToggle]);

  // Handle animations toggle
  const handleAnimationsToggle = useCallback(() => {
    const newState = !animationsEnabled;
    setAnimationsEnabled(newState);

    if (newState) {
      graphAnimations.start();
      graphAnimations.setAnimationSpeed(animationSpeed);
    } else {
      graphAnimations.pause();
    }

    onFeatureToggle?.('animations', newState);
  }, [animationsEnabled, animationSpeed, onFeatureToggle]);

  // Handle AI toggle
  const handleAiToggle = useCallback(async () => {
    const newState = !aiEnabled;
    setAiEnabled(newState);

    if (newState) {
      // Get AI recommendations
      const recs = await aiInsights.getRecommendations('graph1');
      setRecommendations(recs || []);

      // Detect anomalies
      const anom = await aiInsights.detectAnomalies('graph1');
      setAnomalies(anom || []);
    } else {
      setRecommendations([]);
      setAnomalies([]);
    }

    onFeatureToggle?.('ai-insights', newState);
  }, [aiEnabled, onFeatureToggle]);

  // Handle advanced interactions toggle
  const handleInteractionsToggle = useCallback(() => {
    const newState = !interactionsEnabled;
    setInteractionsEnabled(newState);

    if (newState) {
      advancedInteractionModes.initialize();
    } else {
      // Disable all interaction modes
      if (timeTravelActive) {
        advancedInteractionModes.disableTimeTravel();
        setTimeTravelActive(false);
      }
      if (explorationMode) {
        advancedInteractionModes.disableGuidedExploration();
        setExplorationMode(false);
      }
      if (vrMode) {
        advancedInteractionModes.disableVRAR();
        setVrMode(false);
      }
    }

    onFeatureToggle?.('advanced-interactions', newState);
  }, [interactionsEnabled, timeTravelActive, explorationMode, vrMode, onFeatureToggle]);

  // Handle morphing animation
  const handleMorphing = useCallback(async () => {
    if (!animationsEnabled) return;

    const newState = !morphingActive;
    setMorphingActive(newState);

    if (newState) {
      await graphAnimations.morphBetweenGraphs('graph1', 'graph2', {
        duration: 2000,
        easing: 'easeInOutCubic',
        preserveTopology: true
      });
    }
  }, [animationsEnabled, morphingActive]);

  // Handle time travel mode
  const handleTimeTravel = useCallback(() => {
    if (!interactionsEnabled) return;

    const newState = !timeTravelActive;
    setTimeTravelActive(newState);

    if (newState) {
      advancedInteractionModes.enableTimeTravel({
        maxSnapshots: 50,
        captureInterval: 1000
      });
    } else {
      advancedInteractionModes.disableTimeTravel();
    }
  }, [interactionsEnabled, timeTravelActive]);

  // Handle exploration mode
  const handleExplorationMode = useCallback(() => {
    if (!interactionsEnabled) return;

    const newState = !explorationMode;
    setExplorationMode(newState);

    if (newState) {
      advancedInteractionModes.enableGuidedExploration();
    } else {
      advancedInteractionModes.disableGuidedExploration();
    }
  }, [interactionsEnabled, explorationMode]);

  return (
    <div className={`${styles['features-grid']} ${className}`}>
      {/* Synchronization Panel */}
      <div className={styles['innovative-feature-panel']}>
        <h3>Graph Synchronization</h3>
        <div className={styles['feature-controls']}>
          <label>
            <input
              type="checkbox"
              checked={syncEnabled}
              onChange={handleSyncToggle}
            />
            Enable Dual Graph Sync
          </label>
          {syncEnabled && (
            <div className={styles['sync-options']}>
              <select
                value={syncMode}
                onChange={(e) => setSyncMode(e.target.value as 'full' | 'partial')}
              >
                <option value="full">Full Sync</option>
                <option value="partial">Partial Sync</option>
              </select>
              <div className={styles['sync-status']}>
                Synced Graphs: {syncedGraphs.length}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Comparison Panel */}
      <div className={styles['innovative-feature-panel']}>
        <h3>Graph Comparison</h3>
        <div className={styles['feature-controls']}>
          <label>
            <input
              type="checkbox"
              checked={comparisonEnabled}
              onChange={handleComparisonToggle}
            />
            Enable Visual Comparison
          </label>
          {comparisonEnabled && comparisonResults && (
            <div className={styles['comparison-results']}>
              <div className={styles['similarity-metrics']}>
                Overall Similarity: {(similarities * 100).toFixed(1)}%
              </div>
              <div className={styles['match-count']}>
                Node Matches: {comparisonResults.nodeMatches?.length || 0}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Animation Panel */}
      <div className={styles['innovative-feature-panel']}>
        <h3>Advanced Animations</h3>
        <div className={styles['feature-controls']}>
          <label>
            <input
              type="checkbox"
              checked={animationsEnabled}
              onChange={handleAnimationsToggle}
            />
            Enable Smooth Animations
          </label>
          {animationsEnabled && (
            <div className={styles['animation-controls']}>
              <label>
                Speed:
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={animationSpeed}
                  onChange={(e) => {
                    const speed = parseFloat(e.target.value);
                    setAnimationSpeed(speed);
                    graphAnimations.setAnimationSpeed(speed);
                  }}
                />
                {animationSpeed}x
              </label>
              <button
                onClick={handleMorphing}
                disabled={!animationsEnabled}
              >
                {morphingActive ? 'Stop Morphing' : 'Start Morphing'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* AI Insights Panel */}
      <div className={styles['innovative-feature-panel']}>
        <h3>AI-Powered Insights</h3>
        <div className={styles['feature-controls']}>
          <label>
            <input
              type="checkbox"
              checked={aiEnabled}
              onChange={handleAiToggle}
            />
            Enable AI Analysis
          </label>
          {aiEnabled && (
            <div className={styles['ai-controls']}>
              {recommendations.length > 0 && (
                <div className={styles['recommendations']}>
                  <div>Recommendations: {recommendations.length}</div>
                  <div className={styles['recommendation-list']}>
                    {recommendations.slice(0, 3).map((rec, idx) => (
                      <div key={idx} className={styles['recommendation-item']}>
                        <span className={styles['rec-type']}>{rec.type}</span>
                        <span className={styles['rec-confidence']}>
                          {(rec.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {anomalies.length > 0 && (
                <div className={styles['anomaly-count']}>
                  Anomalies Detected: {anomalies.length}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Advanced Interactions Panel */}
      <div className={styles['innovative-feature-panel']}>
        <h3>Advanced Interactions</h3>
        <div className={styles['feature-controls']}>
          <label>
            <input
              type="checkbox"
              checked={interactionsEnabled}
              onChange={handleInteractionsToggle}
            />
            Enable Advanced Modes
          </label>
          {interactionsEnabled && (
            <div className={styles['interaction-modes']}>
              <button
                onClick={handleTimeTravel}
                disabled={!interactionsEnabled}
              >
                {timeTravelActive ? 'Disable' : 'Enable'} Time Travel
              </button>
              <button
                onClick={handleExplorationMode}
                disabled={!interactionsEnabled}
              >
                {explorationMode ? 'Exit' : 'Enter'} Exploration Mode
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Overall Status */}
      <div className={styles['innovative-feature-panel']}>
        <div className={styles['feature-status']}>
          <h4>Innovation Status</h4>
          <div className={styles['status-indicators']}>
            <div className={`${styles['status-item']} ${syncEnabled ? styles['active'] : ''}`}>
              Synchronization {syncEnabled ? 'Active' : 'Inactive'}
            </div>
            <div className={`${styles['status-item']} ${comparisonEnabled ? styles['active'] : ''}`}>
              Comparison {comparisonEnabled ? 'Active' : 'Inactive'}
            </div>
            <div className={`${styles['status-item']} ${animationsEnabled ? styles['active'] : ''}`}>
              Animations {animationsEnabled ? 'Active' : 'Inactive'}
            </div>
            <div className={`${styles['status-item']} ${aiEnabled ? styles['active'] : ''}`}>
              AI Insights {aiEnabled ? 'Active' : 'Inactive'}
            </div>
            <div className={`${styles['status-item']} ${interactionsEnabled ? styles['active'] : ''}`}>
              Advanced Interactions {interactionsEnabled ? 'Active' : 'Inactive'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GraphFeatures;