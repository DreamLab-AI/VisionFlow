import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useOntologyStore } from '../../../ontology/store/useOntologyStore';
import { createLogger } from '../../../../utils/loggerConfig';

const logger = createLogger('SemanticZoomControls');

interface SemanticZoomControlsProps {
  className?: string;
}

/**
 * Semantic zoom controls for hierarchical ontology visualization
 */
export const SemanticZoomControls: React.FC<SemanticZoomControlsProps> = ({ className = '' }) => {
  // Type assertion for extended ontology store methods that may not be in base type
  const store = useOntologyStore() as any;
  const {
    hierarchy,
    semanticZoomLevel,
    expandedClasses,
    visibleClasses = new Set(),
    setZoomLevel = () => {},
    expandAll = () => {},
    collapseAll = () => {},
    toggleClassVisibility = () => {},
  } = store;

  const [autoZoom, setAutoZoom] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const autoZoomRef = useRef(false);

  // Distance thresholds for zoom level mapping
  const DISTANCE_MIN = 100;  // At or below: level 0 (All Instances)
  const DISTANCE_MAX = 1000; // At or above: level 5 (Top Classes)
  const ZOOM_LEVELS = 6;     // 0..5

  /**
   * Maps a camera distance to a semantic zoom level (0-5).
   * close (<100) = 0, far (>1000) = 5, linear interpolation between.
   */
  const distanceToZoomLevel = useCallback((distance: number): number => {
    if (distance <= DISTANCE_MIN) return 0;
    if (distance >= DISTANCE_MAX) return ZOOM_LEVELS - 1;
    const t = (distance - DISTANCE_MIN) / (DISTANCE_MAX - DISTANCE_MIN);
    return Math.round(t * (ZOOM_LEVELS - 1));
  }, []);

  // Listen for camera-distance-change custom events when auto-zoom is enabled.
  // Scene components inside the R3F Canvas should dispatch:
  //   window.dispatchEvent(new CustomEvent('camera-distance-change', { detail: { distance: number } }))
  // on each frame or on OrbitControls change.
  useEffect(() => {
    autoZoomRef.current = autoZoom;

    if (!autoZoom) return;

    const onCameraDistance = (e: Event) => {
      if (!autoZoomRef.current) return;
      const detail = (e as CustomEvent<{ distance: number }>).detail;
      if (detail && typeof detail.distance === 'number') {
        const level = distanceToZoomLevel(detail.distance);
        setZoomLevel(level);
      }
    };

    window.addEventListener('camera-distance-change', onCameraDistance);
    logger.info('Auto-zoom listener attached');

    return () => {
      window.removeEventListener('camera-distance-change', onCameraDistance);
      logger.info('Auto-zoom listener detached');
    };
  }, [autoZoom, distanceToZoomLevel, setZoomLevel]);

  if (!hierarchy) {
    return (
      <div className={`semantic-zoom-controls ${className}`}>
        <p className="text-gray-400">Loading hierarchy...</p>
      </div>
    );
  }

  const handleZoomChange = (newLevel: number) => {
    if (autoZoom) return; // Ignore manual changes while auto-zoom is active
    setZoomLevel(newLevel);
    logger.info('Manual zoom level changed', { level: newLevel });
  };

  const handleAutoZoomToggle = () => {
    const next = !autoZoom;
    setAutoZoom(next);
    logger.info('Auto-zoom toggled', { enabled: next });
  };

  const zoomLevelLabels = [
    'All Instances',
    'Detailed',
    'Standard',
    'Grouped',
    'High-Level',
    'Top Classes'
  ];

  const rootClasses = ((hierarchy as any).rootClasses || [])
    .map((iri: string) => hierarchy.classes.get(iri))
    .filter(Boolean);

  return (
    <div className={`semantic-zoom-controls ${className}`} style={styles.container}>
      {/* Zoom Level Slider */}
      <div style={styles.section}>
        <label style={styles.label}>
          Semantic Zoom Level: {zoomLevelLabels[semanticZoomLevel]}
        </label>
        <div style={styles.sliderContainer}>
          <input
            type="range"
            min="0"
            max="5"
            step="1"
            value={semanticZoomLevel}
            onChange={(e) => handleZoomChange(parseInt(e.target.value))}
            style={styles.slider}
          />
          <div style={styles.sliderLabels}>
            {zoomLevelLabels.map((label, i) => (
              <span key={i} style={styles.sliderLabel}>
                {i}
              </span>
            ))}
          </div>
        </div>
        <div style={styles.zoomInfo}>
          Level {semanticZoomLevel}: {zoomLevelLabels[semanticZoomLevel]}
        </div>
      </div>

      {/* Expand/Collapse Controls */}
      <div style={styles.section}>
        <label style={styles.label}>Expansion Controls</label>
        <div style={styles.buttonGroup}>
          <button
            onClick={expandAll}
            style={styles.button}
            title="Expand all classes"
          >
            Expand All
          </button>
          <button
            onClick={collapseAll}
            style={styles.button}
            title="Collapse all classes"
          >
            Collapse All
          </button>
        </div>
        <div style={styles.info}>
          {expandedClasses.size} / {hierarchy.classes.size} classes expanded
        </div>
      </div>

      {/* Auto-Zoom Toggle */}
      <div style={styles.section}>
        <label style={styles.checkboxLabel}>
          <input
            type="checkbox"
            checked={autoZoom}
            onChange={handleAutoZoomToggle}
            style={styles.checkbox}
          />
          Auto-Zoom (based on camera distance)
        </label>
      </div>

      {/* Class Filters */}
      <div style={styles.section}>
        <button
          onClick={() => setShowFilters(!showFilters)}
          style={styles.button}
        >
          {showFilters ? 'Hide' : 'Show'} Class Filters
        </button>

        {showFilters && (
          <div style={styles.filterList}>
            <div style={styles.filterHeader}>
              Filter by Class ({visibleClasses.size} visible)
            </div>
            {rootClasses.map((classNode: any) => (
              <div key={classNode!.iri} style={styles.filterItem}>
                <label style={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={visibleClasses.has(classNode!.iri)}
                    onChange={() => toggleClassVisibility(classNode!.iri)}
                    style={styles.checkbox}
                  />
                  {classNode!.label}
                  <span style={styles.instanceCount}>
                    ({classNode!.instanceCount} instances)
                  </span>
                </label>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Stats */}
      <div style={styles.section}>
        <div style={styles.stats}>
          <div style={styles.statItem}>
            <span style={styles.statLabel}>Total Classes:</span>
            <span style={styles.statValue}>{hierarchy.classes.size}</span>
          </div>
          <div style={styles.statItem}>
            <span style={styles.statLabel}>Root Classes:</span>
            <span style={styles.statValue}>{((hierarchy as any).rootClasses || []).length}</span>
          </div>
          <div style={styles.statItem}>
            <span style={styles.statLabel}>Max Depth:</span>
            <span style={styles.statValue}>{(hierarchy as any).maxDepth || 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Inline styles for demo (convert to CSS modules in production)
const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderRadius: '8px',
    padding: '16px',
    color: '#ffffff',
    minWidth: '280px',
    maxWidth: '320px',
    backdropFilter: 'blur(10px)',
  },
  section: {
    marginBottom: '16px',
    paddingBottom: '16px',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  },
  label: {
    display: 'block',
    fontSize: '14px',
    fontWeight: 'bold',
    marginBottom: '8px',
    color: '#00ffff',
  },
  sliderContainer: {
    marginBottom: '8px',
  },
  slider: {
    width: '100%',
    height: '6px',
    borderRadius: '3px',
    background: 'linear-gradient(to right, #00ffff, #0066ff)',
    outline: 'none',
    cursor: 'pointer',
  },
  sliderLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    marginTop: '4px',
  },
  sliderLabel: {
    fontSize: '10px',
    color: '#888',
  },
  zoomInfo: {
    fontSize: '12px',
    color: '#aaa',
    textAlign: 'center',
  },
  buttonGroup: {
    display: 'flex',
    gap: '8px',
  },
  button: {
    flex: 1,
    padding: '8px 12px',
    backgroundColor: '#0066ff',
    color: '#ffffff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: '500',
    transition: 'background-color 0.2s',
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    fontSize: '13px',
    cursor: 'pointer',
  },
  checkbox: {
    marginRight: '8px',
    cursor: 'pointer',
  },
  info: {
    fontSize: '12px',
    color: '#aaa',
    marginTop: '8px',
  },
  filterList: {
    marginTop: '12px',
    maxHeight: '200px',
    overflowY: 'auto',
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    borderRadius: '4px',
    padding: '8px',
  },
  filterHeader: {
    fontSize: '12px',
    fontWeight: 'bold',
    marginBottom: '8px',
    color: '#00ffff',
  },
  filterItem: {
    marginBottom: '6px',
  },
  instanceCount: {
    marginLeft: '4px',
    fontSize: '11px',
    color: '#888',
  },
  stats: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  statItem: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '12px',
  },
  statLabel: {
    color: '#aaa',
  },
  statValue: {
    color: '#00ffff',
    fontWeight: 'bold',
  },
};

export default SemanticZoomControls;
