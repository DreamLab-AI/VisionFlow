import React, { useState, useEffect } from 'react';
import { Html } from '@react-three/drei';
import { ClassNode } from '../../ontology/store/useOntologyStore';

interface ClassGroupTooltipProps {
  classNode: ClassNode;
  instanceCount: number;
  position: [number, number, number];
  visible: boolean;
}

/**
 * Tooltip for class group spheres showing detailed information
 */
export const ClassGroupTooltip: React.FC<ClassGroupTooltipProps> = ({
  classNode,
  instanceCount,
  position,
  visible,
}) => {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (visible) {
      const timer = setTimeout(() => setShow(true), 300);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
    }
  }, [visible]);

  if (!show) return null;

  return (
    <Html position={position} center>
      <div style={styles.tooltip}>
        <div style={styles.header}>
          <h3 style={styles.title}>{classNode.label}</h3>
          <span style={styles.badge}>{instanceCount} instances</span>
        </div>

        <div style={styles.content}>
          {classNode.description && (
            <p style={styles.description}>{classNode.description}</p>
          )}

          <div style={styles.metadataGrid}>
            <div style={styles.metadataItem}>
              <span style={styles.metadataLabel}>IRI:</span>
              <span style={styles.metadataValue}>{classNode.iri}</span>
            </div>

            <div style={styles.metadataItem}>
              <span style={styles.metadataLabel}>Depth:</span>
              <span style={styles.metadataValue}>{classNode.depth}</span>
            </div>

            {classNode.parentIri && (
              <div style={styles.metadataItem}>
                <span style={styles.metadataLabel}>Parent:</span>
                <span style={styles.metadataValue}>
                  {classNode.parentIri.split('/').pop()}
                </span>
              </div>
            )}

            {classNode.childIris.length > 0 && (
              <div style={styles.metadataItem}>
                <span style={styles.metadataLabel}>Children:</span>
                <span style={styles.metadataValue}>
                  {classNode.childIris.length} subclasses
                </span>
              </div>
            )}
          </div>
        </div>

        <div style={styles.footer}>
          <span style={styles.hint}>Click to expand</span>
          <span style={styles.hint}>Double-click to highlight</span>
        </div>
      </div>
    </Html>
  );
};

const styles: Record<string, React.CSSProperties> = {
  tooltip: {
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
    borderRadius: '8px',
    padding: '16px',
    minWidth: '280px',
    maxWidth: '400px',
    color: '#ffffff',
    fontSize: '14px',
    boxShadow: '0 8px 32px rgba(0, 255, 255, 0.3)',
    border: '1px solid rgba(0, 255, 255, 0.5)',
    backdropFilter: 'blur(10px)',
    pointerEvents: 'none',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
    paddingBottom: '8px',
    borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
  },
  title: {
    margin: 0,
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#00ffff',
  },
  badge: {
    backgroundColor: 'rgba(0, 255, 255, 0.2)',
    color: '#00ffff',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '12px',
    fontWeight: 'bold',
  },
  content: {
    marginBottom: '12px',
  },
  description: {
    margin: '0 0 12px 0',
    fontSize: '13px',
    color: '#cccccc',
    lineHeight: '1.4',
  },
  metadataGrid: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  metadataItem: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '12px',
  },
  metadataLabel: {
    color: '#888888',
    fontWeight: '500',
  },
  metadataValue: {
    color: '#ffffff',
    textAlign: 'right',
    maxWidth: '60%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  footer: {
    display: 'flex',
    justifyContent: 'space-between',
    paddingTop: '8px',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
  },
  hint: {
    fontSize: '11px',
    color: '#666666',
    fontStyle: 'italic',
  },
};

export default ClassGroupTooltip;
