import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { GraphCanvas } from '../components/Canvas/GraphCanvas';
import { useGraphStore } from '../stores/useGraphStore';
import { useGraphURLState } from '../hooks/useGraphURLState';
import type { OntologyData } from '../types/ontology';
import './OntologyView.css';

export default function OntologyView() {
  const { nodeId } = useParams<{ nodeId?: string }>();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const setSelectedNode = useGraphStore(state => state.setSelectedNode);
  const loadOntology = useGraphStore(state => state.loadOntology);
  const nodeCount = useGraphStore(state => state.nodes.size);
  const [graphState, updateGraphState] = useGraphURLState();

  // Auto-load ontology on component mount
  useEffect(() => {
    const autoLoadOntology = async () => {
      // Skip if already loaded
      if (nodeCount > 0) {
        setIsLoading(false);
        return;
      }

      try {
        console.log('Loading ontology from /data/ontology.json...');
        const response = await fetch('/data/ontology.json');

        if (!response.ok) {
          throw new Error(`Failed to fetch ontology: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Ontology data loaded:', {
          classes: data.class?.length,
          properties: data.property?.length
        });

        // Validate basic structure
        if (!data.class || !Array.isArray(data.class)) {
          throw new Error('Invalid ontology format: missing class array');
        }

        const ontology: OntologyData = {
          header: data.header,
          namespace: data.namespace,
          class: data.class,
          property: data.property || [],
          datatype: data.datatype,
          classAttribute: data.classAttribute,
          propertyAttribute: data.propertyAttribute
        };

        loadOntology(ontology);
        console.log(`Loaded ${ontology.class.length} classes and ${ontology.property?.length || 0} properties`);
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to auto-load ontology:', err);
        setLoadError(err instanceof Error ? err.message : 'Unknown error');
        setIsLoading(false);
      }
    };

    autoLoadOntology();
  }, [loadOntology, nodeCount]);

  // Sync URL state with graph store
  useEffect(() => {
    if (nodeId) {
      setSelectedNode(nodeId);
    }
  }, [nodeId, setSelectedNode]);

  // Handle node click - navigate to page
  const handleNodeClick = (clickedNodeId: string) => {
    const nodes = useGraphStore.getState().nodes;
    const node = nodes.get(clickedNodeId);

    if (node) {
      // Update URL state
      updateGraphState({ selectedNode: clickedNodeId });

      // Navigate to page using term_id if available, fallback to label
      const pageId = node.properties?.term_id || node.label;
      if (pageId) {
        navigate(`/page/${encodeURIComponent(pageId)}`);
      }
    }
  };

  if (isLoading) {
    return (
      <div className="ontology-view loading">
        <div className="loading-container">
          <div className="spinner" />
          <p>Loading ontology visualization...</p>
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="ontology-view error">
        <div className="error-container">
          <h2>Failed to Load Ontology</h2>
          <p>{loadError}</p>
          <button onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="ontology-view">
      <GraphCanvas onNodeClick={handleNodeClick} />
    </div>
  );
}
