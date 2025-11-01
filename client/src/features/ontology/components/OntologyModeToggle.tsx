import React, { useState, useEffect } from 'react';
import { Button } from '../../design-system/components/Button';
import { Badge } from '../../design-system/components/Badge';
import Network from 'lucide-react/dist/esm/icons/network';
import Database from 'lucide-react/dist/esm/icons/database';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('OntologyModeToggle');

export type GraphMode = 'knowledge_graph' | 'ontology';

interface OntologyModeToggleProps {
  onModeChange?: (mode: GraphMode) => void;
  className?: string;
}

export function OntologyModeToggle({ onModeChange, className = '' }: OntologyModeToggleProps) {
  const settingsStore = useSettingsStore();
  const [mode, setMode] = useState<GraphMode>('knowledge_graph');
  const [loading, setLoading] = useState(false);

  
  useEffect(() => {
    const loadMode = async () => {
      try {
        await settingsStore.ensureLoaded(['visualisation.graphs.mode']);
        const currentMode = settingsStore.get<GraphMode>('visualisation.graphs.mode');
        if (currentMode) {
          setMode(currentMode);
        }
      } catch (error) {
        logger.error('Failed to load graph mode:', error);
      }
    };
    loadMode();
  }, [settingsStore]);

  const handleModeToggle = async () => {
    const newMode: GraphMode = mode === 'knowledge_graph' ? 'ontology' : 'knowledge_graph';
    setLoading(true);

    try {
      
      settingsStore.set('visualisation.graphs.mode', newMode);
      setMode(newMode);

      
      onModeChange?.(newMode);

      logger.info(`Graph mode switched to: ${newMode}`);
    } catch (error) {
      logger.error('Failed to switch graph mode:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchGraphData = async (graphMode: GraphMode) => {
    const endpoint = graphMode === 'knowledge_graph' ? '/api/graph' : '/api/ontology/graph';

    try {
      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${graphMode} data`);
      }
      const data = await response.json();
      logger.info(`Fetched ${graphMode} data:`, data);
      return data;
    } catch (error) {
      logger.error(`Error fetching ${graphMode} data:`, error);
      throw error;
    }
  };

  
  useEffect(() => {
    fetchGraphData(mode).catch(error => {
      logger.error('Failed to fetch graph data:', error);
    });
  }, [mode]);

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <Badge variant={mode === 'knowledge_graph' ? 'default' : 'secondary'}>
        {mode === 'knowledge_graph' ? (
          <>
            <Network className="w-4 h-4 mr-1" />
            Knowledge Graph
          </>
        ) : (
          <>
            <Database className="w-4 h-4 mr-1" />
            Ontology
          </>
        )}
      </Badge>

      <Button
        variant="outline"
        size="sm"
        onClick={handleModeToggle}
        disabled={loading}
      >
        {loading ? 'Switching...' : `Switch to ${mode === 'knowledge_graph' ? 'Ontology' : 'Knowledge Graph'}`}
      </Button>
    </div>
  );
}

export default OntologyModeToggle;
