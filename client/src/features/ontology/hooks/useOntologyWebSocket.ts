import { useEffect } from 'react';
import { webSocketService } from '../../../services/WebSocketService';
import { useOntologyStore } from '../store/useOntologyStore';

interface OntologyValidationMessage {
  type: 'ontology_validation_update';
  data: {
    status: 'valid' | 'invalid' | 'validating';
    violations: Array<{
      axiomType: string;
      description: string;
      severity: 'error' | 'warning';
      affectedEntities: string[];
    }>;
    metrics?: {
      axiomCount: number;
      classCount: number;
      propertyCount: number;
      individualCount: number;
      constraintsByType: Record<string, number>;
      cacheHitRate: number;
      validationTimeMs: number;
    };
  };
}

interface OntologyLoadedMessage {
  type: 'ontology_loaded';
  data: {
    success: boolean;
    metrics: {
      axiomCount: number;
      classCount: number;
      propertyCount: number;
      individualCount: number;
      constraintsByType: Record<string, number>;
      cacheHitRate: number;
      validationTimeMs: number;
    };
    constraintGroups?: Array<{
      id: string;
      name: string;
      enabled: boolean;
      strength: number;
      description: string;
      constraintCount: number;
    }>;
  };
}

export function useOntologyWebSocket() {
  const { setValidating, setViolations, setMetrics, setLoaded } = useOntologyStore();

  useEffect(() => {
    const unsubscribeValidation = webSocketService.onMessage((message: any) => {
      if (message.type === 'ontology_validation_update') {
        const msg = message as OntologyValidationMessage;

        if (msg.data.status === 'validating') {
          setValidating(true);
        } else {
          setValidating(false);
          setViolations(msg.data.violations);

          if (msg.data.metrics) {
            setMetrics({
              ...msg.data.metrics,
              lastValidated: Date.now()
            });
          }
        }
      }
    });

    const unsubscribeLoad = webSocketService.onMessage((message: any) => {
      if (message.type === 'ontology_loaded') {
        const msg = message as OntologyLoadedMessage;

        if (msg.data.success) {
          setLoaded(true);
          setMetrics({
            ...msg.data.metrics,
            lastValidated: Date.now()
          });
        }
      }
    });

    return () => {
      unsubscribeValidation();
      unsubscribeLoad();
    };
  }, [setValidating, setViolations, setMetrics, setLoaded]);
}
