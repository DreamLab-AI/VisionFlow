// Components
export { OntologyPanel } from './components/OntologyPanel';
export { ConstraintGroupControl } from './components/ConstraintGroupControl';
export { ValidationStatus } from './components/ValidationStatus';
export { OntologyMetrics } from './components/OntologyMetrics';
export { InferencePanel } from './components/InferencePanel';
export { OntologyContribution } from './components/OntologyContribution';
export { OntologyProposalList } from './components/OntologyProposalList';
export { OntologyBrowser } from './components/OntologyBrowser';

// Store - types and hook
export { useOntologyStore } from './store/useOntologyStore';
export type {
  OntologyState,
  OntologyMetrics as OntologyMetricsType,
  ClassNode,
  OntologyHierarchy,
  Violation,
  ConstraintGroup
} from './store/useOntologyStore';

// Hooks
export * from './hooks/useOntologyWebSocket';
export * from './hooks/useOntologyStore';

// Services
export { jssOntologyService } from './services/JssOntologyService';
export type {
  JsonLdContext,
  JsonLdOntology,
  JsonLdNode,
  OntologyChangeEvent,
  OntologyChangeCallback,
  FetchOptions,
} from './services/JssOntologyService';
