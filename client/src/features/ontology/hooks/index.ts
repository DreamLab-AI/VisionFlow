export { useInferenceService } from './useInferenceService';
export type {
  RunInferenceRequest,
  RunInferenceResponse,
  ValidateOntologyRequest,
  OntologyClassification,
} from './useInferenceService';

export {
  useOntologyContributionStore,
  useOntologyClasses,
  useOntologyProperties,
  useOntologyProposals,
  useOntologyLoading,
  useOntologyError,
} from './useOntologyStore';
export type {
  OntologyClass,
  OntologyProperty,
  OntologyAnnotation,
  OntologyProposal,
  OntologyTreeNode,
  ProposalType,
  ProposalStatus,
} from './useOntologyStore';
