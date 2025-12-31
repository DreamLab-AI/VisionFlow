/**
 * Ontology Services
 *
 * Exports services for ontology data integration with JSS.
 */

export {
  jssOntologyService,
  default as JssOntologyService,
} from './JssOntologyService';

export type {
  JsonLdContext,
  JsonLdOntology,
  JsonLdNode,
  OntologyChangeEvent,
  OntologyChangeCallback,
  FetchOptions,
} from './JssOntologyService';
