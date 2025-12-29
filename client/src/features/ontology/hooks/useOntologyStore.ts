import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { produce } from 'immer';
import { createLogger, createErrorMetadata } from '../../../utils/loggerConfig';
import { debugState } from '../../../utils/clientDebugState';
import { webSocketService } from '../../../services/WebSocketService';

const logger = createLogger('OntologyContributionStore');

// Ontology entity types
export interface OntologyClass {
  iri: string;
  label: string;
  parentClass?: string;
  description?: string;
  annotations?: Record<string, string>;
}

export interface OntologyProperty {
  iri: string;
  label: string;
  domain?: string;
  range?: string;
  propertyType: 'object' | 'data' | 'annotation';
  description?: string;
}

export interface OntologyAnnotation {
  targetIri: string;
  predicate: string;
  value: string;
  language?: string;
}

export type ProposalType = 'class' | 'property' | 'annotation';
export type ProposalStatus = 'draft' | 'pending' | 'approved' | 'rejected' | 'withdrawn';

export interface OntologyProposal {
  id: string;
  type: ProposalType;
  status: ProposalStatus;
  createdAt: number;
  updatedAt: number;
  submittedAt?: number;
  data: OntologyClass | OntologyProperty | OntologyAnnotation;
  diff?: {
    added: string[];
    removed: string[];
    modified: string[];
  };
  reviewNotes?: string;
  mergeCommit?: string;
}

export interface OntologyTreeNode {
  iri: string;
  label: string;
  type: 'class' | 'property';
  children: OntologyTreeNode[];
  expanded?: boolean;
  propertyCount?: number;
}

interface OntologyContributionState {
  // Public ontology data
  classes: OntologyClass[];
  properties: OntologyProperty[];
  classTree: OntologyTreeNode[];
  propertyTree: OntologyTreeNode[];

  // User proposals
  proposals: OntologyProposal[];

  // UI state
  loading: boolean;
  error: string | null;
  searchQuery: string;
  selectedNode: string | null;

  // WebSocket subscription state
  subscribed: boolean;
  lastUpdate: number | null;

  // Actions
  fetchOntology: () => Promise<void>;
  createProposal: (type: ProposalType, data: OntologyClass | OntologyProperty | OntologyAnnotation) => Promise<OntologyProposal>;
  updateProposal: (id: string, data: Partial<OntologyProposal>) => Promise<void>;
  submitProposal: (id: string) => Promise<void>;
  withdrawProposal: (id: string) => Promise<void>;
  deleteProposal: (id: string) => Promise<void>;

  // Search and browse
  setSearchQuery: (query: string) => void;
  setSelectedNode: (iri: string | null) => void;
  toggleNodeExpanded: (iri: string) => void;

  // WebSocket
  subscribeToUpdates: () => () => void;
  handleOntologyUpdate: (update: any) => void;

  // Utilities
  getClassByIri: (iri: string) => OntologyClass | undefined;
  getPropertyByIri: (iri: string) => OntologyProperty | undefined;
  getProposalById: (id: string) => OntologyProposal | undefined;
  searchOntology: (query: string) => Array<{ type: 'class' | 'property'; item: OntologyClass | OntologyProperty }>;

  // Reset
  clearError: () => void;
  reset: () => void;
}

// Build tree structure from flat class list
function buildClassTree(classes: OntologyClass[]): OntologyTreeNode[] {
  const nodeMap = new Map<string, OntologyTreeNode>();
  const roots: OntologyTreeNode[] = [];

  // Create nodes for all classes
  for (const cls of classes) {
    nodeMap.set(cls.iri, {
      iri: cls.iri,
      label: cls.label,
      type: 'class',
      children: [],
      expanded: false
    });
  }

  // Build hierarchy
  for (const cls of classes) {
    const node = nodeMap.get(cls.iri)!;
    if (cls.parentClass && nodeMap.has(cls.parentClass)) {
      nodeMap.get(cls.parentClass)!.children.push(node);
    } else {
      roots.push(node);
    }
  }

  // Sort by label
  const sortNodes = (nodes: OntologyTreeNode[]): OntologyTreeNode[] => {
    nodes.sort((a, b) => a.label.localeCompare(b.label));
    nodes.forEach(node => sortNodes(node.children));
    return nodes;
  };

  return sortNodes(roots);
}

// Build tree structure for properties grouped by domain
function buildPropertyTree(properties: OntologyProperty[]): OntologyTreeNode[] {
  const domainMap = new Map<string, OntologyProperty[]>();

  for (const prop of properties) {
    const domain = prop.domain || 'owl:Thing';
    if (!domainMap.has(domain)) {
      domainMap.set(domain, []);
    }
    domainMap.get(domain)!.push(prop);
  }

  const roots: OntologyTreeNode[] = [];

  for (const [domain, props] of domainMap) {
    roots.push({
      iri: domain,
      label: domain.split(/[#/]/).pop() || domain,
      type: 'class',
      expanded: false,
      propertyCount: props.length,
      children: props.map(prop => ({
        iri: prop.iri,
        label: prop.label,
        type: 'property' as const,
        children: [],
        expanded: false
      }))
    });
  }

  return roots.sort((a, b) => a.label.localeCompare(b.label));
}

// Generate unique proposal ID
function generateProposalId(): string {
  return `proposal-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

export const useOntologyContributionStore = create<OntologyContributionState>()(
  persist(
    (set, get) => ({
      // Initial state
      classes: [],
      properties: [],
      classTree: [],
      propertyTree: [],
      proposals: [],
      loading: false,
      error: null,
      searchQuery: '',
      selectedNode: null,
      subscribed: false,
      lastUpdate: null,

      fetchOntology: async () => {
        set({ loading: true, error: null });

        try {
          const response = await fetch('/api/ontology/public');

          if (!response.ok) {
            throw new Error(`Failed to fetch ontology: ${response.statusText}`);
          }

          const data = await response.json();

          const classes: OntologyClass[] = data.classes || [];
          const properties: OntologyProperty[] = data.properties || [];

          set({
            classes,
            properties,
            classTree: buildClassTree(classes),
            propertyTree: buildPropertyTree(properties),
            loading: false,
            lastUpdate: Date.now()
          });

          if (debugState.isEnabled()) {
            logger.info('Ontology fetched', {
              classCount: classes.length,
              propertyCount: properties.length
            });
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          logger.error('Failed to fetch ontology:', createErrorMetadata(error));
          set({ loading: false, error: message });
          throw error;
        }
      },

      createProposal: async (type, data) => {
        set({ loading: true, error: null });

        try {
          const proposal: OntologyProposal = {
            id: generateProposalId(),
            type,
            status: 'draft',
            createdAt: Date.now(),
            updatedAt: Date.now(),
            data
          };

          // Save to user's pod
          const response = await fetch('/api/ontology/proposals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(proposal)
          });

          if (!response.ok) {
            throw new Error(`Failed to create proposal: ${response.statusText}`);
          }

          const savedProposal = await response.json();

          set(state => produce(state, draft => {
            draft.proposals.push(savedProposal);
            draft.loading = false;
          }));

          if (debugState.isEnabled()) {
            logger.info('Proposal created', { id: savedProposal.id, type });
          }

          return savedProposal;
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          logger.error('Failed to create proposal:', createErrorMetadata(error));
          set({ loading: false, error: message });
          throw error;
        }
      },

      updateProposal: async (id, updates) => {
        set({ loading: true, error: null });

        try {
          const existingProposal = get().proposals.find(p => p.id === id);
          if (!existingProposal) {
            throw new Error('Proposal not found');
          }

          const updatedProposal: OntologyProposal = {
            ...existingProposal,
            ...updates,
            updatedAt: Date.now()
          };

          const response = await fetch(`/api/ontology/proposals/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updatedProposal)
          });

          if (!response.ok) {
            throw new Error(`Failed to update proposal: ${response.statusText}`);
          }

          set(state => produce(state, draft => {
            const index = draft.proposals.findIndex(p => p.id === id);
            if (index !== -1) {
              draft.proposals[index] = updatedProposal;
            }
            draft.loading = false;
          }));

          if (debugState.isEnabled()) {
            logger.info('Proposal updated', { id });
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          logger.error('Failed to update proposal:', createErrorMetadata(error));
          set({ loading: false, error: message });
          throw error;
        }
      },

      submitProposal: async (id) => {
        set({ loading: true, error: null });

        try {
          const response = await fetch(`/api/ontology/proposals/${id}/submit`, {
            method: 'POST'
          });

          if (!response.ok) {
            throw new Error(`Failed to submit proposal: ${response.statusText}`);
          }

          const result = await response.json();

          set(state => produce(state, draft => {
            const index = draft.proposals.findIndex(p => p.id === id);
            if (index !== -1) {
              draft.proposals[index].status = 'pending';
              draft.proposals[index].submittedAt = Date.now();
              draft.proposals[index].updatedAt = Date.now();
              if (result.diff) {
                draft.proposals[index].diff = result.diff;
              }
            }
            draft.loading = false;
          }));

          if (debugState.isEnabled()) {
            logger.info('Proposal submitted', { id });
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          logger.error('Failed to submit proposal:', createErrorMetadata(error));
          set({ loading: false, error: message });
          throw error;
        }
      },

      withdrawProposal: async (id) => {
        set({ loading: true, error: null });

        try {
          const response = await fetch(`/api/ontology/proposals/${id}/withdraw`, {
            method: 'POST'
          });

          if (!response.ok) {
            throw new Error(`Failed to withdraw proposal: ${response.statusText}`);
          }

          set(state => produce(state, draft => {
            const index = draft.proposals.findIndex(p => p.id === id);
            if (index !== -1) {
              draft.proposals[index].status = 'withdrawn';
              draft.proposals[index].updatedAt = Date.now();
            }
            draft.loading = false;
          }));

          if (debugState.isEnabled()) {
            logger.info('Proposal withdrawn', { id });
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          logger.error('Failed to withdraw proposal:', createErrorMetadata(error));
          set({ loading: false, error: message });
          throw error;
        }
      },

      deleteProposal: async (id) => {
        set({ loading: true, error: null });

        try {
          const response = await fetch(`/api/ontology/proposals/${id}`, {
            method: 'DELETE'
          });

          if (!response.ok) {
            throw new Error(`Failed to delete proposal: ${response.statusText}`);
          }

          set(state => produce(state, draft => {
            draft.proposals = draft.proposals.filter(p => p.id !== id);
            draft.loading = false;
          }));

          if (debugState.isEnabled()) {
            logger.info('Proposal deleted', { id });
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          logger.error('Failed to delete proposal:', createErrorMetadata(error));
          set({ loading: false, error: message });
          throw error;
        }
      },

      setSearchQuery: (query) => {
        set({ searchQuery: query });
      },

      setSelectedNode: (iri) => {
        set({ selectedNode: iri });
      },

      toggleNodeExpanded: (iri) => {
        set(state => {
          const toggleInTree = (nodes: OntologyTreeNode[]): OntologyTreeNode[] => {
            return nodes.map(node => ({
              ...node,
              expanded: node.iri === iri ? !node.expanded : node.expanded,
              children: toggleInTree(node.children)
            }));
          };

          return {
            classTree: toggleInTree(state.classTree),
            propertyTree: toggleInTree(state.propertyTree)
          };
        });
      },

      subscribeToUpdates: () => {
        const handleUpdate = get().handleOntologyUpdate;

        // Subscribe to WebSocket ontology updates
        const unsubscribe = webSocketService.on('ontology_update', handleUpdate);

        // Request subscription
        webSocketService.sendMessage('subscribe_ontology', {});

        set({ subscribed: true });

        if (debugState.isEnabled()) {
          logger.info('Subscribed to ontology updates');
        }

        return () => {
          unsubscribe();
          webSocketService.sendMessage('unsubscribe_ontology', {});
          set({ subscribed: false });

          if (debugState.isEnabled()) {
            logger.info('Unsubscribed from ontology updates');
          }
        };
      },

      handleOntologyUpdate: (update) => {
        if (debugState.isEnabled()) {
          logger.info('Received ontology update', update);
        }

        set(state => {
          switch (update.type) {
            case 'class_added':
              return produce(state, draft => {
                draft.classes.push(update.data);
                draft.classTree = buildClassTree(draft.classes);
                draft.lastUpdate = Date.now();
              });

            case 'class_removed':
              return produce(state, draft => {
                draft.classes = draft.classes.filter(c => c.iri !== update.iri);
                draft.classTree = buildClassTree(draft.classes);
                draft.lastUpdate = Date.now();
              });

            case 'property_added':
              return produce(state, draft => {
                draft.properties.push(update.data);
                draft.propertyTree = buildPropertyTree(draft.properties);
                draft.lastUpdate = Date.now();
              });

            case 'property_removed':
              return produce(state, draft => {
                draft.properties = draft.properties.filter(p => p.iri !== update.iri);
                draft.propertyTree = buildPropertyTree(draft.properties);
                draft.lastUpdate = Date.now();
              });

            case 'proposal_status_changed':
              return produce(state, draft => {
                const index = draft.proposals.findIndex(p => p.id === update.proposalId);
                if (index !== -1) {
                  draft.proposals[index].status = update.status;
                  draft.proposals[index].updatedAt = Date.now();
                  if (update.reviewNotes) {
                    draft.proposals[index].reviewNotes = update.reviewNotes;
                  }
                  if (update.mergeCommit) {
                    draft.proposals[index].mergeCommit = update.mergeCommit;
                  }
                }
                draft.lastUpdate = Date.now();
              });

            case 'full_refresh':
              return produce(state, draft => {
                draft.classes = update.classes || [];
                draft.properties = update.properties || [];
                draft.classTree = buildClassTree(draft.classes);
                draft.propertyTree = buildPropertyTree(draft.properties);
                draft.lastUpdate = Date.now();
              });

            default:
              return state;
          }
        });
      },

      getClassByIri: (iri) => {
        return get().classes.find(c => c.iri === iri);
      },

      getPropertyByIri: (iri) => {
        return get().properties.find(p => p.iri === iri);
      },

      getProposalById: (id) => {
        return get().proposals.find(p => p.id === id);
      },

      searchOntology: (query) => {
        const lowerQuery = query.toLowerCase();
        const results: Array<{ type: 'class' | 'property'; item: OntologyClass | OntologyProperty }> = [];

        for (const cls of get().classes) {
          if (
            cls.label.toLowerCase().includes(lowerQuery) ||
            cls.iri.toLowerCase().includes(lowerQuery) ||
            cls.description?.toLowerCase().includes(lowerQuery)
          ) {
            results.push({ type: 'class', item: cls });
          }
        }

        for (const prop of get().properties) {
          if (
            prop.label.toLowerCase().includes(lowerQuery) ||
            prop.iri.toLowerCase().includes(lowerQuery) ||
            prop.description?.toLowerCase().includes(lowerQuery)
          ) {
            results.push({ type: 'property', item: prop });
          }
        }

        return results;
      },

      clearError: () => {
        set({ error: null });
      },

      reset: () => {
        set({
          classes: [],
          properties: [],
          classTree: [],
          propertyTree: [],
          proposals: [],
          loading: false,
          error: null,
          searchQuery: '',
          selectedNode: null,
          subscribed: false,
          lastUpdate: null
        });
      }
    }),
    {
      name: 'ontology-contribution-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        proposals: state.proposals.filter(p => p.status === 'draft')
      }),
      onRehydrateStorage: () => (state) => {
        if (state && debugState.isEnabled()) {
          logger.info('Ontology contribution store rehydrated', {
            draftProposals: state.proposals?.length || 0
          });
        }
      }
    }
  )
);

// Convenience hooks
export const useOntologyClasses = () => useOntologyContributionStore(state => state.classes);
export const useOntologyProperties = () => useOntologyContributionStore(state => state.properties);
export const useOntologyProposals = () => useOntologyContributionStore(state => state.proposals);
export const useOntologyLoading = () => useOntologyContributionStore(state => state.loading);
export const useOntologyError = () => useOntologyContributionStore(state => state.error);
