import { create } from 'zustand';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('OntologyStore');

export interface Violation {
  axiomType: string;
  description: string;
  severity: 'error' | 'warning';
  affectedEntities: string[];
}

export interface ConstraintGroup {
  id: string;
  name: string;
  enabled: boolean;
  strength: number;
  description: string;
  constraintCount: number;
  icon?: string;
}

export interface OntologyMetrics {
  axiomCount: number;
  classCount: number;
  propertyCount: number;
  individualCount: number;
  constraintsByType: Record<string, number>;
  cacheHitRate: number;
  validationTimeMs: number;
  lastValidated?: number;
}

export interface ClassNode {
  id: string;
  label: string;
  parentId?: string;
  level: number;
  depth: number;  // Alias for level, for compatibility
  childIds?: string[];
  childIris?: string[];  // Legacy alias for childIds
  instanceCount?: number;
}

export interface OntologyHierarchy {
  classes: Map<string, ClassNode>;
  roots: string[];
}

export interface OntologyState {
  loaded: boolean;
  validating: boolean;
  violations: Violation[];
  constraintGroups: ConstraintGroup[];
  metrics: OntologyMetrics;

  // Hierarchical visualization state
  hierarchy: OntologyHierarchy | null;
  semanticZoomLevel: number;
  expandedClasses: Set<string>;
  highlightedClass: string | null;

  setLoaded: (loaded: boolean) => void;
  setValidating: (validating: boolean) => void;
  setViolations: (violations: Violation[]) => void;
  setMetrics: (metrics: OntologyMetrics) => void;

  toggleConstraintGroup: (id: string) => void;
  updateStrength: (id: string, strength: number) => void;

  // Hierarchical navigation
  toggleClass: (classId: string) => void;
  setSemanticZoomLevel: (level: number) => void;
  setHighlightedClass: (classId: string | null) => void;
  setHierarchy: (hierarchy: OntologyHierarchy | null) => void;

  loadOntology: (fileUrl: string) => Promise<void>;
  validateOntology: () => Promise<void>;
}

export const useOntologyStore = create<OntologyState>((set, get) => ({
  loaded: false,
  validating: false,
  violations: [],
  constraintGroups: [
    {
      id: 'subsumption',
      name: 'Subsumption',
      enabled: true,
      strength: 0.8,
      description: 'Class hierarchy constraints',
      constraintCount: 0,
      icon: 'hierarchy'
    },
    {
      id: 'disjointness',
      name: 'Disjointness',
      enabled: true,
      strength: 1.0,
      description: 'Disjoint class constraints',
      constraintCount: 0,
      icon: 'split'
    },
    {
      id: 'property_domain',
      name: 'Property Domain',
      enabled: true,
      strength: 0.9,
      description: 'Property domain restrictions',
      constraintCount: 0,
      icon: 'arrow-right'
    },
    {
      id: 'property_range',
      name: 'Property Range',
      enabled: true,
      strength: 0.9,
      description: 'Property range restrictions',
      constraintCount: 0,
      icon: 'arrow-left'
    },
    {
      id: 'cardinality',
      name: 'Cardinality',
      enabled: false,
      strength: 0.7,
      description: 'Property cardinality constraints',
      constraintCount: 0,
      icon: 'hash'
    }
  ],
  metrics: {
    axiomCount: 0,
    classCount: 0,
    propertyCount: 0,
    individualCount: 0,
    constraintsByType: {},
    cacheHitRate: 0,
    validationTimeMs: 0
  },

  // Hierarchical visualization state
  hierarchy: null,
  semanticZoomLevel: 0,
  expandedClasses: new Set<string>(),
  highlightedClass: null,

  setLoaded: (loaded) => set({ loaded }),
  setValidating: (validating) => set({ validating }),
  setViolations: (violations) => set({ violations }),
  setMetrics: (metrics) => set({ metrics }),

  toggleConstraintGroup: (id) => set((state) => ({
    constraintGroups: state.constraintGroups.map(group =>
      group.id === id ? { ...group, enabled: !group.enabled } : group
    )
  })),

  updateStrength: (id, strength) => set((state) => ({
    constraintGroups: state.constraintGroups.map(group =>
      group.id === id ? { ...group, strength } : group
    )
  })),

  // Hierarchical navigation methods
  toggleClass: (classId) => set((state) => {
    const newExpanded = new Set(state.expandedClasses);
    if (newExpanded.has(classId)) {
      newExpanded.delete(classId);
    } else {
      newExpanded.add(classId);
    }
    return { expandedClasses: newExpanded };
  }),

  setSemanticZoomLevel: (level) => set({ semanticZoomLevel: level }),

  setHighlightedClass: (classId) => set({ highlightedClass: classId }),

  setHierarchy: (hierarchy) => set({ hierarchy }),

  loadOntology: async (fileUrl: string) => {
    set({ validating: true, violations: [] });
    try {
      const response = await fetch('/api/ontology/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: fileUrl })
      });

      if (!response.ok) {
        throw new Error(`Failed to load ontology: ${response.statusText}`);
      }

      const data = await response.json();
      set({
        loaded: true,
        metrics: data.metrics || get().metrics,
        constraintGroups: data.constraintGroups || get().constraintGroups
      });
    } catch (error) {
      logger.error('Failed to load ontology:', error);
      throw error;
    } finally {
      set({ validating: false });
    }
  },

  validateOntology: async () => {
    set({ validating: true });
    try {
      const response = await fetch('/api/ontology/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          constraintGroups: get().constraintGroups.filter(g => g.enabled)
        })
      });

      if (!response.ok) {
        throw new Error(`Validation failed: ${response.statusText}`);
      }

      const data = await response.json();
      set({
        violations: data.violations || [],
        metrics: { ...get().metrics, ...data.metrics, lastValidated: Date.now() }
      });
    } catch (error) {
      logger.error('Validation failed:', error);
      throw error;
    } finally {
      set({ validating: false });
    }
  }
}));
