/**
 * Statistics Service
 * Dynamically loads ontology statistics from ontology.json
 */

import type { OntologyData } from '../types/ontology';

export interface OntologyStatistics {
  totalClasses: number;
  totalProperties: number;
  totalDatatypes: number;
  estimatedTriples: number;
  domains: {
    [domain: string]: {
      count: number;
      percentage: number;
    };
  };
  domainLabels: {
    [domain: string]: string;
  };
}

/**
 * Domain labels for display
 */
const DOMAIN_LABELS: Record<string, string> = {
  'AI': 'Artificial Intelligence',
  'BC': 'Blockchain',
  'RB': 'Robotics',
  'MV': 'Metaverse',
  'DT': 'Disruptive Technologies',
  'TELE': 'Telecollaboration',
};

/**
 * Load ontology and calculate statistics
 */
export async function loadOntologyStatistics(): Promise<OntologyStatistics> {
  try {
    const response = await fetch('./data/ontology.json');
    if (!response.ok) {
      throw new Error('Failed to load ontology.json');
    }

    const ontology: OntologyData = await response.json();

    // Count classes and properties
    const totalClasses = ontology.class?.length || 0;
    const totalProperties = ontology.property?.length || 0;
    const totalDatatypes = ontology.datatype?.length || 0;

    // Estimate RDF triples
    // Rough formula: each class has ~10 triples, each property has ~5 triples
    const estimatedTriples = totalClasses * 10 + totalProperties * 5;

    // Count by domain - use classAttribute array to get baseIri
    const domains: Record<string, number> = {};

    // Build map from class id to its baseIri from classAttribute
    const classToBaseIri = new Map<number, string>();
    ontology.classAttribute?.forEach((attr) => {
      if (attr.id && attr.baseIri) {
        classToBaseIri.set(attr.id, attr.baseIri);
      }
    });

    // Count classes by domain extracted from baseIri
    ontology.class?.forEach((cls) => {
      const baseIri = classToBaseIri.get(cls.id);
      if (baseIri) {
        // Extract domain from URL: .../artificial-intelligence -> AI
        // .../blockchain -> BC, .../robotics -> RB, etc.
        const domainMap: Record<string, string> = {
          'artificial-intelligence': 'AI',
          'blockchain': 'BC',
          'robotics': 'RB',
          'metaverse': 'MV',
          'disruptive-technologies': 'DT',
        };

        for (const [urlPart, prefix] of Object.entries(domainMap)) {
          if (baseIri.includes(urlPart)) {
            domains[prefix] = (domains[prefix] || 0) + 1;
            break;
          }
        }
      }
    });

    // Calculate percentages
    const domainStats: OntologyStatistics['domains'] = {};
    Object.entries(domains).forEach(([domain, count]) => {
      domainStats[domain] = {
        count,
        percentage: Math.round((count / totalClasses) * 100),
      };
    });

    // Build domain labels object
    const domainLabels: Record<string, string> = {};
    Object.keys(domainStats).forEach((domain) => {
      domainLabels[domain] = DOMAIN_LABELS[domain] || domain;
    });

    return {
      totalClasses,
      totalProperties,
      totalDatatypes,
      estimatedTriples,
      domains: domainStats,
      domainLabels,
    };
  } catch (error) {
    console.error('Error loading ontology statistics:', error);
    // Return fallback statistics
    return {
      totalClasses: 0,
      totalProperties: 0,
      totalDatatypes: 0,
      estimatedTriples: 0,
      domains: {},
      domainLabels: DOMAIN_LABELS,
    };
  }
}

/**
 * Format large numbers with commas
 */
export function formatNumber(num: number): string {
  return num.toLocaleString();
}
