/**
 * JSS Ontology Service
 *
 * Provides live ontology data integration with JavaScript Solid Server (JSS).
 * Replaces static file loading with JSS LDP endpoints for:
 * - Native speed JSON-LD fetch (no parsing overhead)
 * - Content negotiation for Turtle (Protege compatibility)
 * - Real-time WebSocket updates via solid-0.1 protocol
 * - Live graph updates when ontology changes
 *
 * Architecture:
 * - Humans visiting ontology URLs get JSON-LD (native speed for React)
 * - Semantic tools (Protege, reasoners) get Turtle via Accept headers
 * - WebSocket subscriptions trigger graph re-renders on ontology changes
 */

import { createLogger, createErrorMetadata } from '../../../utils/loggerConfig';
import { debugState } from '../../../utils/clientDebugState';
import { nostrAuth } from '../../../services/nostrAuthService';
import { webSocketService, SolidNotification } from '../../../services/WebSocketService';
import { useOntologyStore, OntologyHierarchy, ClassNode, OntologyMetrics } from '../store/useOntologyStore';

const logger = createLogger('JssOntologyService');

// --- Configuration ---

const JSS_BASE_URL = import.meta.env.VITE_JSS_URL || '/solid';
const JSS_WS_URL = import.meta.env.VITE_JSS_WS_URL || null;
const ONTOLOGY_RESOURCE_PATH = import.meta.env.VITE_JSS_ONTOLOGY_PATH || '/public/ontology';

// --- Types ---

export interface JsonLdContext {
  '@vocab'?: string;
  [key: string]: string | object | undefined;
}

export interface JsonLdOntology {
  '@context': JsonLdContext | string;
  '@graph'?: JsonLdNode[];
  '@id'?: string;
  '@type'?: string | string[];
  [key: string]: unknown;
}

export interface JsonLdNode {
  '@id': string;
  '@type'?: string | string[];
  'rdfs:label'?: string | { '@value': string; '@language'?: string };
  'rdfs:subClassOf'?: { '@id': string } | Array<{ '@id': string }>;
  'rdfs:comment'?: string | { '@value': string };
  'owl:disjointWith'?: { '@id': string } | Array<{ '@id': string }>;
  'rdfs:domain'?: { '@id': string };
  'rdfs:range'?: { '@id': string };
  [key: string]: unknown;
}

export interface OntologyChangeEvent {
  type: 'class_added' | 'class_removed' | 'property_added' | 'property_removed' | 'full_refresh';
  resourceUrl: string;
  timestamp: number;
  data?: JsonLdNode;
}

export type OntologyChangeCallback = (event: OntologyChangeEvent) => void;

export interface FetchOptions {
  skipCache?: boolean;
  timeout?: number;
}

// --- Service Implementation ---

class JssOntologyService {
  private static instance: JssOntologyService;

  // Cache for ontology data
  private cachedJsonLd: JsonLdOntology | null = null;
  private cachedTurtle: string | null = null;
  private cacheTimestamp: number = 0;
  private readonly cacheTtlMs: number = 60000; // 1 minute cache

  // WebSocket subscription management
  private changeCallbacks: Set<OntologyChangeCallback> = new Set();
  private unsubscribeFn: (() => void) | null = null;
  private isSubscribed: boolean = false;

  // Metrics tracking
  private fetchCount: number = 0;
  private lastFetchDurationMs: number = 0;

  private constructor() {}

  public static getInstance(): JssOntologyService {
    if (!JssOntologyService.instance) {
      JssOntologyService.instance = new JssOntologyService();
    }
    return JssOntologyService.instance;
  }

  // --- JSON-LD Fetching (Native Speed) ---

  /**
   * Fetch ontology as JSON-LD from JSS
   * This is the native storage format - zero parsing overhead
   */
  public async fetchOntologyJsonLd(options: FetchOptions = {}): Promise<JsonLdOntology> {
    const { skipCache = false, timeout = 30000 } = options;

    // Return cached data if valid
    if (!skipCache && this.isCacheValid()) {
      if (debugState.isEnabled()) {
        logger.debug('Returning cached JSON-LD ontology');
      }
      return this.cachedJsonLd!;
    }

    const startTime = performance.now();
    const url = this.getOntologyUrl();

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await this.fetchWithAuth(url, {
        headers: {
          'Accept': 'application/ld+json',
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Failed to fetch ontology: ${response.status} ${response.statusText}`);
      }

      const data: JsonLdOntology = await response.json();

      // Update cache
      this.cachedJsonLd = data;
      this.cacheTimestamp = Date.now();
      this.lastFetchDurationMs = performance.now() - startTime;
      this.fetchCount++;

      if (debugState.isEnabled()) {
        logger.info('Fetched JSON-LD ontology', {
          durationMs: this.lastFetchDurationMs.toFixed(2),
          graphSize: data['@graph']?.length || 0,
        });
      }

      return data;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Ontology fetch timeout after ${timeout}ms`);
      }
      logger.error('Failed to fetch JSON-LD ontology', createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Fetch ontology and convert to hierarchy structure for visualization
   */
  public async fetchOntologyHierarchy(options: FetchOptions = {}): Promise<OntologyHierarchy> {
    const jsonLd = await this.fetchOntologyJsonLd(options);
    return this.buildHierarchyFromJsonLd(jsonLd);
  }

  // --- Turtle Fetching (Content Negotiation for Protege) ---

  /**
   * Fetch ontology as Turtle via content negotiation
   * Used for Protege and other semantic web tools
   */
  public async fetchOntologyTurtle(options: FetchOptions = {}): Promise<string> {
    const { skipCache = false, timeout = 30000 } = options;

    // Return cached data if valid
    if (!skipCache && this.cachedTurtle && this.isCacheValid()) {
      if (debugState.isEnabled()) {
        logger.debug('Returning cached Turtle ontology');
      }
      return this.cachedTurtle;
    }

    const startTime = performance.now();
    const url = this.getOntologyUrl();

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await this.fetchWithAuth(url, {
        headers: {
          'Accept': 'text/turtle',
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Failed to fetch Turtle ontology: ${response.status} ${response.statusText}`);
      }

      const data = await response.text();

      // Update cache
      this.cachedTurtle = data;
      this.cacheTimestamp = Date.now();
      this.lastFetchDurationMs = performance.now() - startTime;
      this.fetchCount++;

      if (debugState.isEnabled()) {
        logger.info('Fetched Turtle ontology', {
          durationMs: this.lastFetchDurationMs.toFixed(2),
          sizeBytes: data.length,
        });
      }

      return data;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Turtle fetch timeout after ${timeout}ms`);
      }
      logger.error('Failed to fetch Turtle ontology', createErrorMetadata(error));
      throw error;
    }
  }

  // --- WebSocket Real-time Updates ---

  /**
   * Connect to JSS WebSocket for real-time ontology updates
   * Uses solid-0.1 protocol
   */
  public connectWebSocket(): void {
    if (!JSS_WS_URL) {
      logger.warn('JSS WebSocket URL not configured (VITE_JSS_WS_URL)');
      return;
    }

    // Use the main WebSocketService's Solid connection
    webSocketService.connectSolid();

    // Subscribe to ontology resource if not already
    if (!this.isSubscribed) {
      this.subscribeToOntology();
    }
  }

  /**
   * Subscribe to ontology resource changes
   */
  private subscribeToOntology(): void {
    const ontologyUrl = this.getOntologyUrl();

    if (debugState.isEnabled()) {
      logger.info('Subscribing to ontology changes', { url: ontologyUrl });
    }

    // Unsubscribe from previous subscription if any
    if (this.unsubscribeFn) {
      this.unsubscribeFn();
    }

    // Subscribe to the ontology resource
    this.unsubscribeFn = webSocketService.subscribeSolidResource(
      ontologyUrl,
      (notification: SolidNotification) => {
        this.handleSolidNotification(notification);
      }
    );

    this.isSubscribed = true;
  }

  /**
   * Handle Solid notification for ontology changes
   */
  private async handleSolidNotification(notification: SolidNotification): Promise<void> {
    if (notification.type !== 'pub') {
      // Only process 'pub' (change) notifications, not 'ack'
      return;
    }

    if (debugState.isEnabled()) {
      logger.info('Ontology resource changed', { url: notification.url });
    }

    // Invalidate cache
    this.invalidateCache();

    // Fetch fresh data
    try {
      const jsonLd = await this.fetchOntologyJsonLd({ skipCache: true });
      const hierarchy = this.buildHierarchyFromJsonLd(jsonLd);

      // Update the store
      const store = useOntologyStore.getState();
      store.setHierarchy(hierarchy);
      store.setLoaded(true);

      // Notify all change callbacks
      const event: OntologyChangeEvent = {
        type: 'full_refresh',
        resourceUrl: notification.url,
        timestamp: Date.now(),
      };

      this.notifyChangeCallbacks(event);

      if (debugState.isEnabled()) {
        logger.info('Ontology store updated from WebSocket notification');
      }
    } catch (error) {
      logger.error('Failed to refresh ontology after change notification', createErrorMetadata(error));
    }
  }

  /**
   * Register a callback for ontology changes
   * @returns Unsubscribe function
   */
  public onResourceChange(callback: OntologyChangeCallback): () => void {
    this.changeCallbacks.add(callback);

    return () => {
      this.changeCallbacks.delete(callback);
    };
  }

  /**
   * Notify all registered change callbacks
   */
  private notifyChangeCallbacks(event: OntologyChangeEvent): void {
    this.changeCallbacks.forEach((callback) => {
      try {
        callback(event);
      } catch (error) {
        logger.error('Error in ontology change callback', createErrorMetadata(error));
      }
    });
  }

  /**
   * Disconnect WebSocket and clean up subscriptions
   */
  public disconnect(): void {
    if (this.unsubscribeFn) {
      this.unsubscribeFn();
      this.unsubscribeFn = null;
    }
    this.isSubscribed = false;
    this.changeCallbacks.clear();
  }

  // --- Store Integration ---

  /**
   * Load ontology into the store
   * Replaces static file loading with live JSS endpoint
   */
  public async loadIntoStore(options: FetchOptions = {}): Promise<void> {
    const store = useOntologyStore.getState();
    store.setValidating(true);

    try {
      const jsonLd = await this.fetchOntologyJsonLd(options);
      const hierarchy = this.buildHierarchyFromJsonLd(jsonLd);
      const metrics = this.extractMetricsFromJsonLd(jsonLd);

      store.setHierarchy(hierarchy);
      store.setMetrics(metrics);
      store.setLoaded(true);

      if (debugState.isEnabled()) {
        logger.info('Ontology loaded into store', {
          classCount: hierarchy.classes.size,
          rootCount: hierarchy.roots.length,
        });
      }
    } catch (error) {
      logger.error('Failed to load ontology into store', createErrorMetadata(error));
      throw error;
    } finally {
      store.setValidating(false);
    }
  }

  /**
   * Initialize service: load ontology and connect WebSocket
   */
  public async initialize(): Promise<void> {
    // Load initial data
    await this.loadIntoStore();

    // Connect for real-time updates
    this.connectWebSocket();
  }

  // --- Helper Methods ---

  private getOntologyUrl(): string {
    return `${JSS_BASE_URL}${ONTOLOGY_RESOURCE_PATH}`;
  }

  private async fetchWithAuth(
    url: string,
    options: RequestInit = {}
  ): Promise<Response> {
    const token = nostrAuth.getSessionToken();

    const headers = new Headers(options.headers);

    if (token) {
      headers.set('Authorization', `Bearer ${token}`);
    }

    return fetch(url, {
      ...options,
      headers,
      credentials: 'include',
    });
  }

  private isCacheValid(): boolean {
    if (!this.cachedJsonLd) return false;
    return Date.now() - this.cacheTimestamp < this.cacheTtlMs;
  }

  private invalidateCache(): void {
    this.cachedJsonLd = null;
    this.cachedTurtle = null;
    this.cacheTimestamp = 0;
  }

  /**
   * Build OntologyHierarchy from JSON-LD graph
   */
  private buildHierarchyFromJsonLd(jsonLd: JsonLdOntology): OntologyHierarchy {
    const classes = new Map<string, ClassNode>();
    const roots: string[] = [];
    const childMap = new Map<string, string[]>();

    const graph = jsonLd['@graph'] || [];

    // First pass: create all class nodes
    for (const node of graph) {
      if (!this.isOwlClass(node)) continue;

      const id = node['@id'];
      const label = this.extractLabel(node);
      const parentId = this.extractParentId(node);

      classes.set(id, {
        id,
        label,
        parentId,
        level: 0, // Will be computed in second pass
        depth: 0,
        childIds: [],
        instanceCount: 0,
      });

      // Track parent-child relationships
      if (parentId) {
        if (!childMap.has(parentId)) {
          childMap.set(parentId, []);
        }
        childMap.get(parentId)!.push(id);
      }
    }

    // Second pass: assign children and compute levels
    for (const [id, node] of classes) {
      const childIds = childMap.get(id) || [];
      node.childIds = childIds;
      node.childIris = childIds; // Legacy alias

      if (!node.parentId || !classes.has(node.parentId)) {
        roots.push(id);
      }
    }

    // Compute levels (depth from root)
    const computeLevel = (id: string, level: number): void => {
      const node = classes.get(id);
      if (node) {
        node.level = level;
        node.depth = level;
        for (const childId of node.childIds || []) {
          computeLevel(childId, level + 1);
        }
      }
    };

    for (const rootId of roots) {
      computeLevel(rootId, 0);
    }

    return { classes, roots };
  }

  private isOwlClass(node: JsonLdNode): boolean {
    const type = node['@type'];
    if (!type) return false;

    const types = Array.isArray(type) ? type : [type];
    return types.some(
      (t) =>
        t === 'owl:Class' ||
        t === 'http://www.w3.org/2002/07/owl#Class' ||
        t === 'rdfs:Class' ||
        t === 'http://www.w3.org/2000/01/rdf-schema#Class'
    );
  }

  private extractLabel(node: JsonLdNode): string {
    const label = node['rdfs:label'];

    if (!label) {
      // Extract local name from IRI
      const id = node['@id'];
      const hashIndex = id.lastIndexOf('#');
      const slashIndex = id.lastIndexOf('/');
      const index = Math.max(hashIndex, slashIndex);
      return index >= 0 ? id.slice(index + 1) : id;
    }

    if (typeof label === 'string') return label;
    if (typeof label === 'object' && '@value' in label) return label['@value'];

    return node['@id'];
  }

  private extractParentId(node: JsonLdNode): string | undefined {
    const subClassOf = node['rdfs:subClassOf'];
    if (!subClassOf) return undefined;

    if (Array.isArray(subClassOf)) {
      // Return the first non-blank node parent
      for (const parent of subClassOf) {
        if (typeof parent === 'object' && '@id' in parent) {
          const id = parent['@id'];
          if (!id.startsWith('_:')) return id;
        }
      }
      return undefined;
    }

    if (typeof subClassOf === 'object' && '@id' in subClassOf) {
      const id = subClassOf['@id'];
      if (!id.startsWith('_:')) return id;
    }

    return undefined;
  }

  /**
   * Extract metrics from JSON-LD ontology
   */
  private extractMetricsFromJsonLd(jsonLd: JsonLdOntology): OntologyMetrics {
    const graph = jsonLd['@graph'] || [];

    let classCount = 0;
    let propertyCount = 0;
    let individualCount = 0;
    const constraintsByType: Record<string, number> = {};

    for (const node of graph) {
      const types = this.getTypes(node);

      if (types.includes('owl:Class') || types.includes('rdfs:Class')) {
        classCount++;
      }

      if (
        types.includes('owl:ObjectProperty') ||
        types.includes('owl:DatatypeProperty') ||
        types.includes('rdf:Property')
      ) {
        propertyCount++;
      }

      if (types.includes('owl:NamedIndividual')) {
        individualCount++;
      }

      // Count constraints
      if (node['owl:disjointWith']) {
        constraintsByType['disjointness'] = (constraintsByType['disjointness'] || 0) + 1;
      }
      if (node['rdfs:subClassOf']) {
        constraintsByType['subsumption'] = (constraintsByType['subsumption'] || 0) + 1;
      }
      if (node['rdfs:domain']) {
        constraintsByType['property_domain'] = (constraintsByType['property_domain'] || 0) + 1;
      }
      if (node['rdfs:range']) {
        constraintsByType['property_range'] = (constraintsByType['property_range'] || 0) + 1;
      }
    }

    return {
      axiomCount: graph.length,
      classCount,
      propertyCount,
      individualCount,
      constraintsByType,
      cacheHitRate: this.fetchCount > 0 ? 0 : 1, // Placeholder
      validationTimeMs: this.lastFetchDurationMs,
      lastValidated: Date.now(),
    };
  }

  private getTypes(node: JsonLdNode): string[] {
    const type = node['@type'];
    if (!type) return [];
    return Array.isArray(type) ? type : [type];
  }

  // --- Public Getters ---

  public isConnected(): boolean {
    return this.isSubscribed && webSocketService.isSolidWebSocketConnected();
  }

  public getCacheStats(): { hits: number; age: number; valid: boolean } {
    return {
      hits: this.fetchCount,
      age: this.cacheTimestamp > 0 ? Date.now() - this.cacheTimestamp : -1,
      valid: this.isCacheValid(),
    };
  }

  public getLastFetchDuration(): number {
    return this.lastFetchDurationMs;
  }
}

// --- Export Singleton ---

export const jssOntologyService = JssOntologyService.getInstance();
export default jssOntologyService;
