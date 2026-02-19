/**
 * Solid Pod Service
 *
 * Provides integration with JavaScript Solid Server (JSS) for:
 * - Pod management (create, check, access)
 * - LDP CRUD operations
 * - WebSocket notifications (solid-0.1 protocol)
 * - Content negotiation (JSON-LD, Turtle)
 *
 * Works with VisionFlow's Nostr authentication system.
 */

import { createLogger } from '../utils/loggerConfig';
import { nostrAuth } from './nostrAuthService';
import { webSocketRegistry } from './WebSocketRegistry';
import { webSocketEventBus } from './WebSocketEventBus';

const logger = createLogger('SolidPodService');

const REGISTRY_NAME = 'solid-pod';

// --- Interfaces ---

/**
 * Pod directory structure matching backend PodStructure
 */
export interface PodStructure {
  profile: string;
  ontology_contributions: string;
  ontology_proposals: string;
  ontology_annotations: string;
  preferences: string;
  inbox: string;
}

export interface PodInfo {
  exists: boolean;
  podUrl?: string;
  webId?: string;
  suggestedUrl?: string;
  structure?: PodStructure;
}

export interface PodCreationResult {
  success: boolean;
  podUrl?: string;
  webId?: string;
  created?: boolean;
  structure?: PodStructure;
  error?: string;
}

export interface PodInitResult {
  success: boolean;
  podUrl?: string;
  webId?: string;
  created: boolean;
  structure?: PodStructure;
  npub?: string;
  error?: string;
}

export interface JsonLdDocument {
  '@context': string | object;
  '@type'?: string;
  '@id'?: string;
  [key: string]: unknown;
}

export interface SolidNotification {
  type: 'pub' | 'ack';
  url: string;
}

type NotificationCallback = (notification: SolidNotification) => void;

// --- Configuration ---

const JSS_BASE_URL = import.meta.env.VITE_JSS_URL || '/solid';
const JSS_WS_URL = import.meta.env.VITE_JSS_WS_URL || null;

// --- Service Implementation ---

/**
 * Sanitize a preference key to prevent path traversal.
 * Strips path separators and dot-sequences that could escape the container.
 */
function sanitizePreferenceKey(key: string): string {
  // Remove sequences of two or more dots/slashes/backslashes (path traversal)
  let sanitized = key.replace(/[\/\\\.]{2,}/g, '');
  // Replace any remaining slashes/backslashes with hyphens
  sanitized = sanitized.replace(/[\/\\]/g, '-');
  // Remove leading dots or hyphens
  sanitized = sanitized.replace(/^[.\-]+/, '');
  if (!sanitized) {
    throw new Error('Invalid preference key');
  }
  return sanitized;
}

class SolidPodService {
  private static instance: SolidPodService;
  private wsConnection: WebSocket | null = null;
  private subscriptions: Map<string, Set<NotificationCallback>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private reconnectTimerId: ReturnType<typeof setTimeout> | null = null;
  private isDisconnecting = false;

  private constructor() {}

  public static getInstance(): SolidPodService {
    if (!SolidPodService.instance) {
      SolidPodService.instance = new SolidPodService();
    }
    return SolidPodService.instance;
  }

  // --- Pod Management ---

  /**
   * Check if the current user has a pod
   */
  public async checkPodExists(): Promise<PodInfo> {
    try {
      const response = await this.fetchWithAuth(`${JSS_BASE_URL}/pods/check`);

      if (!response.ok) {
        throw new Error(`Failed to check pod: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      logger.error('Failed to check pod existence', { error });
      return { exists: false };
    }
  }

  /**
   * Create a pod for the current user
   */
  public async createPod(name?: string): Promise<PodCreationResult> {
    try {
      const response = await this.fetchWithAuth(`${JSS_BASE_URL}/pods`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });

      if (!response.ok) {
        const error = await response.json();
        return {
          success: false,
          error: error.error || 'Pod creation failed',
        };
      }

      const result = await response.json();
      logger.info('Pod created successfully', { podUrl: result.pod_url });

      return {
        success: true,
        podUrl: result.pod_url,
        webId: result.webid,
      };
    } catch (error) {
      logger.error('Failed to create pod', { error });
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Get the user's pod URL
   */
  public async getPodUrl(): Promise<string | null> {
    const info = await this.checkPodExists();
    return info.podUrl || null;
  }

  /**
   * Initialize pod for the current user (auto-provision if needed)
   * Call this on user login to ensure their pod exists with full structure
   */
  public async initPod(): Promise<PodInitResult> {
    try {
      const response = await this.fetchWithAuth(`${JSS_BASE_URL}/pods/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Initialization failed' }));
        return {
          success: false,
          created: false,
          error: error.error || 'Pod initialization failed',
        };
      }

      const result = await response.json();
      logger.info('Pod initialized', {
        podUrl: result.pod_url,
        created: result.created,
      });

      return {
        success: true,
        podUrl: result.pod_url,
        webId: result.webid,
        created: result.created,
        structure: result.structure,
      };
    } catch (error) {
      logger.error('Failed to initialize pod', { error });
      return {
        success: false,
        created: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Ensure user's pod exists, creating if necessary
   * Convenience wrapper that checks and creates in one call
   */
  public async ensurePodExists(): Promise<PodInitResult> {
    return this.initPod();
  }

  /**
   * Get the structure of the user's pod (directory URLs)
   */
  public async getPodStructure(): Promise<PodStructure | null> {
    const result = await this.initPod();
    return result.structure || null;
  }

  /**
   * Store a preference in the user's preferences container
   */
  public async setPreference(key: string, value: unknown): Promise<boolean> {
    const structure = await this.getPodStructure();
    if (!structure) {
      logger.error('Cannot set preference: pod not initialized');
      return false;
    }

    const safeKey = sanitizePreferenceKey(key);
    const prefPath = `${structure.preferences}${safeKey}.jsonld`;
    return this.putResource(prefPath, {
      '@context': 'https://www.w3.org/ns/ldp',
      '@type': 'Preference',
      key,
      value,
      modified: new Date().toISOString(),
    });
  }

  /**
   * Get a preference from the user's preferences container
   */
  public async getPreference(key: string): Promise<unknown | null> {
    const structure = await this.getPodStructure();
    if (!structure) {
      logger.error('Cannot get preference: pod not initialized');
      return null;
    }

    try {
      const safeKey = sanitizePreferenceKey(key);
      const prefPath = `${structure.preferences}${safeKey}.jsonld`;
      const doc = await this.fetchJsonLd(prefPath);
      return (doc as { value?: unknown }).value ?? null;
    } catch {
      return null;
    }
  }

  /**
   * Submit an ontology contribution to the user's pod
   */
  public async submitOntologyContribution(contribution: {
    title: string;
    description: string;
    changes: unknown;
  }): Promise<string | null> {
    const structure = await this.getPodStructure();
    if (!structure) {
      logger.error('Cannot submit contribution: pod not initialized');
      return null;
    }

    const slug = `contrib-${Date.now()}`;
    return this.postResource(structure.ontology_contributions, {
      '@context': {
        '@vocab': 'https://narrativegoldmine.com/ontology#',
        schema: 'https://schema.org/',
      },
      '@type': 'OntologyContribution',
      'schema:name': contribution.title,
      'schema:description': contribution.description,
      changes: contribution.changes,
      status: 'draft',
      'schema:dateCreated': new Date().toISOString(),
    }, slug);
  }

  /**
   * Submit an ontology proposal for review
   */
  public async submitOntologyProposal(proposal: {
    title: string;
    description: string;
    contributionUrl: string;
  }): Promise<string | null> {
    const structure = await this.getPodStructure();
    if (!structure) {
      logger.error('Cannot submit proposal: pod not initialized');
      return null;
    }

    const slug = `proposal-${Date.now()}`;
    return this.postResource(structure.ontology_proposals, {
      '@context': {
        '@vocab': 'https://narrativegoldmine.com/ontology#',
        schema: 'https://schema.org/',
      },
      '@type': 'OntologyProposal',
      'schema:name': proposal.title,
      'schema:description': proposal.description,
      contribution: proposal.contributionUrl,
      status: 'pending',
      'schema:dateCreated': new Date().toISOString(),
    }, slug);
  }

  // --- Connection Methods ---

  /**
   * Connect to a user's pod by their npub (Nostr public key in bech32 format)
   * Sets up the pod context for subsequent operations
   * @param npub - Nostr public key in npub format (e.g., npub1...)
   * @returns Pod connection info
   */
  public async connectToPod(npub: string): Promise<PodInfo> {
    logger.info('Connecting to pod for npub', { npub });

    // Validate npub format
    if (!npub.startsWith('npub1')) {
      throw new Error('Invalid npub format. Expected npub1...');
    }

    try {
      const response = await this.fetchWithAuth(`${JSS_BASE_URL}/pods/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ npub }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Connection failed' }));
        throw new Error(error.error || `Pod connection failed: ${response.status}`);
      }

      const info = await response.json();
      logger.info('Connected to pod', { podUrl: info.podUrl });

      // Auto-connect WebSocket if configured
      if (JSS_WS_URL && !this.wsConnection) {
        this.connectWebSocket();
      }

      return {
        exists: true,
        podUrl: info.podUrl || info.pod_url,
        webId: info.webId || info.webid,
      };
    } catch (error) {
      logger.error('Failed to connect to pod', { npub, error });
      throw error;
    }
  }

  // --- LDP Operations ---

  /**
   * Read a resource from the pod as JSON-LD
   * Alias for fetchJsonLd with enhanced error handling
   * @param path - Resource path relative to pod or absolute URL
   * @returns JSON-LD document
   */
  public async readResource(path: string): Promise<JsonLdDocument> {
    return this.fetchJsonLd(path);
  }

  /**
   * Write a resource to the pod
   * Alias for putResource with JSON-LD default and flexible content type
   * @param path - Resource path relative to pod or absolute URL
   * @param content - Content to write (object or string)
   * @returns Success status
   */
  public async writeResource(
    path: string,
    content: JsonLdDocument | Record<string, unknown> | string
  ): Promise<boolean> {
    // Convert plain objects to JSON-LD format
    const jsonLdContent = this.ensureJsonLd(content);
    return this.putResource(path, jsonLdContent);
  }

  /**
   * Subscribe to changes on a resource path
   * Wrapper around subscribe with simplified callback
   * @param path - Resource path to watch
   * @param callback - Function called when resource changes
   * @returns Unsubscribe function
   */
  public subscribeToChanges(
    path: string,
    callback: (url: string, type: 'created' | 'updated' | 'deleted') => void
  ): () => void {
    const resourceUrl = this.resolvePath(path);

    // Ensure WebSocket is connected
    if (!this.wsConnection) {
      this.connectWebSocket();
    }

    return this.subscribe(resourceUrl, (notification) => {
      if (notification.type === 'pub') {
        // Determine change type based on resource state
        callback(notification.url, 'updated');
      }
    });
  }

  /**
   * Fetch a resource as JSON-LD
   */
  public async fetchJsonLd(resourcePath: string): Promise<JsonLdDocument> {
    const url = this.resolvePath(resourcePath);
    const response = await this.fetchWithAuth(url, {
      headers: { Accept: 'application/ld+json' },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch ${resourcePath}: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Fetch a resource as Turtle (for external tools)
   */
  public async fetchTurtle(resourcePath: string): Promise<string> {
    const url = this.resolvePath(resourcePath);
    const response = await this.fetchWithAuth(url, {
      headers: { Accept: 'text/turtle' },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch Turtle ${resourcePath}: ${response.status}`);
    }

    return response.text();
  }

  /**
   * Create or update a resource
   */
  public async putResource(
    resourcePath: string,
    data: JsonLdDocument | string,
    contentType: 'application/ld+json' | 'text/turtle' = 'application/ld+json'
  ): Promise<boolean> {
    const url = this.resolvePath(resourcePath);
    const body = typeof data === 'string' ? data : JSON.stringify(data);

    const response = await this.fetchWithAuth(url, {
      method: 'PUT',
      headers: { 'Content-Type': contentType },
      body,
    });

    if (!response.ok) {
      logger.error('PUT failed', { resourcePath, status: response.status });
      return false;
    }

    logger.debug('Resource updated', { resourcePath });
    return true;
  }

  /**
   * Create a resource in a container (POST)
   */
  public async postResource(
    containerPath: string,
    data: JsonLdDocument,
    slug?: string
  ): Promise<string | null> {
    const url = this.resolvePath(containerPath);
    const headers: Record<string, string> = {
      'Content-Type': 'application/ld+json',
    };

    if (slug) {
      headers['Slug'] = slug;
    }

    const response = await this.fetchWithAuth(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      logger.error('POST failed', { containerPath, status: response.status });
      return null;
    }

    // Return the created resource URL from Location header
    return response.headers.get('Location');
  }

  /**
   * Delete a resource
   */
  public async deleteResource(resourcePath: string): Promise<boolean> {
    const url = this.resolvePath(resourcePath);

    const response = await this.fetchWithAuth(url, {
      method: 'DELETE',
    });

    if (!response.ok && response.status !== 404) {
      logger.error('DELETE failed', { resourcePath, status: response.status });
      return false;
    }

    return true;
  }

  /**
   * Check if a resource exists (HEAD)
   */
  public async resourceExists(resourcePath: string): Promise<boolean> {
    const url = this.resolvePath(resourcePath);

    try {
      const response = await this.fetchWithAuth(url, { method: 'HEAD' });
      return response.ok;
    } catch {
      return false;
    }
  }

  // --- WebSocket Notifications ---

  /**
   * Connect to JSS WebSocket for real-time notifications
   */
  public connectWebSocket(): void {
    if (!JSS_WS_URL) {
      logger.warn('JSS WebSocket URL not configured');
      return;
    }

    if (this.wsConnection?.readyState === WebSocket.OPEN) {
      logger.debug('WebSocket already connected');
      return;
    }

    try {
      // Validate WebSocket URL before connecting
      const validatedUrl = new URL(JSS_WS_URL);
      if (validatedUrl.protocol !== 'ws:' && validatedUrl.protocol !== 'wss:') {
        logger.error('Invalid WebSocket protocol', { protocol: validatedUrl.protocol });
        return;
      }
      this.wsConnection = new WebSocket(validatedUrl.href);

      this.wsConnection.onopen = () => {
        logger.info('JSS WebSocket connected');
        this.reconnectAttempts = 0;
        webSocketRegistry.register(REGISTRY_NAME, validatedUrl.href, this.wsConnection!);
        webSocketEventBus.emit('connection:open', { name: REGISTRY_NAME, url: validatedUrl.href });
        // Protocol handshake will be handled in onmessage
      };

      this.wsConnection.onmessage = (event) => {
        const msg = event.data.toString().trim();
        webSocketEventBus.emit('message:pod', { data: msg });
        this.handleWebSocketMessage(msg);
      };

      this.wsConnection.onerror = (error) => {
        logger.error('JSS WebSocket error', { error });
        webSocketEventBus.emit('connection:error', { name: REGISTRY_NAME, error });
      };

      this.wsConnection.onclose = (event) => {
        logger.info('JSS WebSocket disconnected');
        webSocketRegistry.unregister(REGISTRY_NAME);
        webSocketEventBus.emit('connection:close', {
          name: REGISTRY_NAME,
          code: event.code,
          reason: event.reason,
        });
        if (this.isDisconnecting) {
          this.isDisconnecting = false;
          return;
        }
        this.handleReconnect();
      };
    } catch (error) {
      logger.error('Failed to connect WebSocket', { error });
    }
  }

  /**
   * Subscribe to notifications for a resource
   */
  public subscribe(resourceUrl: string, callback: NotificationCallback): () => void {
    if (!this.subscriptions.has(resourceUrl)) {
      this.subscriptions.set(resourceUrl, new Set());

      // Send subscription if connected
      if (this.wsConnection?.readyState === WebSocket.OPEN) {
        this.wsConnection.send(`sub ${resourceUrl}`);
      }
    }

    this.subscriptions.get(resourceUrl)!.add(callback);

    // Return unsubscribe function
    return () => {
      this.subscriptions.get(resourceUrl)?.delete(callback);

      if (this.subscriptions.get(resourceUrl)?.size === 0) {
        if (this.wsConnection?.readyState === WebSocket.OPEN) {
          this.wsConnection.send(`unsub ${resourceUrl}`);
        }
        this.subscriptions.delete(resourceUrl);
      }
    };
  }

  /**
   * Disconnect WebSocket
   */
  public disconnect(): void {
    if (this.reconnectTimerId !== null) {
      clearTimeout(this.reconnectTimerId);
      this.reconnectTimerId = null;
    }
    this.isDisconnecting = true;
    webSocketRegistry.unregister(REGISTRY_NAME);
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
    this.subscriptions.clear();
  }

  // --- Private Methods ---

  /**
   * Ensure content is in JSON-LD format
   */
  private ensureJsonLd(
    content: JsonLdDocument | Record<string, unknown> | string
  ): JsonLdDocument {
    if (typeof content === 'string') {
      try {
        const parsed = JSON.parse(content);
        return this.ensureJsonLd(parsed);
      } catch {
        // Treat as plain text content
        return {
          '@context': 'https://www.w3.org/ns/ldp',
          '@type': 'Resource',
          content: content,
        };
      }
    }

    // Already has @context - return as-is
    if ('@context' in content && content['@context']) {
      return content as JsonLdDocument;
    }

    // Wrap plain object in JSON-LD structure
    const { '@context': _, '@type': __, ...rest } = content as Record<string, unknown>;
    return {
      '@context': {
        '@vocab': 'https://narrativegoldmine.com/ontology#',
        ldp: 'https://www.w3.org/ns/ldp#',
        xsd: 'http://www.w3.org/2001/XMLSchema#',
      },
      '@type': 'Resource',
      ...rest,
    };
  }

  private async fetchWithAuth(
    url: string,
    options: RequestInit = {}
  ): Promise<Response> {
    const headers = new Headers(options.headers);

    if (nostrAuth.isAuthenticated()) {
      if (nostrAuth.isDevMode()) {
        headers.set('Authorization', 'Bearer dev-session-token');
        const user = nostrAuth.getCurrentUser();
        if (user?.pubkey) headers.set('X-Nostr-Pubkey', user.pubkey);
      } else {
        try {
          const method = (options.method || 'GET').toUpperCase();
          const body = typeof options.body === 'string' ? options.body : undefined;
          const token = await nostrAuth.signRequest(url, method, body);
          headers.set('Authorization', `Nostr ${token}`);
        } catch (e) {
          console.warn('[SolidPodService] NIP-98 signing failed:', e);
        }
      }
    }

    return fetch(url, {
      ...options,
      headers,
      credentials: 'include',
    });
  }

  private resolvePath(path: string): string {
    if (path.startsWith('http://') || path.startsWith('https://')) {
      return path;
    }

    // Remove leading slash if present
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    return `${JSS_BASE_URL}/${cleanPath}`;
  }

  private handleWebSocketMessage(msg: string): void {
    if (msg.startsWith('protocol ')) {
      // Handshake complete, resubscribe to all resources
      logger.debug('WebSocket protocol handshake complete');
      for (const url of this.subscriptions.keys()) {
        this.wsConnection?.send(`sub ${url}`);
      }
    } else if (msg.startsWith('ack ')) {
      const url = msg.slice(4);
      logger.debug('Subscription acknowledged', { url });
      this.notifySubscribers(url, { type: 'ack', url });
    } else if (msg.startsWith('pub ')) {
      const url = msg.slice(4);
      logger.debug('Resource changed', { url });
      this.notifySubscribers(url, { type: 'pub', url });
    }
  }

  private notifySubscribers(url: string, notification: SolidNotification): void {
    // Notify exact URL subscribers
    const callbacks = this.subscriptions.get(url);
    callbacks?.forEach((cb) => cb(notification));

    // Also notify container subscribers (parent directory)
    const containerUrl = url.substring(0, url.lastIndexOf('/') + 1);
    if (containerUrl !== url) {
      const containerCallbacks = this.subscriptions.get(containerUrl);
      containerCallbacks?.forEach((cb) => cb(notification));
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.warn('Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    logger.info(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimerId = setTimeout(() => {
      this.reconnectTimerId = null;
      this.connectWebSocket();
    }, delay);
  }
}

// Export singleton instance
const solidPodService = SolidPodService.getInstance();
export default solidPodService;
