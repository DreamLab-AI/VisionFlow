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

const logger = createLogger('SolidPodService');

// --- Interfaces ---

export interface PodInfo {
  exists: boolean;
  podUrl?: string;
  webId?: string;
  suggestedUrl?: string;
}

export interface PodCreationResult {
  success: boolean;
  podUrl?: string;
  webId?: string;
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

class SolidPodService {
  private static instance: SolidPodService;
  private wsConnection: WebSocket | null = null;
  private subscriptions: Map<string, Set<NotificationCallback>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

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

  // --- LDP Operations ---

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
      this.wsConnection = new WebSocket(JSS_WS_URL);

      this.wsConnection.onopen = () => {
        logger.info('JSS WebSocket connected');
        this.reconnectAttempts = 0;
        // Protocol handshake will be handled in onmessage
      };

      this.wsConnection.onmessage = (event) => {
        const msg = event.data.toString().trim();
        this.handleWebSocketMessage(msg);
      };

      this.wsConnection.onerror = (error) => {
        logger.error('JSS WebSocket error', { error });
      };

      this.wsConnection.onclose = () => {
        logger.info('JSS WebSocket disconnected');
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
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
    this.subscriptions.clear();
  }

  // --- Private Methods ---

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

    setTimeout(() => {
      this.connectWebSocket();
    }, delay);
  }
}

// Export singleton instance
const solidPodService = SolidPodService.getInstance();
export default solidPodService;
