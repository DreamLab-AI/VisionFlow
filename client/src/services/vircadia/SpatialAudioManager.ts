/**
 * SpatialAudioManager - 3D spatial audio for multi-user XR
 *
 * Manages WebRTC peer connections and Web Audio API for
 * positional audio in Quest 3 multi-user sessions.
 */

import * as BABYLON from '@babylonjs/core';
import { ClientCore } from './VircadiaClientCore';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('SpatialAudioManager');

export interface SpatialAudioConfig {
    iceServers: RTCIceServer[];
    audioConstraints: MediaStreamConstraints['audio'];
    maxDistance: number;
    rolloffFactor: number;
    refDistance: number;
}

interface PeerConnection {
    agentId: string;
    username: string;
    pc: RTCPeerConnection;
    audioElement?: HTMLAudioElement;
    pannerNode?: PannerNode;
}

export class SpatialAudioManager {
    private audioContext: AudioContext | null = null;
    private localStream: MediaStream | null = null;
    private peerConnections = new Map<string, PeerConnection>();
    private localAgentId: string | null = null;
    private isMuted = false;

    private defaultConfig: SpatialAudioConfig = {
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' }
        ],
        audioConstraints: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 48000
        },
        maxDistance: 20,
        rolloffFactor: 1,
        refDistance: 1
    };

    constructor(
        private client: ClientCore,
        private scene: BABYLON.Scene,
        config?: Partial<SpatialAudioConfig>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
        this.setupConnectionListeners();
    }

    /**
     * Initialize audio context and local stream
     */
    async initialize(): Promise<void> {
        logger.info('Initializing spatial audio...');

        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            logger.info(`Audio context created: ${this.audioContext.state}`);

            // Get local microphone stream
            this.localStream = await navigator.mediaDevices.getUserMedia({
                audio: this.defaultConfig.audioConstraints,
                video: false
            });

            logger.info('Local audio stream acquired');

            // Get local agent ID
            const info = this.client.Utilities.Connection.getConnectionInfo();
            if (info.agentId) {
                this.localAgentId = info.agentId;
            }

        } catch (error) {
            logger.error('Failed to initialize spatial audio:', error);
            throw error;
        }
    }

    /**
     * Set up connection event listeners
     */
    private setupConnectionListeners(): void {
        // Listen for WebRTC signaling messages
        this.client.Utilities.Connection.addEventListener('syncUpdate', async () => {
            await this.handleSignalingMessages();
        });

        // Listen for new users joining
        this.client.Utilities.Connection.addEventListener('statusChange', () => {
            const info = this.client.Utilities.Connection.getConnectionInfo();
            if (info.isConnected && info.agentId) {
                this.localAgentId = info.agentId;
            }
        });
    }

    /**
     * Connect to remote peer
     */
    async connectToPeer(agentId: string, username: string): Promise<void> {
        if (this.peerConnections.has(agentId)) {
            logger.warn(`Already connected to peer: ${agentId}`);
            return;
        }

        if (!this.localStream) {
            logger.error('Cannot connect to peer: local stream not initialized');
            return;
        }

        logger.info(`Connecting to peer: ${username} (${agentId})`);

        try {
            // Create peer connection
            const pc = new RTCPeerConnection({
                iceServers: this.defaultConfig.iceServers
            });

            // Add local stream tracks
            this.localStream.getTracks().forEach(track => {
                pc.addTrack(track, this.localStream!);
            });

            // Handle remote stream
            pc.ontrack = (event) => {
                this.handleRemoteTrack(agentId, username, event);
            };

            // Handle ICE candidates
            pc.onicecandidate = (event) => {
                if (event.candidate) {
                    this.sendICECandidate(agentId, event.candidate);
                }
            };

            // Handle connection state
            pc.onconnectionstatechange = () => {
                logger.debug(`Peer connection state: ${pc.connectionState} (${username})`);
                if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
                    this.removePeer(agentId);
                }
            };

            // Store connection
            const peerConn: PeerConnection = {
                agentId,
                username,
                pc
            };
            this.peerConnections.set(agentId, peerConn);

            // Create and send offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            await this.sendOffer(agentId, offer);

            logger.info(`Offer sent to ${username}`);

        } catch (error) {
            logger.error(`Failed to connect to peer ${username}:`, error);
        }
    }

    /**
     * Handle remote audio track
     */
    private handleRemoteTrack(agentId: string, username: string, event: RTCTrackEvent): void {
        logger.info(`Received remote track from ${username}`);

        const peerConn = this.peerConnections.get(agentId);
        if (!peerConn || !this.audioContext) {
            return;
        }

        // Create audio element for remote stream
        const audioElement = new Audio();
        audioElement.srcObject = event.streams[0];
        audioElement.autoplay = true;
        audioElement.muted = true; // Will be unmuted through Web Audio API

        // Create Web Audio nodes for spatial audio
        const source = this.audioContext.createMediaStreamSource(event.streams[0]);
        const panner = this.audioContext.createPanner();

        // Configure panner for 3D audio
        panner.panningModel = 'HRTF';
        panner.distanceModel = 'inverse';
        panner.refDistance = this.defaultConfig.refDistance;
        panner.maxDistance = this.defaultConfig.maxDistance;
        panner.rolloffFactor = this.defaultConfig.rolloffFactor;

        // Connect nodes
        source.connect(panner);
        panner.connect(this.audioContext.destination);

        // Store references
        peerConn.audioElement = audioElement;
        peerConn.pannerNode = panner;

        logger.info(`Spatial audio configured for ${username}`);
    }

    /**
     * Update listener position (local user camera)
     */
    updateListenerPosition(position: BABYLON.Vector3, forward: BABYLON.Vector3, up: BABYLON.Vector3): void {
        if (!this.audioContext?.listener) {
            return;
        }

        // Update listener position
        if (this.audioContext.listener.positionX) {
            this.audioContext.listener.positionX.value = position.x;
            this.audioContext.listener.positionY.value = position.y;
            this.audioContext.listener.positionZ.value = position.z;
        } else {
            // Fallback for older browsers
            (this.audioContext.listener as any).setPosition(position.x, position.y, position.z);
        }

        // Update listener orientation
        if (this.audioContext.listener.forwardX) {
            this.audioContext.listener.forwardX.value = forward.x;
            this.audioContext.listener.forwardY.value = forward.y;
            this.audioContext.listener.forwardZ.value = forward.z;
            this.audioContext.listener.upX.value = up.x;
            this.audioContext.listener.upY.value = up.y;
            this.audioContext.listener.upZ.value = up.z;
        } else {
            // Fallback for older browsers
            (this.audioContext.listener as any).setOrientation(
                forward.x, forward.y, forward.z,
                up.x, up.y, up.z
            );
        }
    }

    /**
     * Update remote peer audio position
     */
    updatePeerPosition(agentId: string, position: BABYLON.Vector3): void {
        const peerConn = this.peerConnections.get(agentId);
        if (!peerConn?.pannerNode) {
            return;
        }

        // Update panner position
        if (peerConn.pannerNode.positionX) {
            peerConn.pannerNode.positionX.value = position.x;
            peerConn.pannerNode.positionY.value = position.y;
            peerConn.pannerNode.positionZ.value = position.z;
        } else {
            // Fallback for older browsers
            (peerConn.pannerNode as any).setPosition(position.x, position.y, position.z);
        }
    }

    /**
     * Mute/unmute local microphone
     */
    setMuted(muted: boolean): void {
        if (!this.localStream) {
            return;
        }

        this.isMuted = muted;
        this.localStream.getAudioTracks().forEach(track => {
            track.enabled = !muted;
        });

        logger.info(`Microphone ${muted ? 'muted' : 'unmuted'}`);
    }

    /**
     * Toggle microphone mute
     */
    toggleMute(): boolean {
        this.setMuted(!this.isMuted);
        return this.isMuted;
    }

    /**
     * Send WebRTC offer via Vircadia
     */
    private async sendOffer(targetAgentId: string, offer: RTCSessionDescriptionInit): Promise<void> {
        try {
            const query = `
                INSERT INTO entity.entities (
                    general__entity_name,
                    general__semantic_version,
                    group__sync,
                    meta__data
                ) VALUES (
                    'webrtc_offer_${this.localAgentId}_${targetAgentId}',
                    '1.0.0',
                    'public.NORMAL',
                    '${JSON.stringify({
                        type: 'offer',
                        from: this.localAgentId,
                        to: targetAgentId,
                        offer: offer,
                        timestamp: Date.now()
                    })}'::jsonb
                )
            `;

            await this.client.Utilities.Connection.query({ query, timeoutMs: 3000 });

        } catch (error) {
            logger.error('Failed to send offer:', error);
        }
    }

    /**
     * Send ICE candidate via Vircadia
     */
    private async sendICECandidate(targetAgentId: string, candidate: RTCIceCandidate): Promise<void> {
        try {
            const query = `
                INSERT INTO entity.entities (
                    general__entity_name,
                    general__semantic_version,
                    group__sync,
                    meta__data
                ) VALUES (
                    'webrtc_ice_${this.localAgentId}_${targetAgentId}_${Date.now()}',
                    '1.0.0',
                    'public.NORMAL',
                    '${JSON.stringify({
                        type: 'ice-candidate',
                        from: this.localAgentId,
                        to: targetAgentId,
                        candidate: candidate,
                        timestamp: Date.now()
                    })}'::jsonb
                )
            `;

            await this.client.Utilities.Connection.query({ query, timeoutMs: 2000 });

        } catch (error) {
            logger.debug('Failed to send ICE candidate:', error);
        }
    }

    /**
     * Handle incoming signaling messages
     */
    private async handleSignalingMessages(): Promise<void> {
        try {
            const query = `
                SELECT * FROM entity.entities
                WHERE general__entity_name LIKE 'webrtc_%'
                AND meta__data->>'to' = '${this.localAgentId}'
                AND general__created_at > NOW() - INTERVAL '5 seconds'
            `;

            const result = await this.client.Utilities.Connection.query<{ result: any[] }>({
                query,
                timeoutMs: 3000
            });

            if (!result?.result) {
                return;
            }

            const messages = result.result as any[];

            for (const msg of messages) {
                const data = msg.meta__data;
                const fromAgentId = data.from;

                if (data.type === 'offer') {
                    await this.handleOffer(fromAgentId, data.offer);
                } else if (data.type === 'answer') {
                    await this.handleAnswer(fromAgentId, data.answer);
                } else if (data.type === 'ice-candidate') {
                    await this.handleICECandidate(fromAgentId, data.candidate);
                }
            }

        } catch (error) {
            logger.debug('Failed to handle signaling messages:', error);
        }
    }

    /**
     * Handle incoming WebRTC offer
     */
    private async handleOffer(fromAgentId: string, offer: RTCSessionDescriptionInit): Promise<void> {
        logger.info(`Received offer from ${fromAgentId}`);

        const peerConn = this.peerConnections.get(fromAgentId);
        if (!peerConn) {
            logger.warn('No peer connection found for offer');
            return;
        }

        await peerConn.pc.setRemoteDescription(offer);
        const answer = await peerConn.pc.createAnswer();
        await peerConn.pc.setLocalDescription(answer);

        // Send answer back
        await this.sendAnswer(fromAgentId, answer);
    }

    /**
     * Send WebRTC answer
     */
    private async sendAnswer(targetAgentId: string, answer: RTCSessionDescriptionInit): Promise<void> {
        try {
            const query = `
                INSERT INTO entity.entities (
                    general__entity_name,
                    general__semantic_version,
                    group__sync,
                    meta__data
                ) VALUES (
                    'webrtc_answer_${this.localAgentId}_${targetAgentId}',
                    '1.0.0',
                    'public.NORMAL',
                    '${JSON.stringify({
                        type: 'answer',
                        from: this.localAgentId,
                        to: targetAgentId,
                        answer: answer,
                        timestamp: Date.now()
                    })}'::jsonb
                )
            `;

            await this.client.Utilities.Connection.query({ query, timeoutMs: 3000 });

        } catch (error) {
            logger.error('Failed to send answer:', error);
        }
    }

    /**
     * Handle incoming answer
     */
    private async handleAnswer(fromAgentId: string, answer: RTCSessionDescriptionInit): Promise<void> {
        const peerConn = this.peerConnections.get(fromAgentId);
        if (peerConn) {
            await peerConn.pc.setRemoteDescription(answer);
            logger.info(`Answer received from ${fromAgentId}`);
        }
    }

    /**
     * Handle incoming ICE candidate
     */
    private async handleICECandidate(fromAgentId: string, candidate: RTCIceCandidateInit): Promise<void> {
        const peerConn = this.peerConnections.get(fromAgentId);
        if (peerConn) {
            await peerConn.pc.addIceCandidate(new RTCIceCandidate(candidate));
        }
    }

    /**
     * Remove peer connection
     */
    private removePeer(agentId: string): void {
        const peerConn = this.peerConnections.get(agentId);
        if (!peerConn) {
            return;
        }

        logger.info(`Removing peer: ${peerConn.username}`);

        peerConn.pc.close();
        if (peerConn.audioElement) {
            peerConn.audioElement.srcObject = null;
        }

        this.peerConnections.delete(agentId);
    }

    /**
     * Get connected peer count
     */
    getPeerCount(): number {
        return this.peerConnections.size;
    }

    /**
     * Dispose audio manager
     */
    dispose(): void {
        logger.info('Disposing SpatialAudioManager');

        // Stop local stream
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }

        // Close all peer connections
        this.peerConnections.forEach((_, agentId) => {
            this.removePeer(agentId);
        });
        this.peerConnections.clear();

        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}
