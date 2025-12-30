

import { ClientCore } from './VircadiaClientCore';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('NetworkOptimizer');

export interface NetworkOptimizerConfig {
    batchIntervalMs: number;
    maxBatchSize: number;
    compressionEnabled: boolean;
    adaptiveQuality: boolean;
    bandwidthTargetKbps: number;
}

export interface PositionUpdate {
    entityName: string;
    x: number;
    y: number;
    z: number;
    timestamp: number;
}

export interface DeltaCompressedUpdate {
    entityName: string;
    dx: number;
    dy: number;
    dz: number;
    timestamp: number;
}

export interface NetworkStats {
    bytesSent: number;
    bytesReceived: number;
    messagesSent: number;
    messagesReceived: number;
    compressionRatio: number;
    averageLatency: number;
    currentBandwidthKbps: number;
}

export class NetworkOptimizer {
    private pendingUpdates = new Map<string, PositionUpdate>();
    private lastPositions = new Map<string, { x: number; y: number; z: number }>();
    private batchInterval: ReturnType<typeof setInterval> | null = null;
    private stats: NetworkStats = {
        bytesSent: 0,
        bytesReceived: 0,
        messagesSent: 0,
        messagesReceived: 0,
        compressionRatio: 1.0,
        averageLatency: 0,
        currentBandwidthKbps: 0
    };
    private latencyMeasurements: number[] = [];
    private lastBatchTime = Date.now();

    private defaultConfig: NetworkOptimizerConfig = {
        batchIntervalMs: 100, 
        maxBatchSize: 100,
        compressionEnabled: true,
        adaptiveQuality: true,
        bandwidthTargetKbps: 5000 
    };

    constructor(
        private client: ClientCore,
        config?: Partial<NetworkOptimizerConfig>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
        this.startBatchInterval();
    }

    
    queuePositionUpdate(entityName: string, x: number, y: number, z: number): void {
        this.pendingUpdates.set(entityName, {
            entityName,
            x,
            y,
            z,
            timestamp: Date.now()
        });
    }

    
    private compressPositionUpdate(update: PositionUpdate): DeltaCompressedUpdate {
        const lastPos = this.lastPositions.get(update.entityName);

        if (!lastPos) {
            
            this.lastPositions.set(update.entityName, {
                x: update.x,
                y: update.y,
                z: update.z
            });

            return {
                entityName: update.entityName,
                dx: update.x,
                dy: update.y,
                dz: update.z,
                timestamp: update.timestamp
            };
        }

        
        const dx = update.x - lastPos.x;
        const dy = update.y - lastPos.y;
        const dz = update.z - lastPos.z;

        
        this.lastPositions.set(update.entityName, {
            x: update.x,
            y: update.y,
            z: update.z
        });

        return {
            entityName: update.entityName,
            dx,
            dy,
            dz,
            timestamp: update.timestamp
        };
    }

    
    private encodePositionsToBinary(updates: DeltaCompressedUpdate[]): ArrayBuffer {
        
        

        let totalSize = 4; 
        const textEncoder = new TextEncoder();
        const encodedNames: Uint8Array[] = [];

        
        updates.forEach(update => {
            const encoded = textEncoder.encode(update.entityName);
            encodedNames.push(encoded);
            totalSize += 1 + encoded.length + 12 + 4; 
        });

        const buffer = new ArrayBuffer(totalSize);
        const view = new DataView(buffer);
        let offset = 0;

        
        view.setUint32(offset, updates.length, true);
        offset += 4;

        
        updates.forEach((update, i) => {
            const nameBytes = encodedNames[i];

            
            view.setUint8(offset, nameBytes.length);
            offset += 1;

            
            new Uint8Array(buffer, offset, nameBytes.length).set(nameBytes);
            offset += nameBytes.length;

            
            view.setFloat32(offset, update.dx, true);
            offset += 4;
            view.setFloat32(offset, update.dy, true);
            offset += 4;
            view.setFloat32(offset, update.dz, true);
            offset += 4;

            
            view.setUint32(offset, update.timestamp, true);
            offset += 4;
        });

        return buffer;
    }

    
    decodePositionsFromBinary(buffer: ArrayBuffer): DeltaCompressedUpdate[] {
        const view = new DataView(buffer);
        const textDecoder = new TextDecoder();
        let offset = 0;

        
        const count = view.getUint32(offset, true);
        offset += 4;

        const updates: DeltaCompressedUpdate[] = [];

        
        for (let i = 0; i < count; i++) {
            
            const nameLength = view.getUint8(offset);
            offset += 1;

            
            const nameBytes = new Uint8Array(buffer, offset, nameLength);
            const entityName = textDecoder.decode(nameBytes);
            offset += nameLength;

            
            const dx = view.getFloat32(offset, true);
            offset += 4;
            const dy = view.getFloat32(offset, true);
            offset += 4;
            const dz = view.getFloat32(offset, true);
            offset += 4;

            
            const timestamp = view.getUint32(offset, true);
            offset += 4;

            updates.push({ entityName, dx, dy, dz, timestamp });
        }

        return updates;
    }

    
    private async flushBatch(): Promise<void> {
        if (this.pendingUpdates.size === 0) {
            return;
        }

        const startTime = Date.now();
        const updates = Array.from(this.pendingUpdates.values());
        this.pendingUpdates.clear();

        try {
            if (this.defaultConfig.compressionEnabled) {
                
                const compressed = updates.map(u => this.compressPositionUpdate(u));

                
                const binary = this.encodePositionsToBinary(compressed);

                
                const base64 = btoa(String.fromCharCode(...new Uint8Array(binary)));

                
                const query = `
                    INSERT INTO entity.entities (
                        general__entity_name,
                        general__semantic_version,
                        group__sync,
                        meta__data
                    ) VALUES (
                        'batch_update_${Date.now()}',
                        '1.0.0',
                        'public.NORMAL',
                        '${JSON.stringify({
                            type: 'batch_position_update',
                            binary: base64,
                            count: compressed.length,
                            timestamp: Date.now()
                        })}'::jsonb
                    )
                `;

                await this.client.Utilities.Connection.query({ query, timeoutMs: 2000 });

                
                const originalSize = updates.length * 32; 
                const compressedSize = binary.byteLength;
                this.stats.compressionRatio = originalSize / compressedSize;
                this.stats.bytesSent += compressedSize;

            } else {
                
                const statements = updates.map(update => {
                    return `
                        UPDATE entity.entities
                        SET meta__data = jsonb_set(
                            jsonb_set(
                                jsonb_set(
                                    meta__data,
                                    '{position,x}', '${update.x}'::text::jsonb
                                ),
                                '{position,y}', '${update.y}'::text::jsonb
                            ),
                            '{position,z}', '${update.z}'::text::jsonb
                        )
                        WHERE general__entity_name = '${update.entityName}'
                    `;
                });

                const batchQuery = statements.join('\n');
                await this.client.Utilities.Connection.query({ query: batchQuery, timeoutMs: 3000 });

                this.stats.bytesSent += batchQuery.length;
            }

            const latency = Date.now() - startTime;
            this.latencyMeasurements.push(latency);
            if (this.latencyMeasurements.length > 10) {
                this.latencyMeasurements.shift();
            }

            this.stats.averageLatency = this.latencyMeasurements.reduce((a, b) => a + b, 0) / this.latencyMeasurements.length;
            this.stats.messagesSent++;

            
            const timeDelta = (Date.now() - this.lastBatchTime) / 1000; 
            if (timeDelta > 0) {
                const bytesPerSecond = this.stats.bytesSent / timeDelta;
                this.stats.currentBandwidthKbps = (bytesPerSecond * 8) / 1000;
            }
            this.lastBatchTime = Date.now();

            
            if (this.defaultConfig.adaptiveQuality) {
                this.adjustQuality();
            }

            logger.debug(`Batch flushed: ${updates.length} updates, ${latency}ms latency, ${this.stats.compressionRatio.toFixed(2)}x compression`);

        } catch (error) {
            logger.error('Failed to flush batch:', error);
        }
    }

    
    private adjustQuality(): void {
        const { bandwidthTargetKbps } = this.defaultConfig;

        if (this.stats.currentBandwidthKbps > bandwidthTargetKbps * 1.2) {
            
            if (this.defaultConfig.batchIntervalMs < 500) {
                this.defaultConfig.batchIntervalMs += 50;
                this.restartBatchInterval();
                logger.info(`Reduced update rate: ${this.defaultConfig.batchIntervalMs}ms interval`);
            }
        } else if (this.stats.currentBandwidthKbps < bandwidthTargetKbps * 0.5) {
            
            if (this.defaultConfig.batchIntervalMs > 50) {
                this.defaultConfig.batchIntervalMs -= 25;
                this.restartBatchInterval();
                logger.info(`Increased update rate: ${this.defaultConfig.batchIntervalMs}ms interval`);
            }
        }
    }

    
    private startBatchInterval(): void {
        if (this.batchInterval) {
            return;
        }

        this.batchInterval = setInterval(() => {
            this.flushBatch().catch(err => {
                logger.error('Batch flush error:', err);
            });
        }, this.defaultConfig.batchIntervalMs);

        logger.info(`Batch interval started: ${this.defaultConfig.batchIntervalMs}ms`);
    }

    
    private restartBatchInterval(): void {
        this.stopBatchInterval();
        this.startBatchInterval();
    }

    
    private stopBatchInterval(): void {
        if (this.batchInterval) {
            clearInterval(this.batchInterval);
            this.batchInterval = null;
        }
    }

    
    getStats(): NetworkStats {
        return { ...this.stats };
    }

    
    resetStats(): void {
        this.stats = {
            bytesSent: 0,
            bytesReceived: 0,
            messagesSent: 0,
            messagesReceived: 0,
            compressionRatio: 1.0,
            averageLatency: 0,
            currentBandwidthKbps: 0
        };
        this.latencyMeasurements = [];
        this.lastBatchTime = Date.now();
    }

    
    dispose(): void {
        logger.info('Disposing NetworkOptimizer');
        this.stopBatchInterval();
        this.pendingUpdates.clear();
        this.lastPositions.clear();
    }
}
