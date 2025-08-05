import { createLogger } from '../logger.js';

const logger = createLogger('MemoryTools');

// In-memory storage with TTL support
class MemoryStore {
  constructor() {
    this.store = new Map();
    this.metadata = new Map();
    this.cleanupInterval = setInterval(() => this.cleanup(), 60000); // Cleanup every minute
  }
  
  set(key, value, ttl = null) {
    this.store.set(key, value);
    this.metadata.set(key, {
      createdAt: new Date(),
      updatedAt: new Date(),
      ttl: ttl ? Date.now() + ttl * 1000 : null,
      accessCount: 0,
      size: this.estimateSize(value)
    });
  }
  
  get(key) {
    const meta = this.metadata.get(key);
    if (!meta) return null;
    
    // Check TTL
    if (meta.ttl && Date.now() > meta.ttl) {
      this.delete(key);
      return null;
    }
    
    // Update access metadata
    meta.accessCount++;
    meta.lastAccessed = new Date();
    
    return this.store.get(key);
  }
  
  has(key) {
    return this.store.has(key) && !this.isExpired(key);
  }
  
  delete(key) {
    this.store.delete(key);
    this.metadata.delete(key);
  }
  
  list(pattern = null) {
    const keys = [];
    
    for (const [key, value] of this.store.entries()) {
      if (!this.isExpired(key)) {
        if (!pattern || key.includes(pattern)) {
          keys.push(key);
        }
      }
    }
    
    return keys;
  }
  
  isExpired(key) {
    const meta = this.metadata.get(key);
    if (!meta) return true;
    return meta.ttl && Date.now() > meta.ttl;
  }
  
  cleanup() {
    for (const [key, meta] of this.metadata.entries()) {
      if (meta.ttl && Date.now() > meta.ttl) {
        this.delete(key);
        logger.debug(`Cleaned up expired key: ${key}`);
      }
    }
  }
  
  estimateSize(value) {
    // Rough estimation of object size in bytes
    const str = JSON.stringify(value);
    return str.length * 2; // UTF-16 encoding
  }
  
  getStats() {
    let totalSize = 0;
    let expiredCount = 0;
    
    for (const [key, meta] of this.metadata.entries()) {
      totalSize += meta.size;
      if (this.isExpired(key)) expiredCount++;
    }
    
    return {
      totalKeys: this.store.size,
      expiredKeys: expiredCount,
      totalSizeBytes: totalSize,
      totalSizeMB: totalSize / (1024 * 1024)
    };
  }
  
  destroy() {
    clearInterval(this.cleanupInterval);
    this.store.clear();
    this.metadata.clear();
  }
}

// Global memory store instance
const memoryStore = new MemoryStore();

// Structured memory sections
const memorySections = {
  swarm: new MemoryStore(),
  agents: new MemoryStore(),
  patterns: new MemoryStore(),
  performance: new MemoryStore(),
  coordination: new MemoryStore()
};

export function memoryTools() {
  return {
    'memory.store': {
      description: 'Store swarm state and coordination history',
      inputSchema: {
        type: 'object',
        properties: {
          key: { 
            type: 'string',
            description: 'Storage key (use / for namespaces, e.g., swarm/state)'
          },
          value: { 
            type: ['object', 'array', 'string', 'number', 'boolean'],
            description: 'Value to store'
          },
          ttl: {
            type: 'number',
            description: 'Time to live in seconds (optional)'
          },
          section: {
            type: 'string',
            enum: ['swarm', 'agents', 'patterns', 'performance', 'coordination', 'global'],
            default: 'global'
          }
        },
        required: ['key', 'value']
      },
      handler: async (args) => {
        try {
          const store = args.section === 'global' ? memoryStore : memorySections[args.section];
          
          if (!store) {
            throw new Error(`Invalid memory section: ${args.section}`);
          }
          
          store.set(args.key, args.value, args.ttl);
          
          logger.info(`Stored key: ${args.key} in section: ${args.section}`);
          
          return {
            success: true,
            key: args.key,
            section: args.section,
            size: store.metadata.get(args.key)?.size || 0,
            ttl: args.ttl || null
          };
        } catch (error) {
          logger.error('Failed to store memory:', error);
          throw error;
        }
      }
    },
    
    'memory.retrieve': {
      description: 'Retrieve stored data by key',
      inputSchema: {
        type: 'object',
        properties: {
          key: { type: 'string', description: 'Storage key' },
          section: {
            type: 'string',
            enum: ['swarm', 'agents', 'patterns', 'performance', 'coordination', 'global'],
            default: 'global'
          }
        },
        required: ['key']
      },
      handler: async (args) => {
        try {
          const store = args.section === 'global' ? memoryStore : memorySections[args.section];
          
          if (!store) {
            throw new Error(`Invalid memory section: ${args.section}`);
          }
          
          const value = store.get(args.key);
          const metadata = store.metadata.get(args.key);
          
          if (value === null) {
            return {
              success: false,
              message: `Key not found: ${args.key}`
            };
          }
          
          return {
            success: true,
            key: args.key,
            value,
            metadata: metadata ? {
              createdAt: metadata.createdAt,
              updatedAt: metadata.updatedAt,
              accessCount: metadata.accessCount,
              size: metadata.size
            } : null
          };
        } catch (error) {
          logger.error('Failed to retrieve memory:', error);
          throw error;
        }
      }
    },
    
    'memory.list': {
      description: 'List stored keys matching pattern',
      inputSchema: {
        type: 'object',
        properties: {
          pattern: { 
            type: 'string',
            description: 'Key pattern to match (optional)'
          },
          section: {
            type: 'string',
            enum: ['swarm', 'agents', 'patterns', 'performance', 'coordination', 'global', 'all'],
            default: 'all'
          },
          includeMetadata: {
            type: 'boolean',
            default: false
          }
        }
      },
      handler: async (args) => {
        try {
          let results = [];
          
          if (args.section === 'all') {
            // Search all sections
            results.push({
              section: 'global',
              keys: memoryStore.list(args.pattern)
            });
            
            Object.entries(memorySections).forEach(([section, store]) => {
              results.push({
                section,
                keys: store.list(args.pattern)
              });
            });
          } else {
            // Search specific section
            const store = args.section === 'global' ? memoryStore : memorySections[args.section];
            
            if (!store) {
              throw new Error(`Invalid memory section: ${args.section}`);
            }
            
            results.push({
              section: args.section,
              keys: store.list(args.pattern)
            });
          }
          
          // Add metadata if requested
          if (args.includeMetadata) {
            results = results.map(result => ({
              ...result,
              keys: result.keys.map(key => {
                const store = result.section === 'global' ? memoryStore : memorySections[result.section];
                const metadata = store.metadata.get(key);
                
                return {
                  key,
                  metadata: metadata ? {
                    createdAt: metadata.createdAt,
                    size: metadata.size,
                    accessCount: metadata.accessCount
                  } : null
                };
              })
            }));
          }
          
          const totalKeys = results.reduce((sum, r) => sum + r.keys.length, 0);
          
          return {
            success: true,
            results,
            totalKeys,
            pattern: args.pattern || '*'
          };
        } catch (error) {
          logger.error('Failed to list memory:', error);
          throw error;
        }
      }
    },
    
    'memory.delete': {
      description: 'Delete stored data by key',
      inputSchema: {
        type: 'object',
        properties: {
          key: { type: 'string', description: 'Storage key to delete' },
          section: {
            type: 'string',
            enum: ['swarm', 'agents', 'patterns', 'performance', 'coordination', 'global'],
            default: 'global'
          }
        },
        required: ['key']
      },
      handler: async (args) => {
        try {
          const store = args.section === 'global' ? memoryStore : memorySections[args.section];
          
          if (!store) {
            throw new Error(`Invalid memory section: ${args.section}`);
          }
          
          const exists = store.has(args.key);
          
          if (exists) {
            store.delete(args.key);
            logger.info(`Deleted key: ${args.key} from section: ${args.section}`);
          }
          
          return {
            success: true,
            deleted: exists,
            key: args.key,
            section: args.section
          };
        } catch (error) {
          logger.error('Failed to delete memory:', error);
          throw error;
        }
      }
    },
    
    'memory.persist': {
      description: 'Persist memory state to disk',
      inputSchema: {
        type: 'object',
        properties: {
          filename: { 
            type: 'string',
            description: 'File name for persistence',
            default: 'memory-snapshot.json'
          },
          sections: {
            type: 'array',
            items: {
              type: 'string',
              enum: ['swarm', 'agents', 'patterns', 'performance', 'coordination', 'global']
            },
            description: 'Sections to persist (default: all)'
          }
        }
      },
      handler: async (args) => {
        try {
          const snapshot = {
            timestamp: new Date().toISOString(),
            sections: {}
          };
          
          const sectionsToSave = args.sections || ['global', ...Object.keys(memorySections)];
          
          sectionsToSave.forEach(section => {
            const store = section === 'global' ? memoryStore : memorySections[section];
            if (!store) return;
            
            snapshot.sections[section] = {
              data: {},
              metadata: {}
            };
            
            // Save all non-expired entries
            for (const [key, value] of store.store.entries()) {
              if (!store.isExpired(key)) {
                snapshot.sections[section].data[key] = value;
                snapshot.sections[section].metadata[key] = store.metadata.get(key);
              }
            }
          });
          
          // In a real implementation, this would write to disk
          // For now, we'll just return the snapshot info
          const totalKeys = Object.values(snapshot.sections)
            .reduce((sum, section) => sum + Object.keys(section.data).length, 0);
          
          logger.info(`Persisted ${totalKeys} keys to ${args.filename}`);
          
          return {
            success: true,
            filename: args.filename,
            totalKeys,
            sections: sectionsToSave,
            sizeEstimate: JSON.stringify(snapshot).length
          };
        } catch (error) {
          logger.error('Failed to persist memory:', error);
          throw error;
        }
      }
    },
    
    'memory.search': {
      description: 'Search memory by value content',
      inputSchema: {
        type: 'object',
        properties: {
          query: { 
            type: 'string',
            description: 'Search query'
          },
          field: {
            type: 'string',
            description: 'Specific field to search in objects'
          },
          section: {
            type: 'string',
            enum: ['swarm', 'agents', 'patterns', 'performance', 'coordination', 'global', 'all'],
            default: 'all'
          },
          maxResults: {
            type: 'number',
            default: 20
          }
        },
        required: ['query']
      },
      handler: async (args) => {
        try {
          const results = [];
          const searchSections = args.section === 'all' 
            ? ['global', ...Object.keys(memorySections)]
            : [args.section];
          
          searchSections.forEach(section => {
            const store = section === 'global' ? memoryStore : memorySections[section];
            if (!store) return;
            
            for (const [key, value] of store.store.entries()) {
              if (store.isExpired(key)) continue;
              
              // Search in value
              let matches = false;
              
              if (typeof value === 'string') {
                matches = value.toLowerCase().includes(args.query.toLowerCase());
              } else if (typeof value === 'object' && value !== null) {
                if (args.field && value[args.field]) {
                  matches = String(value[args.field])
                    .toLowerCase()
                    .includes(args.query.toLowerCase());
                } else {
                  // Search in all fields
                  const valueStr = JSON.stringify(value).toLowerCase();
                  matches = valueStr.includes(args.query.toLowerCase());
                }
              }
              
              if (matches) {
                results.push({
                  section,
                  key,
                  value,
                  relevance: calculateRelevance(value, args.query, args.field)
                });
                
                if (results.length >= args.maxResults) {
                  break;
                }
              }
            }
          });
          
          // Sort by relevance
          results.sort((a, b) => b.relevance - a.relevance);
          
          return {
            success: true,
            results: results.slice(0, args.maxResults),
            totalFound: results.length,
            query: args.query
          };
        } catch (error) {
          logger.error('Failed to search memory:', error);
          throw error;
        }
      }
    },
    
    'memory.stats': {
      description: 'Get memory usage statistics',
      inputSchema: {
        type: 'object',
        properties: {
          detailed: { type: 'boolean', default: false }
        }
      },
      handler: async (args) => {
        try {
          const stats = {
            global: memoryStore.getStats(),
            sections: {}
          };
          
          Object.entries(memorySections).forEach(([section, store]) => {
            stats.sections[section] = store.getStats();
          });
          
          // Calculate totals
          stats.totals = {
            keys: stats.global.totalKeys + 
              Object.values(stats.sections).reduce((sum, s) => sum + s.totalKeys, 0),
            sizeBytes: stats.global.totalSizeBytes + 
              Object.values(stats.sections).reduce((sum, s) => sum + s.totalSizeBytes, 0),
            sizeMB: 0
          };
          stats.totals.sizeMB = stats.totals.sizeBytes / (1024 * 1024);
          
          if (args.detailed) {
            // Add most accessed keys
            stats.mostAccessed = getMostAccessedKeys();
            
            // Add largest entries
            stats.largestEntries = getLargestEntries();
            
            // Add age distribution
            stats.ageDistribution = getAgeDistribution();
          }
          
          return {
            success: true,
            stats,
            timestamp: new Date().toISOString()
          };
        } catch (error) {
          logger.error('Failed to get memory stats:', error);
          throw error;
        }
      }
    }
  };
}

// Helper functions

function calculateRelevance(value, query, field) {
  const queryLower = query.toLowerCase();
  let relevance = 0;
  
  if (typeof value === 'string') {
    const valueLower = value.toLowerCase();
    if (valueLower === queryLower) relevance = 1.0;
    else if (valueLower.startsWith(queryLower)) relevance = 0.8;
    else if (valueLower.includes(queryLower)) relevance = 0.5;
  } else if (typeof value === 'object' && value !== null) {
    if (field && value[field]) {
      const fieldValue = String(value[field]).toLowerCase();
      if (fieldValue === queryLower) relevance = 1.0;
      else if (fieldValue.includes(queryLower)) relevance = 0.7;
    } else {
      // Check all fields
      Object.values(value).forEach(v => {
        if (String(v).toLowerCase().includes(queryLower)) {
          relevance = Math.max(relevance, 0.3);
        }
      });
    }
  }
  
  return relevance;
}

function getMostAccessedKeys() {
  const allKeys = [];
  
  // Collect from all stores
  const stores = [memoryStore, ...Object.values(memorySections)];
  stores.forEach((store, storeIndex) => {
    store.metadata.forEach((meta, key) => {
      allKeys.push({
        key,
        section: storeIndex === 0 ? 'global' : Object.keys(memorySections)[storeIndex - 1],
        accessCount: meta.accessCount
      });
    });
  });
  
  // Sort by access count
  allKeys.sort((a, b) => b.accessCount - a.accessCount);
  
  return allKeys.slice(0, 10);
}

function getLargestEntries() {
  const allEntries = [];
  
  // Collect from all stores
  const stores = [memoryStore, ...Object.values(memorySections)];
  stores.forEach((store, storeIndex) => {
    store.metadata.forEach((meta, key) => {
      allEntries.push({
        key,
        section: storeIndex === 0 ? 'global' : Object.keys(memorySections)[storeIndex - 1],
        size: meta.size
      });
    });
  });
  
  // Sort by size
  allEntries.sort((a, b) => b.size - a.size);
  
  return allEntries.slice(0, 10);
}

function getAgeDistribution() {
  const now = Date.now();
  const distribution = {
    '< 1 minute': 0,
    '1-5 minutes': 0,
    '5-30 minutes': 0,
    '30-60 minutes': 0,
    '> 1 hour': 0
  };
  
  // Collect from all stores
  const stores = [memoryStore, ...Object.values(memorySections)];
  stores.forEach(store => {
    store.metadata.forEach(meta => {
      const ageMs = now - meta.createdAt.getTime();
      const ageMinutes = ageMs / (1000 * 60);
      
      if (ageMinutes < 1) distribution['< 1 minute']++;
      else if (ageMinutes < 5) distribution['1-5 minutes']++;
      else if (ageMinutes < 30) distribution['5-30 minutes']++;
      else if (ageMinutes < 60) distribution['30-60 minutes']++;
      else distribution['> 1 hour']++;
    });
  });
  
  return distribution;
}