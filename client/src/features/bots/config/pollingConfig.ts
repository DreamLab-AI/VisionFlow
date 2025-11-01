import type { PollingConfig } from '../services/AgentPollingService';


export const POLLING_PRESETS = {
  
  realtime: {
    activePollingInterval: 500,   
    idlePollingInterval: 2000,    
    enableSmartPolling: true,
    maxRetries: 5,
    retryDelay: 1000
  } as PollingConfig,

  
  standard: {
    activePollingInterval: 1000,  
    idlePollingInterval: 5000,    
    enableSmartPolling: true,
    maxRetries: 3,
    retryDelay: 2000
  } as PollingConfig,

  
  performance: {
    activePollingInterval: 2000,  
    idlePollingInterval: 10000,   
    enableSmartPolling: true,
    maxRetries: 3,
    retryDelay: 3000
  } as PollingConfig,

  
  debug: {
    activePollingInterval: 250,   
    idlePollingInterval: 1000,    
    enableSmartPolling: false,    
    maxRetries: 10,
    retryDelay: 500
  } as PollingConfig
};


export const ACTIVITY_THRESHOLDS = {
  
  activeAgentThreshold: 0.2,      
  
  
  pendingTaskThreshold: 1,        
  
  
  idleTimeThreshold: 30000,       
  
  
  activityDebounceTime: 5000      
};