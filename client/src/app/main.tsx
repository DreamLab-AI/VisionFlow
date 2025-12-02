import { extend } from '@react-three/fiber';
import { GeodesicPolyhedronGeometry } from '../utils/three-geometries';

extend({ GeodesicPolyhedronGeometry });
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { initializeDebugSystem } from '../utils/debugConfig';
import { unifiedApiClient } from '../services/api/UnifiedApiClient';
import { initializeAuthInterceptor, setupAuthStateListener } from '../services/api/authInterceptor';
import { useSettingsStore } from '../store/settingsStore';
import { webSocketService } from '../services/WebSocketService';
import '../styles/index.css';


// Initialize debug system before app starts
initializeDebugSystem();

// Expose stores and services for testing/debugging (dev mode only)
if (import.meta.env.DEV) {
  (window as any).useSettingsStore = useSettingsStore;
  (window as any).webSocketService = webSocketService;
  console.log('[Dev] Exposed useSettingsStore and webSocketService on window for testing');
}

// Initialize authentication interceptor for all API calls
initializeAuthInterceptor(unifiedApiClient);
setupAuthStateListener();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
