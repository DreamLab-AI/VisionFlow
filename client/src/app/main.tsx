import { extend } from '@react-three/fiber';
import { GeodesicPolyhedronGeometry } from '../utils/three-geometries';

extend({ GeodesicPolyhedronGeometry });
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { initializeDebugSystem } from '../utils/debugConfig';
import { unifiedApiClient } from '../services/api/UnifiedApiClient';
import { initializeAuthInterceptor, setupAuthStateListener } from '../services/api/authInterceptor';
import '../styles/index.css'; // Use relative path


// Initialize debug system before app starts
initializeDebugSystem();

// Initialize authentication interceptor for all API calls
initializeAuthInterceptor(unifiedApiClient);
setupAuthStateListener();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
