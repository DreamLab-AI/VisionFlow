import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { initializeDebugSystem } from '../utils/debugConfig';
import '../styles/index.css'; // Use relative path

// Clear bad physics from localStorage BEFORE anything else
// This ensures server settings are used as source of truth
if (typeof window !== 'undefined' && window.localStorage) {
  try {
    // The storage key is 'graph-viz-settings' as defined in the zustand persist config
    const stored = localStorage.getItem('graph-viz-settings');
    if (stored) {
      const parsed = JSON.parse(stored);
      // Clear physics settings to force reload from server
      if (parsed?.state?.settings?.visualisation?.graphs) {
        delete parsed.state.settings.visualisation.graphs.logseq?.physics;
        delete parsed.state.settings.visualisation.graphs.visionflow?.physics;
        localStorage.setItem('graph-viz-settings', JSON.stringify(parsed));
        console.log('[Settings] Cleared cached physics from localStorage to use server values');
      }
    }
  } catch (e) {
    console.warn('Could not clear cached physics settings:', e);
  }
}

// Initialize debug system before app starts
initializeDebugSystem();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
