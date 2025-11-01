import { useEffect, useCallback, useState } from 'react'
import AppInitializer from './AppInitializer'
import { ApplicationModeProvider } from '../contexts/ApplicationModeContext';
import { useSettingsStore } from '../store/settingsStore';
import { createLogger } from '../utils/loggerConfig';
import MainLayout from './MainLayout';
import { useQuest3Integration } from '../hooks/useQuest3Integration';
import { ImmersiveApp } from '../immersive/components/ImmersiveApp';
import { BotsDataProvider } from '../features/bots/contexts/BotsDataContext';
import { CommandPalette } from '../features/command-palette/components/CommandPalette';
import { initializeCommandPalette } from '../features/command-palette/defaultCommands';
import { HelpProvider } from '../features/help/components/HelpProvider';
import { registerSettingsHelp } from '../features/help/settingsHelp';
import { OnboardingProvider } from '../features/onboarding/components/OnboardingProvider';
import { registerOnboardingCommands } from '../features/onboarding/flows/defaultFlows';
import { TooltipProvider } from '../features/design-system/components/Tooltip';
import { useBotsWebSocketIntegration } from '../features/bots/hooks/useBotsWebSocketIntegration';
import { DebugControlPanel } from '../components/DebugControlPanel';
import { ConnectionWarning } from '../components/ConnectionWarning';
import { useAutoBalanceNotifications } from '../hooks/useAutoBalanceNotifications';
import ErrorBoundary from '../components/ErrorBoundary';
import { remoteLogger } from '../services/remoteLogger';
import { VircadiaProvider } from '../contexts/VircadiaContext';
import { VircadiaBridgesProvider } from '../contexts/VircadiaBridgesContext';

const logger = createLogger('App');

// Initialize remote logging for Quest 3 debugging
if (typeof window !== 'undefined') {
  console.log('[RemoteLogger] Initializing remote logging service...');
  remoteLogger.logXRInfo(); 
}

function App() {
  const [initializationState, setInitializationState] = useState<'loading' | 'initialized' | 'error'>('loading');
  const [initializationError, setInitializationError] = useState<Error | null>(null);
  const initialized = useSettingsStore(state => state.initialized);

  const { shouldUseQuest3Layout, isQuest3Detected, autoStartSuccessful } = useQuest3Integration({
    enableAutoStart: false
  });

  
  const botsConnectionStatus = useBotsWebSocketIntegration();
  
  
  useAutoBalanceNotifications();

  
  const shouldUseImmersiveClient = () => {
    const userAgent = navigator.userAgent;
    
    const isQuest3Browser = userAgent.includes('Quest 3') ||
                            userAgent.includes('Quest3') ||
                            userAgent.includes('OculusBrowser') ||
                            (userAgent.includes('VR') && userAgent.includes('Quest')) ||
                            userAgent.toLowerCase().includes('meta quest');

    
    const forceQuest3 = window.location.search.includes('force=quest3') ||
                        window.location.search.includes('directar=true') ||
                        window.location.search.includes('immersive=true');

    return (isQuest3Browser || forceQuest3 || shouldUseQuest3Layout) && initialized;
  };

  useEffect(() => {
    
    if (initialized) {
      initializeCommandPalette();
      registerSettingsHelp();
      registerOnboardingCommands();

      const hasVisited = localStorage.getItem('hasVisited');
      if (!hasVisited) {
        localStorage.setItem('hasVisited', 'true');
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('start-onboarding', {
            detail: { flowId: 'welcome' }
          }));
        }, 1000);
      }
    }
  }, [initialized])

  const handleInitialized = useCallback(() => {
    setInitializationState('initialized');
    const settings = useSettingsStore.getState().settings;
    const debugEnabled = settings?.system?.debug?.enabled === true;
    if (debugEnabled) {
      logger.debug('Application initialized');
      logger.debug('Bots WebSocket connection status:', botsConnectionStatus);
    }
  }, [botsConnectionStatus]);

  const handleInitializationError = useCallback((error: Error) => {
    setInitializationError(error);
    setInitializationState('error');
  }, []);

  const renderContent = () => {
    switch (initializationState) {
      case 'loading':
        return <div>Connecting to server...</div>;
      case 'error':
        return (
          <div>
            <h2>Error Initializing Application</h2>
            <p>{initializationError?.message || 'An unknown error occurred.'}</p>
            <button onClick={() => window.location.reload()}>Retry</button>
          </div>
        );
      case 'initialized':
        return shouldUseImmersiveClient() ? (
          <BotsDataProvider>
            <VircadiaBridgesProvider enableBotsBridge={true} enableGraphBridge={true}>
              <ImmersiveApp />
            </VircadiaBridgesProvider>
          </BotsDataProvider>
        ) : (
          <BotsDataProvider>
            <VircadiaBridgesProvider enableBotsBridge={true} enableGraphBridge={false}>
              <MainLayout />
            </VircadiaBridgesProvider>
          </BotsDataProvider>
        );
    }
  };

  return (
    <VircadiaProvider autoConnect={false}>
      <TooltipProvider delayDuration={300} skipDelayDuration={100}>
        <HelpProvider>
          <OnboardingProvider>
            <ErrorBoundary>
              <ApplicationModeProvider>
                {renderContent()}
                {initializationState === 'loading' && (
                  <AppInitializer onInitialized={handleInitialized} onError={handleInitializationError} />
                )}
                {initializationState === 'initialized' && (
                  <>
                    <ConnectionWarning />
                    <CommandPalette />
                    <DebugControlPanel />
                  </>
                )}
              </ApplicationModeProvider>
            </ErrorBoundary>
          </OnboardingProvider>
        </HelpProvider>
      </TooltipProvider>
    </VircadiaProvider>
  );
}

export default App
