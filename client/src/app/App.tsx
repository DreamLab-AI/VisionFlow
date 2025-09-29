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
import { NeuralDashboard } from '../components/NeuralDashboard';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Toaster } from 'react-hot-toast';
const logger = createLogger('App')

// Neural UI Theme
const neuralTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#64ffda',
    },
    secondary: {
      main: '#2196f3',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a2e',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: '#64ffda #0a0a0a',
          '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
            backgroundColor: 'transparent',
          },
          '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
            backgroundColor: '#64ffda',
            borderRadius: 8,
          },
        },
      },
    },
  },
});

function App() {
  const [initializationState, setInitializationState] = useState<'loading' | 'initialized' | 'error'>('loading');
  const [initializationError, setInitializationError] = useState<Error | null>(null);
  const [appMode, setAppMode] = useState<'standard' | 'neural'>('standard');
  const initialized = useSettingsStore(state => state.initialized);

  const { shouldUseQuest3Layout, isQuest3Detected, autoStartSuccessful } = useQuest3Integration({
    enableAutoStart: false
  });

  // Initialize bots WebSocket integration
  const botsConnectionStatus = useBotsWebSocketIntegration();
  
  // Initialize auto-balance notifications polling
  useAutoBalanceNotifications();

  // Check if we should use immersive client (new Babylon.js implementation)
  const shouldUseImmersiveClient = () => {
    const userAgent = navigator.userAgent;
    // Check with original case (Quest 3 browsers use "Quest 3" with capital letters)
    const isQuest3Browser = userAgent.includes('Quest 3') ||
                            userAgent.includes('Quest3') ||
                            userAgent.includes('OculusBrowser') ||
                            (userAgent.includes('VR') && userAgent.includes('Quest')) ||
                            userAgent.toLowerCase().includes('meta quest');

    // Check for force parameter
    const forceQuest3 = window.location.search.includes('force=quest3') ||
                        window.location.search.includes('directar=true') ||
                        window.location.search.includes('immersive=true');

    // Log for debugging
    if (initialized) {
      console.log('Immersive mode check:', {
        userAgent: userAgent.substring(0, 100),
        isQuest3Browser,
        forceQuest3,
        shouldUseQuest3Layout,
        willUseImmersive: isQuest3Browser || forceQuest3 || shouldUseQuest3Layout
      });
    }

    return (isQuest3Browser || forceQuest3 || shouldUseQuest3Layout) && initialized;
  };

  // Check if we should use neural dashboard
  const shouldUseNeuralMode = () => {
    const forceNeural = window.location.search.includes('mode=neural') ||
                        window.location.search.includes('neural=true') ||
                        window.location.pathname.includes('/neural');

    return forceNeural || appMode === 'neural';
  };

  useEffect(() => {
    // Initialize command palette, help system, and onboarding on first load
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

      // Register neural mode command
      window.addEventListener('toggle-neural-mode', () => {
        setAppMode(prev => prev === 'neural' ? 'standard' : 'neural');
      });
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
        if (shouldUseNeuralMode()) {
          return (
            <ThemeProvider theme={neuralTheme}>
              <CssBaseline />
              <NeuralDashboard />
              <Toaster
                position="top-right"
                toastOptions={{
                  style: {
                    background: '#1a1a2e',
                    color: '#ffffff',
                    border: '1px solid #64ffda',
                  },
                }}
              />
            </ThemeProvider>
          );
        } else if (shouldUseImmersiveClient()) {
          return (
            <BotsDataProvider>
              <ImmersiveApp />
            </BotsDataProvider>
          );
        } else {
          return <MainLayout />;
        }
    }
  };

  return (
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
  );
}

export default App
