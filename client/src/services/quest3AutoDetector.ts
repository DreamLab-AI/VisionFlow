import { createLogger } from '../utils/loggerConfig';
import { usePlatformStore } from './platformManager';
import { useSettingsStore } from '../store/settingsStore';
import { ClientCore } from './vircadia/VircadiaClientCore';

const logger = createLogger('Quest3AutoDetector');

export interface Quest3DetectionResult {
  isQuest3: boolean;
  isQuest3Browser: boolean;
  supportsAR: boolean;
  shouldAutoStart: boolean;
}

export class Quest3AutoDetector {
  private static instance: Quest3AutoDetector;
  private detectionResult: Quest3DetectionResult | null = null;
  private autoStartAttempted: boolean = false;
  private vircadiaClient: ClientCore | null = null;

  private constructor() {}

  public static getInstance(): Quest3AutoDetector {
    if (!Quest3AutoDetector.instance) {
      Quest3AutoDetector.instance = new Quest3AutoDetector();
    }
    return Quest3AutoDetector.instance;
  }

  
  public async detectQuest3Environment(): Promise<Quest3DetectionResult> {
    if (this.detectionResult) {
      return this.detectionResult;
    }

    const userAgent = navigator.userAgent || '';
    const platform = usePlatformStore.getState();

    
    const isQuest3Hardware = userAgent.includes('Quest 3') ||
                            userAgent.includes('Quest3') ||
                            userAgent.includes('Quest 3');

    
    const isQuest3Browser = isQuest3Hardware ||
      userAgent.includes('OculusBrowser') ||
      userAgent.includes('Quest') ||
      (userAgent.includes('Mobile') && userAgent.includes('VR')) ||
      (userAgent.includes('X11') && userAgent.includes('Linux') && userAgent.includes('VR'));

    
    let supportsAR = false;
    try {
      if ('xr' in navigator && navigator.xr) {
        supportsAR = await navigator.xr.isSessionSupported('immersive-ar');
      }
    } catch (error) {
      logger.warn('Error checking AR support:', error);
      supportsAR = false;
    }

    
    const shouldAutoStart = isQuest3Hardware && isQuest3Browser && supportsAR;

    this.detectionResult = {
      isQuest3: isQuest3Hardware,
      isQuest3Browser,
      supportsAR,
      shouldAutoStart
    };

    logger.info('Quest 3 Detection Results:', {
      userAgent: userAgent.substring(0, 100) + '...',
      isQuest3Hardware,
      isQuest3Browser,
      supportsAR,
      shouldAutoStart,
      platformDetected: platform.platform
    });

    return this.detectionResult;
  }

  
  public async autoStartQuest3AR(): Promise<boolean> {
    if (this.autoStartAttempted) {
      logger.info('Quest 3 AR auto-start already attempted');
      return false;
    }

    this.autoStartAttempted = true;

    try {
      const detection = await this.detectQuest3Environment();

      if (!detection.shouldAutoStart) {
        logger.info('Quest 3 AR auto-start conditions not met:', detection);
        return false;
      }

      logger.info('Attempting Quest 3 AR auto-start...');

      
      await this.configureQuest3Settings();

      
      const sessionInit: XRSessionInit = {
        requiredFeatures: ['local-floor'],
        optionalFeatures: [
          'hand-tracking',
          'hit-test',
          'anchors',
          'plane-detection',
          'light-estimation',
          'depth-sensing',
          'mesh-detection'
        ]
      };

      logger.info('Requesting immersive-ar session for Quest 3...');
      const session = await navigator.xr!.requestSession('immersive-ar', sessionInit);

      
      if (session.environmentBlendMode !== 'additive' && session.environmentBlendMode !== 'alpha-blend') {
        logger.warn('Quest 3 AR session does not have expected blend mode:', session.environmentBlendMode);
      }

      logger.info('Quest 3 AR session started successfully', {
        environmentBlendMode: session.environmentBlendMode,
        supportedFrameRates: session.supportedFrameRates,
        inputSources: session.inputSources?.length || 0
      });

      
      usePlatformStore.getState().setXRMode(true);
      usePlatformStore.getState().setXRSessionState('active');

      
      await this.initializeVircadiaConnection();

      return true;

    } catch (error) {
      logger.error('Failed to auto-start Quest 3 AR session:', error);
      this.autoStartAttempted = false; 
      return false;
    }
  }

  
  private async configureQuest3Settings(): Promise<void> {
    const settingsStore = useSettingsStore.getState();

    
    const quest3Settings = {
      xr: {
        enabled: true,
        clientSideEnableXR: true,
        displayMode: 'immersive-ar' as const,
        spaceType: 'local-floor' as const,
        enableHandTracking: true,
        enablePassthroughPortal: true,
        passthroughOpacity: 1.0,
        passthroughBrightness: 1.0,
        passthroughContrast: 1.0,
        enablePlaneDetection: true,
        enableSceneUnderstanding: true,
        locomotionMethod: 'teleport' as const,
        movementSpeed: 1.0,
        interactionDistance: 1.5,
        quality: 'high' as const
      },
      auth: {
        enabled: false, 
        required: false
      },
      visualisation: {
        rendering: {
          context: 'quest3-ar' as const,
          enableAntialiasing: true,
          enableShadows: true,
          backgroundColor: 'transparent' 
        },
        physics: {
          enabled: true,
          boundsSize: 5.0, 
          maxVelocity: 0.01 
        }
      },
      system: {
        debug: {
          enabled: false 
        }
      }
    };

    
    settingsStore.updateSettings((draft) => {
      
      Object.assign(draft.xr, quest3Settings.xr);
      Object.assign(draft.auth, quest3Settings.auth);
      Object.assign(draft.visualisation.rendering, quest3Settings.visualisation.rendering);
      Object.assign(draft.visualisation.physics, quest3Settings.visualisation.physics);
      Object.assign(draft.system.debug, quest3Settings.system.debug);
    });

    logger.info('Quest 3 AR settings configured');
  }

  
  public isInQuest3ARMode(): boolean {
    const platformState = usePlatformStore.getState();
    return platformState.isXRMode &&
           platformState.platform === 'quest3' &&
           this.detectionResult?.shouldAutoStart === true;
  }

  
  private async initializeVircadiaConnection(): Promise<void> {
    try {
      logger.info('Initializing Vircadia connection for Quest 3 XR...');

      
      this.vircadiaClient = new ClientCore({
        serverUrl: import.meta.env.VITE_VIRCADIA_SERVER_URL || 'ws://localhost:3020/world/ws',
        authToken: import.meta.env.VITE_VIRCADIA_AUTH_TOKEN || 'system-token',
        authProvider: import.meta.env.VITE_VIRCADIA_AUTH_PROVIDER || 'system',
        reconnectAttempts: 5,
        reconnectDelay: 5000,
        debug: import.meta.env.DEV || false,
        suppress: false
      });

      
      const connectionInfo = await this.vircadiaClient.Utilities.Connection.connect({
        timeoutMs: 10000
      });

      logger.info('Vircadia connected for Quest 3 XR', {
        agentId: connectionInfo.agentId,
        sessionId: connectionInfo.sessionId
      });

    } catch (error) {
      logger.error('Failed to initialize Vircadia connection:', error);
      
    }
  }

  
  public getVircadiaClient(): ClientCore | null {
    return this.vircadiaClient;
  }

  
  public disconnectVircadia(): void {
    if (this.vircadiaClient) {
      logger.info('Disconnecting Vircadia client');
      this.vircadiaClient.dispose();
      this.vircadiaClient = null;
    }
  }

  
  public resetDetection(): void {
    this.detectionResult = null;
    this.autoStartAttempted = false;
    this.disconnectVircadia();
  }
}

export const quest3AutoDetector = Quest3AutoDetector.getInstance();