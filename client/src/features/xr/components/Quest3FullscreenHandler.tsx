import React, { useEffect, useState } from 'react';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('Quest3FullscreenHandler');

interface Quest3FullscreenHandlerProps {
  children: React.ReactNode;
  onFullscreenChange?: (isFullscreen: boolean) => void;
}

/**
 * Handles Quest 3 browser-specific fullscreen requirements for WebXR
 * The Quest 3 browser requires content to be in fullscreen mode before
 * it can properly activate AR/VR projection modes.
 */
export const Quest3FullscreenHandler: React.FC<Quest3FullscreenHandlerProps> = ({ 
  children, 
  onFullscreenChange 
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showFullscreenPrompt, setShowFullscreenPrompt] = useState(false);

  useEffect(() => {
    // Check if we're on Quest 3 browser
    const isQuest3 = /OculusBrowser/.test(navigator.userAgent) && 
                     /Quest 3/.test(navigator.userAgent);
    
    if (isQuest3) {
      logger.info('Quest 3 browser detected, setting up fullscreen handler');
      setShowFullscreenPrompt(true);
    }

    // Listen for fullscreen changes
    const handleFullscreenChange = () => {
      const fullscreenElement = document.fullscreenElement || 
                               (document as any).webkitFullscreenElement ||
                               (document as any).mozFullScreenElement ||
                               (document as any).msFullscreenElement;
      
      const isNowFullscreen = !!fullscreenElement;
      setIsFullscreen(isNowFullscreen);
      setShowFullscreenPrompt(!isNowFullscreen);
      
      if (onFullscreenChange) {
        onFullscreenChange(isNowFullscreen);
      }

      logger.info('Fullscreen state changed:', { isFullscreen: isNowFullscreen });
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, [onFullscreenChange]);

  const requestFullscreen = async () => {
    try {
      const element = document.documentElement;
      
      if (element.requestFullscreen) {
        await element.requestFullscreen();
      } else if ((element as any).webkitRequestFullscreen) {
        await (element as any).webkitRequestFullscreen();
      } else if ((element as any).mozRequestFullScreen) {
        await (element as any).mozRequestFullScreen();
      } else if ((element as any).msRequestFullscreen) {
        await (element as any).msRequestFullscreen();
      }
      
      logger.info('Fullscreen requested successfully');
    } catch (error) {
      logger.error('Failed to request fullscreen:', error);
    }
  };

  return (
    <>
      {/* Fullscreen prompt for Quest 3 browser */}
      {showFullscreenPrompt && (
        <div style={{
          position: 'fixed',
          top: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 2000,
          backgroundColor: 'rgba(0, 0, 0, 0.9)',
          color: 'white',
          padding: '20px',
          borderRadius: '10px',
          textAlign: 'center',
          maxWidth: '90%',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
        }}>
          <h3 style={{ margin: '0 0 10px 0', fontSize: '18px' }}>
            Quest 3 AR Mode
          </h3>
          <p style={{ margin: '0 0 20px 0', fontSize: '14px' }}>
            To enable AR mode, the browser needs to be in fullscreen.
            This allows proper projection mode selection.
          </p>
          <button
            onClick={requestFullscreen}
            style={{
              backgroundColor: '#1976d2',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              padding: '10px 30px',
              fontSize: '16px',
              cursor: 'pointer',
              transition: 'background-color 0.3s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#1565c0';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#1976d2';
            }}
          >
            Enter Fullscreen
          </button>
        </div>
      )}

      {/* Media player wrapper - Quest 3 needs this for projection mode detection */}
      <div 
        id="quest3-media-wrapper"
        style={{
          width: '100%',
          height: '100%',
          position: 'relative'
        }}
        data-vr-projection="360"
        data-vr-format="3d"
      >
        {children}
      </div>
    </>
  );
};