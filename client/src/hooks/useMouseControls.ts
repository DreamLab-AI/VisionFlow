import { useEffect } from 'react';
import { OrbitControls } from '@react-three/drei';
import { SpaceDriver } from '../services/SpaceDriverService';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('useMouseControls');


export function useMouseControls(orbitControlsRef: React.RefObject<OrbitControls>) {
  useEffect(() => {
    
    const ensureControlsEnabled = () => {
      if (orbitControlsRef.current && 'enabled' in orbitControlsRef.current) {
        orbitControlsRef.current.enabled = true;
        logger.debug('OrbitControls enabled');
      }
    };

    
    const hasWebHID = 'hid' in navigator;
    const isSecureContext = window.isSecureContext;

    if (!hasWebHID || !isSecureContext) {
      
      ensureControlsEnabled();
      logger.info('WebHID unavailable or insecure context - OrbitControls permanently enabled', { hasWebHID, isSecureContext });
    }

    
    const handleWebHIDUnavailable = () => {
      ensureControlsEnabled();
    };

    
    const handleDisconnect = () => {
      ensureControlsEnabled();
    };

    SpaceDriver.addEventListener('webhid-unavailable', handleWebHIDUnavailable);
    SpaceDriver.addEventListener('disconnect', handleDisconnect);

    
    setTimeout(ensureControlsEnabled, 100);

    return () => {
      SpaceDriver.removeEventListener('webhid-unavailable', handleWebHIDUnavailable);
      SpaceDriver.removeEventListener('disconnect', handleDisconnect);
    };
  }, [orbitControlsRef]);
}