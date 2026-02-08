import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import { useSpacePilot } from '../hooks/useSpacePilot';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('SpacePilot');

interface SpacePilotIntegrationProps {
  orbitControlsRef?: React.RefObject<any>;
}


export const SpacePilotIntegration: React.FC<SpacePilotIntegrationProps> = ({ orbitControlsRef }) => {
  const { camera } = useThree();
  const settings = useSettingsStore(state => state.settings);
  const spacePilotEnabled = settings?.visualisation?.spacePilot?.enabled !== false;

  
  const spacePilot = useSpacePilot({
    enabled: spacePilotEnabled,
    orbitControlsRef: orbitControlsRef,
    onConnect: () => {
      logger.info('Connected to 6DOF controller');
    },
    onDisconnect: () => {
      logger.info('Disconnected from 6DOF controller');
    },
    onModeChange: (mode) => {
      logger.info('Mode changed to:', mode);
    }
  });

  
  
  

  
  useEffect(() => {
    if (spacePilot.isConnected) {
      localStorage.setItem('spacepilot-auto-connect', 'true');
    }
  }, [spacePilot.isConnected]);

  
  
  

  
  return null;
};

export default SpacePilotIntegration;