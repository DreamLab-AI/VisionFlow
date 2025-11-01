import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useSpacePilot } from '../hooks/useSpacePilot';
import { useSettingsStore } from '../../../store/settingsStore';

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
      console.log('[SpacePilot] Connected to 6DOF controller');
    },
    onDisconnect: () => {
      console.log('[SpacePilot] Disconnected from 6DOF controller');
    },
    onModeChange: (mode) => {
      console.log('[SpacePilot] Mode changed to:', mode);
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