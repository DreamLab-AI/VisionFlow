import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useSpacePilot } from '../hooks/useSpacePilot';
import { useSettingsStore } from '../../../store/settingsStore';

interface SpacePilotIntegrationProps {
  orbitControlsRef?: React.RefObject<any>;
}

/**
 * SpacePilot integration that enhances existing OrbitControls with 6DOF input
 * This component should be placed in the same Canvas as your OrbitControls
 */
export const SpacePilotIntegration: React.FC<SpacePilotIntegrationProps> = ({ orbitControlsRef }) => {
  const { camera } = useThree();
  const settings = useSettingsStore(state => state.settings);
  const spacePilotEnabled = settings?.visualisation?.spacePilot?.enabled !== false;

  // Initialize SpacePilot hook with orbitControlsRef
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

  // Note: Auto-connect removed due to browser security requirements
  // WebHID requires user gesture for device permissions
  // Connection must be initiated through user interaction (button click)

  // Store connection state for auto-reconnect
  useEffect(() => {
    if (spacePilot.isConnected) {
      localStorage.setItem('spacepilot-auto-connect', 'true');
    }
  }, [spacePilot.isConnected]);

  // The actual 6DOF control is handled by the SpacePilotController
  // which is initialized inside the useSpacePilot hook
  // It automatically integrates with the camera and OrbitControls

  // This component doesn't render anything visible
  return null;
};

export default SpacePilotIntegration;