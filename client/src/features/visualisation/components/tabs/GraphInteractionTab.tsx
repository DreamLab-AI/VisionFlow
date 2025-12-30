

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Clock,
  Users,
  // @ts-ignore - Icon exists in lucide-react but types may be outdated
  Glasses,
  Play,
  Pause,
  RotateCcw,
  // @ts-ignore - Icon exists in lucide-react but types may be outdated
  FastForward,
  // @ts-ignore - Icon exists in lucide-react but types may be outdated
  Rewind,
  MapPin,
  // @ts-ignore - Icon exists in lucide-react but types may be outdated
  Radio,
  // @ts-ignore - Icon exists in lucide-react but types may be outdated
  Gamepad2,
  AlertCircle,
  Sparkles,
  Navigation,
  Eye
} from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Switch } from '@/features/design-system/components/Switch';
import { Label } from '@/features/design-system/components/Label';
import { Badge } from '@/features/design-system/components/Badge';
import { Slider } from '@/features/design-system/components/Slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Progress } from '@/features/design-system/components/Progress';
import { toast } from '@/features/design-system/components/Toast';
import { interactionApi, type GraphProcessingProgress, type GraphProcessingResult } from '@/services/interactionApi';
import { webSocketService } from '@/services/WebSocketService';
import { useSettingsStore } from '@/store/settingsStore';

interface GraphInteractionTabProps {
  graphId?: string;
  onFeatureUpdate?: (feature: string, data: any) => void;
}

interface ProcessingState {
  taskId: string | null;
  isProcessing: boolean;
  progress: number;
  stage: string;
  currentOperation: string;
  estimatedTimeRemaining?: number;
  metrics?: {
    stepsProcessed: number;
    totalSteps: number;
    currentStep: string;
    operationsCompleted: number;
  };
  error?: string;
}

export const GraphInteractionTab: React.FC<GraphInteractionTabProps> = ({
  graphId = 'default',
  onFeatureUpdate
}) => {
  
  const [processingState, setProcessingState] = useState<ProcessingState>({
    taskId: null,
    isProcessing: false,
    progress: 0,
    stage: 'idle',
    currentOperation: 'Ready'
  });
  const [retryCount, setRetryCount] = useState(0);
  const maxRetries = 3;

  
  const [timeTravelActive, setTimeTravelActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState([1]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [timeTravelTaskId, setTimeTravelTaskId] = useState<string | null>(null);

  
  const [collaborationActive, setCollaborationActive] = useState(false);
  const [participantCount, setParticipantCount] = useState(1);
  const [sessionId, setSessionId] = useState<string>('');
  
  
  const [vrModeActive, setVrModeActive] = useState(false);
  const [arModeActive, setArModeActive] = useState(false);
  const [handTrackingEnabled, setHandTrackingEnabled] = useState(false);
  const [hapticFeedback, setHapticFeedback] = useState(true);
  
  
  const [explorationMode, setExplorationMode] = useState(false);
  const [tourActive, setTourActive] = useState(false);
  const [currentTour, setCurrentTour] = useState<string>('');
  const [tourWaypoints, setTourWaypoints] = useState(0);

  
  const [wsConnected, setWsConnected] = useState(false);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  
  const { settings, updateSettings } = useSettingsStore();
  const headTrackingEnabled = settings?.visualisation?.interaction?.headTrackedParallax?.enabled ?? false;

  const handleHeadTrackingToggle = useCallback((enabled: boolean) => {
    updateSettings(draft => {
      if (!draft.visualisation) draft.visualisation = {} as any;
      if (!draft.visualisation.interaction) {
        draft.visualisation.interaction = {
          headTrackedParallax: {
            enabled: false,
            sensitivity: 1.0,
            cameraMode: 'asymmetricFrustum'
          }
        };
      }
      if (!draft.visualisation.interaction.headTrackedParallax) {
        draft.visualisation.interaction.headTrackedParallax = {
          enabled: false,
          sensitivity: 1.0,
          cameraMode: 'asymmetricFrustum'
        };
      }
      draft.visualisation.interaction.headTrackedParallax.enabled = enabled;
    });
    onFeatureUpdate?.('headTracking', { enabled });
    toast({
      title: `Head Tracking ${enabled ? 'Enabled' : 'Disabled'}`,
      description: enabled
        ? 'Webcam will be used to create a parallax effect.'
        : 'Head tracking has been turned off.'
    });
  }, [updateSettings, onFeatureUpdate]);

  const handleTimeTravelToggle = useCallback(() => {
    const newState = !timeTravelActive;
    setTimeTravelActive(newState);
    
    if (newState) {
      
      setTotalSteps(10);
      setCurrentStep(10); 
      onFeatureUpdate?.('timeTravel', { 
        enabled: true, 
        currentStep: 10, 
        totalSteps: 10 
      });
      
      toast({
        title: "Time Travel Mode Activated",
        description: "Navigate through graph history with the timeline controls"
      });
    } else {
      setIsPlaying(false);
      onFeatureUpdate?.('timeTravel', { enabled: false });
      
      toast({
        title: "Time Travel Mode Deactivated",
        description: "Returned to current graph state"
      });
    }
  }, [timeTravelActive, onFeatureUpdate]);

  const handleCollaborationToggle = useCallback(() => {
    const newState = !collaborationActive;
    setCollaborationActive(newState);
    
    if (newState) {
      const newSessionId = `session-${Date.now()}`;
      setSessionId(newSessionId);
      onFeatureUpdate?.('collaboration', { 
        enabled: true, 
        sessionId: newSessionId,
        participants: 1
      });
      
      toast({
        title: "Collaboration Session Started",
        description: "Share this session to invite collaborators"
      });
    } else {
      onFeatureUpdate?.('collaboration', { enabled: false });
      setSessionId('');
      setParticipantCount(1);
      
      toast({
        title: "Collaboration Session Ended",
        description: "All participants have been disconnected"
      });
    }
  }, [collaborationActive, onFeatureUpdate]);

  const handleVrModeToggle = useCallback(() => {
    const newState = !vrModeActive;
    setVrModeActive(newState);
    
    onFeatureUpdate?.('vrMode', { 
      enabled: newState,
      handTracking: handTrackingEnabled,
      hapticFeedback
    });
    
    toast({
      title: newState ? "VR Mode Activated" : "VR Mode Deactivated",
      description: newState 
        ? "Entering immersive virtual reality environment..." 
        : "Returned to standard 2D interface"
    });
  }, [vrModeActive, handTrackingEnabled, hapticFeedback, onFeatureUpdate]);

  const handleArModeToggle = useCallback(() => {
    const newState = !arModeActive;
    setArModeActive(newState);
    
    onFeatureUpdate?.('arMode', { enabled: newState });
    
    toast({
      title: newState ? "AR Mode Activated" : "AR Mode Deactivated",
      description: newState 
        ? "Overlaying graph data onto real world..." 
        : "Returned to standard interface"
    });
  }, [arModeActive, onFeatureUpdate]);

  const handleTimelineNavigation = useCallback((direction: 'previous' | 'next') => {
    if (!timeTravelActive) return;
    
    const newStep = direction === 'next' 
      ? Math.min(totalSteps, currentStep + 1)
      : Math.max(0, currentStep - 1);
    
    setCurrentStep(newStep);
    onFeatureUpdate?.('timeTravel', { 
      currentStep: newStep, 
      totalSteps,
      action: 'navigate'
    });
    
    toast({
      title: `Navigated to Step ${newStep}`,
      description: `Viewing graph state ${newStep}/${totalSteps}`
    });
  }, [timeTravelActive, currentStep, totalSteps, onFeatureUpdate]);

  const togglePlayback = useCallback(() => {
    const newPlayState = !isPlaying;
    setIsPlaying(newPlayState);
    
    onFeatureUpdate?.('timeTravel', { 
      isPlaying: newPlayState,
      playbackSpeed: playbackSpeed[0]
    });
    
    toast({
      title: newPlayState ? "Timeline Playback Started" : "Timeline Playback Paused",
      description: newPlayState ? `Playing at ${playbackSpeed[0]}x speed` : "Timeline paused"
    });
  }, [isPlaying, playbackSpeed, onFeatureUpdate]);

  const createExplorationTour = useCallback(() => {
    const tourId = `tour-${Date.now()}`;
    setCurrentTour(tourId);
    setTourActive(true);
    
    onFeatureUpdate?.('exploration', { 
      tourId,
      active: true,
      waypoints: []
    });
    
    toast({
      title: "Exploration Tour Created",
      description: "Add waypoints by clicking on interesting nodes"
    });
  }, [onFeatureUpdate]);

  return (
    <div className="space-y-4">
      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Head-Tracked Parallax
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="head-tracking-toggle">Enable Head Tracking</Label>
            <Switch
              id="head-tracking-toggle"
              checked={headTrackingEnabled}
              onCheckedChange={handleHeadTrackingToggle}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Uses your webcam to create a 3D parallax effect based on your head position.
          </p>
        </CardContent>
      </Card>

      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Time Travel Mode
            <Badge variant={wsConnected && !processingState.error ? "default" : "secondary"} className="text-xs">
              {processingState.isProcessing ? (
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                  Processing
                </span>
              ) : wsConnected ? (
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  Ready
                </span>
              ) : (
                <>
                  <AlertCircle className="h-3 w-3 mr-1" />
                  Offline
                </>
              )}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Enable Time Travel</Label>
            <Switch
              checked={timeTravelActive}
              onCheckedChange={handleTimeTravelToggle}
            />
          </div>
          
          {timeTravelActive && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="text-xs">Timeline Position</Label>
                  <span className="text-xs font-mono">{currentStep}/{totalSteps}</span>
                </div>
                <Progress value={(currentStep / totalSteps) * 100} className="w-full" />
              </div>
              
              <div className="flex items-center gap-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => handleTimelineNavigation('previous')}
                  disabled={currentStep <= 0}
                >
                  <Rewind className="h-3 w-3" />
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={togglePlayback}
                >
                  {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => handleTimelineNavigation('next')}
                  disabled={currentStep >= totalSteps}
                >
                  <FastForward className="h-3 w-3" />
                </Button>
              </div>
              
              <div className="space-y-1">
                <Label className="text-xs">Playback Speed</Label>
                <Slider
                  value={playbackSpeed}
                  onValueChange={setPlaybackSpeed}
                  min={0.1}
                  max={5}
                  step={0.1}
                  className="w-full"
                />
                <span className="text-xs text-muted-foreground">{playbackSpeed[0]}x speed</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Users className="h-4 w-4" />
            Collaboration
            <Badge variant={collaborationActive ? "default" : wsConnected ? "outline" : "secondary"} className="text-xs">
              {collaborationActive ? (
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  Active
                </span>
              ) : wsConnected ? (
                "Ready"
              ) : (
                <>
                  <AlertCircle className="h-3 w-3 mr-1" />
                  Offline
                </>
              )}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Enable Collaboration</Label>
            <Switch
              checked={collaborationActive}
              onCheckedChange={handleCollaborationToggle}
            />
          </div>
          
          {collaborationActive && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="text-xs space-y-1 p-2 bg-muted rounded">
                <div className="flex justify-between">
                  <span>Session ID:</span>
                  <span className="font-mono">{sessionId.slice(-8)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Participants:</span>
                  <span className="font-mono text-green-600">{participantCount}</span>
                </div>
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span className="text-green-600">Active</span>
                </div>
              </div>
              
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full"
                onClick={() => {
                  navigator.clipboard.writeText(`${window.location.origin}?session=${sessionId}`);
                  toast({ title: "Session link copied to clipboard" });
                }}
              >
                <Radio className="h-3 w-3 mr-1" />
                Copy Session Link
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Glasses className="h-4 w-4" />
            Immersive Modes
            <Badge variant={vrModeActive || arModeActive ? "default" : "outline"} className="text-xs">
              {vrModeActive ? (
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-purple-500 rounded-full" />
                  VR Active
                </span>
              ) : arModeActive ? (
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  AR Active
                </span>
              ) : (
                "Ready"
              )}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <Button 
              variant={vrModeActive ? "default" : "outline"}
              size="sm" 
              className="w-full"
              onClick={handleVrModeToggle}
            >
              <Glasses className="h-3 w-3 mr-1" />
              {vrModeActive ? "Exit VR" : "Enter VR"}
            </Button>
            <Button 
              variant={arModeActive ? "default" : "outline"}
              size="sm" 
              className="w-full"
              onClick={handleArModeToggle}
            >
              <Sparkles className="h-3 w-3 mr-1" />
              {arModeActive ? "Exit AR" : "Enter AR"}
            </Button>
          </div>
          
          {(vrModeActive || arModeActive) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Hand Tracking</Label>
                <Switch
                  checked={handTrackingEnabled}
                  onCheckedChange={setHandTrackingEnabled}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-xs">Haptic Feedback</Label>
                <Switch
                  checked={hapticFeedback}
                  onCheckedChange={setHapticFeedback}
                />
              </div>
              
              <div className="text-xs space-y-1 p-2 bg-muted rounded">
                <div className="flex justify-between">
                  <span>Mode:</span>
                  <span className="font-mono text-blue-600">
                    {vrModeActive ? 'VR' : arModeActive ? 'AR' : 'Standard'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Tracking:</span>
                  <span className="font-mono text-green-600">
                    {handTrackingEnabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Navigation className="h-4 w-4" />
            Exploration Tools
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Exploration Mode</Label>
            <Switch
              checked={explorationMode}
              onCheckedChange={setExplorationMode}
            />
          </div>
          
          <div className="grid grid-cols-1 gap-2">
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={createExplorationTour}
              disabled={!explorationMode || !wsConnected}
            >
              <MapPin className="h-3 w-3 mr-1" />
              Create Guided Tour
            </Button>
          </div>
          
          {tourActive && (
            <div className="text-xs space-y-1 p-2 bg-muted rounded">
              <div className="flex justify-between">
                <span>Active Tour:</span>
                <span className="font-mono">{currentTour.slice(-8)}</span>
              </div>
              <div className="flex justify-between">
                <span>Waypoints:</span>
                <span className="font-mono">{tourWaypoints}</span>
              </div>
            </div>
          )}
          
          <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
            <div className="flex items-center gap-1 mb-1">
              <div className={`w-2 h-2 rounded-full ${
                wsConnected ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <strong>Status:</strong> {wsConnected ? 'Connected' : 'Disconnected'}
            </div>
            {processingState.error && (
              <div className="text-red-500 mt-1">Error: {processingState.error}</div>
            )}
            {!wsConnected && (
              <div>WebSocket connection required for real-time features</div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GraphInteractionTab;