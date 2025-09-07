import React, { useState, useEffect } from 'react';
import { useVoiceInteraction } from '../hooks/useVoiceInteraction';
import { VoiceWebSocketService } from '../services/VoiceWebSocketService';
import { AudioInputService } from '../services/AudioInputService';
import { Mic, MicOff, Volume2 } from 'lucide-react';

export interface VoiceStatusIndicatorProps {
  className?: string;
}

export const VoiceStatusIndicator: React.FC<VoiceStatusIndicatorProps> = ({
  className = ''
}) => {
  const [audioLevel, setAudioLevel] = useState(0);
  const [hasError, setHasError] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  
  const { isListening, isSpeaking, toggleListening } = useVoiceInteraction({
    onError: (error) => {
      console.error('Voice interaction error:', error);
      setHasError(true);
    }
  });

  // Check browser support on mount
  useEffect(() => {
    // DEVELOPER WORKAROUND: Force enable for testing
    // Uncomment the line below to bypass browser support checks
    setIsSupported(true); // <- DEVELOPER OVERRIDE
    
    // Original check (commented out for testing)
    // const support = AudioInputService.getBrowserSupport();
    // const supported = support.getUserMedia && support.audioContext && support.mediaRecorder;
    // setIsSupported(supported);
    // if (!supported) {
    //   console.log('Browser support check:', support);
    // }
  }, []);

  // Get audio level for visual feedback
  useEffect(() => {
    const voiceService = VoiceWebSocketService.getInstance();
    const handleAudioLevel = (level: number) => setAudioLevel(level);
    const audioInput = voiceService.getAudioInput();

    audioInput.on('audioLevel', handleAudioLevel);

    return () => {
      audioInput.off('audioLevel', handleAudioLevel);
    };
  }, []);

  // Add spacebar hotkey support
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only activate if spacebar is pressed and no input is focused
      if (e.code === 'Space' && 
          !e.repeat && 
          !(e.target instanceof HTMLInputElement) && 
          !(e.target instanceof HTMLTextAreaElement)) {
        e.preventDefault();
        setIsSpacePressed(true);
        if (!isListening) {
          handleToggle();
        }
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        setIsSpacePressed(false);
        if (isListening) {
          handleToggle();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isListening]);

  // Clear error when listening state changes
  useEffect(() => {
    if (isListening && hasError) {
      setHasError(false);
    }
  }, [isListening, hasError]);

  const handleToggle = async () => {
    if (!isSupported) {
      setHasError(true);
      console.error('Voice features not supported in this browser/environment');
      return;
    }

    try {
      setHasError(false);
      await toggleListening();
    } catch (error) {
      console.error('Failed to toggle voice input:', error);
      setHasError(true);
    }
  };

  return (
    <button 
      onClick={handleToggle}
      className={`flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity ${className}`}
      aria-label={isListening ? 'Stop voice input' : 'Start voice input'}>
      {/* Status Icon */}
      <div className="relative">
        {/* Base icon */}
        <div className={`
          w-6 h-6 rounded-full flex items-center justify-center
          transition-all duration-200
          ${isListening ? 'bg-red-500' : isSpeaking ? 'bg-blue-500' : 'bg-gray-600'}
        `}>
          {isSpeaking ? (
            <Volume2 className="w-3 h-3 text-white" />
          ) : isListening ? (
            <Mic className="w-3 h-3 text-white" />
          ) : (
            <MicOff className="w-3 h-3 text-white" />
          )}
        </div>

        {/* Pulsing ring when active */}
        {(isListening || isSpeaking) && (
          <div 
            className={`
              absolute inset-0 rounded-full
              ${isListening ? 'bg-red-500' : 'bg-blue-500'}
              opacity-30 animate-ping
            `}
          />
        )}

        {/* Audio level indicator */}
        {isListening && (
          <div
            className="absolute inset-0 rounded-full bg-red-400 opacity-50"
            style={{
              transform: `scale(${1 + audioLevel * 0.3})`,
              transition: 'transform 100ms ease-out'
            }}
          />
        )}
      </div>

      {/* Status Text */}
      <div className="text-xs font-medium">
        {isListening ? (
          <span className="text-red-400">Recording (Space)</span>
        ) : isSpeaking ? (
          <span className="text-blue-400">Speaking</span>
        ) : (
          <span className="text-gray-400">Voice Ready (Space)</span>
        )}
      </div>
    </button>
  );
};