import React, { useState, useEffect } from 'react';
import { useVoiceInteraction } from '../hooks/useVoiceInteraction';
import { VoiceWebSocketService } from '../services/VoiceWebSocketService';
import { AudioInputService } from '../services/AudioInputService';
import { Mic, MicOff, Volume2 } from 'lucide-react';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('VoiceStatusIndicator');

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
      logger.error('Voice interaction error:', error);
      setHasError(true);
    }
  });

  
  useEffect(() => {
    
    
    setIsSupported(true); 
    
    
    
    
    
    
    
    
  }, []);

  
  useEffect(() => {
    const voiceService = VoiceWebSocketService.getInstance();
    const handleAudioLevel = (level: number) => setAudioLevel(level);
    const audioInput = voiceService.getAudioInput();

    audioInput.on('audioLevel', handleAudioLevel);

    return () => {
      audioInput.off('audioLevel', handleAudioLevel);
    };
  }, []);

  
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      
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

  
  useEffect(() => {
    if (isListening && hasError) {
      setHasError(false);
    }
  }, [isListening, hasError]);

  const handleToggle = async () => {
    if (!isSupported) {
      setHasError(true);
      logger.error('Voice features not supported in this browser/environment');
      return;
    }

    try {
      setHasError(false);
      await toggleListening();
    } catch (error) {
      logger.error('Failed to toggle voice input:', error);
      setHasError(true);
    }
  };

  return (
    <button 
      onClick={handleToggle}
      className={`flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity ${className}`}
      aria-label={isListening ? 'Stop voice input' : 'Start voice input'}>
      {}
      <div className="relative">
        {}
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

        {}
        {(isListening || isSpeaking) && (
          <div 
            className={`
              absolute inset-0 rounded-full
              ${isListening ? 'bg-red-500' : 'bg-blue-500'}
              opacity-30 animate-ping
            `}
          />
        )}

        {}
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

      {}
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