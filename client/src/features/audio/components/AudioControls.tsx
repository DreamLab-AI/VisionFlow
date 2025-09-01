import React from 'react';
import { useSelectiveSetting, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Slider } from '@/features/design-system/components/Slider';
import { Switch } from '@/features/design-system/components/Switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Volume2, VolumeX, Mic, MicOff } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('AudioControls');

interface AudioControlsProps {
  className?: string;
}

export const AudioControls: React.FC<AudioControlsProps> = ({ className }) => {
  const { set } = useSettingSetter();
  
  // Subscribe to specific audio settings only
  const masterVolume = useSelectiveSetting<number>('audio.masterVolume');
  const microphoneEnabled = useSelectiveSetting<boolean>('audio.microphone.enabled');
  const microphoneGain = useSelectiveSetting<number>('audio.microphone.gain');
  const outputDevice = useSelectiveSetting<string>('audio.output.device');
  const inputDevice = useSelectiveSetting<string>('audio.input.device');
  const noiseSuppressionEnabled = useSelectiveSetting<boolean>('audio.noiseSuppression.enabled');
  const echoCancellationEnabled = useSelectiveSetting<boolean>('audio.echoCancellation.enabled');
  
  const handleVolumeChange = (value: number[]) => {
    set('audio.masterVolume', value[0]);
  };
  
  const handleMicrophoneToggle = (enabled: boolean) => {
    set('audio.microphone.enabled', enabled);
  };
  
  const handleMicrophoneGainChange = (value: number[]) => {
    set('audio.microphone.gain', value[0]);
  };
  
  const handleOutputDeviceChange = (device: string) => {
    set('audio.output.device', device);
  };
  
  const handleInputDeviceChange = (device: string) => {
    set('audio.input.device', device);
  };
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Volume2 size={20} />
          Audio Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Master Volume */}
        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            {masterVolume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
            Master Volume ({Math.round(masterVolume * 100)}%)
          </label>
          <Slider
            value={[masterVolume]}
            onValueChange={handleVolumeChange}
            min={0}
            max={1}
            step={0.01}
            className="w-full"
          />
        </div>
        
        {/* Microphone Controls */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium flex items-center gap-2">
              {microphoneEnabled ? <Mic size={16} /> : <MicOff size={16} />}
              Microphone
            </label>
            <Switch
              checked={microphoneEnabled}
              onCheckedChange={handleMicrophoneToggle}
            />
          </div>
          
          {microphoneEnabled && (
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Microphone Gain ({Math.round(microphoneGain * 100)}%)
              </label>
              <Slider
                value={[microphoneGain]}
                onValueChange={handleMicrophoneGainChange}
                min={0}
                max={2}
                step={0.01}
                className="w-full"
              />
            </div>
          )}
        </div>
        
        {/* Device Selection */}
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Output Device</label>
            <Select value={outputDevice} onValueChange={handleOutputDeviceChange}>
              <SelectTrigger>
                <SelectValue placeholder="Select output device" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">Default System Audio</SelectItem>
                <SelectItem value="speakers">Speakers</SelectItem>
                <SelectItem value="headphones">Headphones</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Input Device</label>
            <Select value={inputDevice} onValueChange={handleInputDeviceChange}>
              <SelectTrigger>
                <SelectValue placeholder="Select input device" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">Default System Microphone</SelectItem>
                <SelectItem value="internal">Internal Microphone</SelectItem>
                <SelectItem value="external">External Microphone</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        
        {/* Audio Processing */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Noise Suppression</label>
            <Switch
              checked={noiseSuppressionEnabled}
              onCheckedChange={(enabled) => set('audio.noiseSuppression.enabled', enabled)}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Echo Cancellation</label>
            <Switch
              checked={echoCancellationEnabled}
              onCheckedChange={(enabled) => set('audio.echoCancellation.enabled', enabled)}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default AudioControls;