import React, { useRef, useEffect } from 'react';
import { useSelectiveSetting } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { createLogger } from '@/utils/logger';

const logger = createLogger('AudioVisualizer');

interface AudioVisualizerProps {
  className?: string;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ className }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Only subscribe to audio-related settings
  const enabled = useSelectiveSetting<boolean>('audio.visualizer.enabled');
  const sensitivity = useSelectiveSetting<number>('audio.visualizer.sensitivity');
  const barCount = useSelectiveSetting<number>('audio.visualizer.barCount');
  const colorScheme = useSelectiveSetting<string>('audio.visualizer.colorScheme');
  
  useEffect(() => {
    if (!enabled || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    let animationFrame: number;
    
    const renderVisualizer = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Simulate audio bars for demo
      const barWidth = canvas.width / barCount;
      
      for (let i = 0; i < barCount; i++) {
        const height = Math.random() * canvas.height * sensitivity;
        const x = i * barWidth;
        
        // Apply color scheme
        ctx.fillStyle = colorScheme === 'rainbow' 
          ? `hsl(${(i / barCount) * 360}, 70%, 60%)`
          : '#00ff88';
          
        ctx.fillRect(x, canvas.height - height, barWidth - 2, height);
      }
      
      animationFrame = requestAnimationFrame(renderVisualizer);
    };
    
    renderVisualizer();
    
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [enabled, sensitivity, barCount, colorScheme]);
  
  if (!enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Audio Visualizer</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Audio visualizer is disabled</p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Audio Visualizer</CardTitle>
      </CardHeader>
      <CardContent>
        <canvas
          ref={canvasRef}
          width={400}
          height={200}
          className="w-full border rounded"
          style={{ maxHeight: '200px' }}
        />
      </CardContent>
    </Card>
  );
};

export default AudioVisualizer;