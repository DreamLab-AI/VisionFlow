/**
 * Preset Selector Component
 *
 * One-click quality preset selection for 571 settings
 */

import React, { useState } from 'react';
import { Zap, Battery, Cpu, Rocket, Info, Check } from 'lucide-react';
import { Button } from '../../design-system/components/Button';
import { QUALITY_PRESETS, type QualityPreset } from '../presets/qualityPresets';
import { useSettingsStore } from '../../../store/settingsStore';
import { cn } from '../../../lib/utils';

interface PresetSelectorProps {
  className?: string;
  compact?: boolean;
  showDescription?: boolean;
}

export const PresetSelector: React.FC<PresetSelectorProps> = ({
  className,
  compact = false,
  showDescription = true
}) => {
  const { settings, updateSettings } = useSettingsStore();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [isApplying, setIsApplying] = useState(false);
  const [showInfo, setShowInfo] = useState<string | null>(null);

  const applyPreset = async (preset: QualityPreset) => {
    setIsApplying(true);
    setSelectedPreset(preset.id);

    try {
      // Apply all preset settings
      await updateSettings(preset.settings);

      // Optional: Show success notification
      console.log(`Applied ${preset.name} preset successfully`);

      // Store the current preset ID
      localStorage.setItem('quality-preset', preset.id);
    } catch (error) {
      console.error('Failed to apply preset:', error);
    } finally {
      setIsApplying(false);
    }
  };

  const icons = {
    low: Battery,
    medium: Cpu,
    high: Zap,
    ultra: Rocket
  };

  const categoryColors = {
    performance: 'bg-blue-500/10 hover:bg-blue-500/20 border-blue-500/30',
    balanced: 'bg-green-500/10 hover:bg-green-500/20 border-green-500/30',
    quality: 'bg-purple-500/10 hover:bg-purple-500/20 border-purple-500/30',
    ultra: 'bg-orange-500/10 hover:bg-orange-500/20 border-orange-500/30'
  };

  if (compact) {
    return (
      <div className={cn("flex gap-2", className)}>
        {QUALITY_PRESETS.map(preset => {
          const Icon = icons[preset.id as keyof typeof icons];
          const isActive = selectedPreset === preset.id;

          return (
            <button
              key={preset.id}
              onClick={() => applyPreset(preset)}
              disabled={isApplying}
              className={cn(
                "relative px-3 py-2 rounded-lg border transition-all",
                "flex items-center gap-2",
                categoryColors[preset.category],
                isActive && "ring-2 ring-primary",
                isApplying && "opacity-50 cursor-not-allowed"
              )}
              title={preset.description}
            >
              <Icon className="w-4 h-4" />
              <span className="text-sm font-medium">{preset.name.split(' ')[0]}</span>
              {isActive && <Check className="w-3 h-3 ml-1" />}
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className={cn("grid grid-cols-2 gap-4", className)}>
      {QUALITY_PRESETS.map(preset => {
        const Icon = icons[preset.id as keyof typeof icons];
        const isActive = selectedPreset === preset.id;
        const showingInfo = showInfo === preset.id;

        return (
          <div
            key={preset.id}
            className={cn(
              "relative rounded-lg border p-4 transition-all",
              categoryColors[preset.category],
              isActive && "ring-2 ring-primary shadow-lg",
              "hover:shadow-md"
            )}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <Icon className="w-6 h-6" />
                <div>
                  <div className="text-sm font-semibold">{preset.name}</div>
                  {isActive && (
                    <div className="flex items-center gap-1 text-xs text-primary">
                      <Check className="w-3 h-3" />
                      Active
                    </div>
                  )}
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowInfo(showingInfo ? null : preset.id);
                }}
                className="p-1 hover:bg-black/10 rounded"
              >
                <Info className="w-4 h-4" />
              </button>
            </div>

            {/* Description */}
            {showDescription && (
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-3">
                {preset.description}
              </p>
            )}

            {/* System Requirements (when info is shown) */}
            {showingInfo && preset.systemRequirements && (
              <div className="mb-3 p-2 bg-black/5 rounded text-xs space-y-1">
                <div className="font-semibold mb-1">System Requirements:</div>
                {preset.systemRequirements.minRAM && (
                  <div>RAM: {preset.systemRequirements.minRAM}GB+</div>
                )}
                {preset.systemRequirements.minVRAM && (
                  <div>VRAM: {preset.systemRequirements.minVRAM}GB+</div>
                )}
                {preset.systemRequirements.recommendedGPU && (
                  <div>GPU: {preset.systemRequirements.recommendedGPU}</div>
                )}
              </div>
            )}

            {/* Apply Button */}
            <Button
              onClick={() => applyPreset(preset)}
              disabled={isApplying}
              className={cn(
                "w-full",
                isActive && "bg-primary text-white"
              )}
              variant={isActive ? "default" : "outline"}
            >
              {isApplying && selectedPreset === preset.id ? (
                <>
                  <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin mr-2" />
                  Applying...
                </>
              ) : isActive ? (
                'Current Preset'
              ) : (
                'Apply Preset'
              )}
            </Button>
          </div>
        );
      })}
    </div>
  );
};

/**
 * Compact preset selector for header/toolbar
 */
export const PresetSelectorCompact: React.FC<{ className?: string }> = ({ className }) => {
  return <PresetSelector compact className={className} />;
};
