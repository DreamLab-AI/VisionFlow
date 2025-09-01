import { useSelectiveSetting, useSettingSetter } from './useSelectiveSettingsStore';
import { SettingsPath } from '../types/generated/settings';
import { useCallback } from 'react';

/**
 * Hook for theme and appearance settings with selective access
 * Provides optimized access to UI theme configuration
 */
export function useThemeSettings() {
  const { set, batchSet } = useSettingSetter();
  
  // Core theme settings
  const theme = useSelectiveSetting<'light' | 'dark' | 'auto' | 'system'>('ui.theme');
  const accentColor = useSelectiveSetting<string>('ui.accentColor');
  const colorScheme = useSelectiveSetting<'default' | 'monochrome' | 'colorblind' | 'high-contrast'>('ui.colorScheme');
  const fontSize = useSelectiveSetting<'small' | 'medium' | 'large' | 'extra-large'>('ui.fontSize');
  const fontFamily = useSelectiveSetting<string>('ui.fontFamily');
  const borderRadius = useSelectiveSetting<number>('ui.borderRadius');
  
  // Layout settings
  const sidebarPosition = useSelectiveSetting<'left' | 'right' | 'hidden'>('ui.layout.sidebarPosition');
  const sidebarWidth = useSelectiveSetting<number>('ui.layout.sidebarWidth');
  const headerHeight = useSelectiveSetting<number>('ui.layout.headerHeight');
  const contentMaxWidth = useSelectiveSetting<number>('ui.layout.contentMaxWidth');
  const compactMode = useSelectiveSetting<boolean>('ui.layout.compactMode');
  
  // Visual effects
  const animations = useSelectiveSetting<boolean>('ui.effects.animations');
  const transitions = useSelectiveSetting<boolean>('ui.effects.transitions');
  const shadows = useSelectiveSetting<boolean>('ui.effects.shadows');
  const blur = useSelectiveSetting<boolean>('ui.effects.blur');
  const glassmorphism = useSelectiveSetting<boolean>('ui.effects.glassmorphism');
  const reducedMotion = useSelectiveSetting<boolean>('ui.accessibility.reducedMotion');
  
  // Accessibility settings
  const highContrast = useSelectiveSetting<boolean>('ui.accessibility.highContrast');
  const focusRing = useSelectiveSetting<boolean>('ui.accessibility.focusRing');
  const keyboardNavigation = useSelectiveSetting<boolean>('ui.accessibility.keyboardNavigation');
  const screenReaderSupport = useSelectiveSetting<boolean>('ui.accessibility.screenReader');
  
  // Color customization
  const primaryColor = useSelectiveSetting<string>('ui.colors.primary');
  const secondaryColor = useSelectiveSetting<string>('ui.colors.secondary');
  const backgroundColor = useSelectiveSetting<string>('ui.colors.background');
  const surfaceColor = useSelectiveSetting<string>('ui.colors.surface');
  const textColor = useSelectiveSetting<string>('ui.colors.text');
  const mutedTextColor = useSelectiveSetting<string>('ui.colors.mutedText');
  const borderColor = useSelectiveSetting<string>('ui.colors.border');
  const errorColor = useSelectiveSetting<string>('ui.colors.error');
  const warningColor = useSelectiveSetting<string>('ui.colors.warning');
  const successColor = useSelectiveSetting<string>('ui.colors.success');
  
  // Helper functions
  const updateThemeSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `ui.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateLayoutSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `ui.layout.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateEffectsSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `ui.effects.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateAccessibilitySetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `ui.accessibility.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  const updateColorSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `ui.colors.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  // Theme preset functions
  const setThemePreset = useCallback(async (preset: 'default' | 'minimal' | 'vibrant' | 'professional' | 'accessibility') => {
    const presets = {
      default: [
        { path: 'ui.theme', value: 'system' },
        { path: 'ui.accentColor', value: '#3b82f6' },
        { path: 'ui.colorScheme', value: 'default' },
        { path: 'ui.fontSize', value: 'medium' },
        { path: 'ui.fontFamily', value: 'system-ui, -apple-system, sans-serif' },
        { path: 'ui.borderRadius', value: 8 },
        { path: 'ui.effects.animations', value: true },
        { path: 'ui.effects.transitions', value: true },
        { path: 'ui.effects.shadows', value: true },
        { path: 'ui.effects.blur', value: false },
        { path: 'ui.layout.compactMode', value: false }
      ],
      minimal: [
        { path: 'ui.theme', value: 'light' },
        { path: 'ui.accentColor', value: '#000000' },
        { path: 'ui.colorScheme', value: 'monochrome' },
        { path: 'ui.borderRadius', value: 0 },
        { path: 'ui.effects.animations', value: false },
        { path: 'ui.effects.transitions', value: false },
        { path: 'ui.effects.shadows', value: false },
        { path: 'ui.effects.blur', value: false },
        { path: 'ui.layout.compactMode', value: true }
      ],
      vibrant: [
        { path: 'ui.theme', value: 'dark' },
        { path: 'ui.accentColor', value: '#ff6b6b' },
        { path: 'ui.colorScheme', value: 'default' },
        { path: 'ui.borderRadius', value: 16 },
        { path: 'ui.effects.animations', value: true },
        { path: 'ui.effects.transitions', value: true },
        { path: 'ui.effects.shadows', value: true },
        { path: 'ui.effects.blur', value: true },
        { path: 'ui.effects.glassmorphism', value: true }
      ],
      professional: [
        { path: 'ui.theme', value: 'light' },
        { path: 'ui.accentColor', value: '#1f2937' },
        { path: 'ui.colorScheme', value: 'default' },
        { path: 'ui.fontSize', value: 'small' },
        { path: 'ui.fontFamily', value: 'Georgia, serif' },
        { path: 'ui.borderRadius', value: 4 },
        { path: 'ui.effects.animations', value: false },
        { path: 'ui.layout.compactMode', value: true }
      ],
      accessibility: [
        { path: 'ui.colorScheme', value: 'high-contrast' },
        { path: 'ui.fontSize', value: 'large' },
        { path: 'ui.accessibility.highContrast', value: true },
        { path: 'ui.accessibility.focusRing', value: true },
        { path: 'ui.accessibility.reducedMotion', value: true },
        { path: 'ui.effects.animations', value: false },
        { path: 'ui.effects.transitions', value: false },
        { path: 'ui.borderRadius', value: 8 }
      ]
    };
    
    await batchSet(presets[preset]);
  }, [batchSet]);
  
  // Accessibility optimizations
  const enableHighContrast = useCallback(async () => {
    await batchSet([
      { path: 'ui.accessibility.highContrast', value: true },
      { path: 'ui.colorScheme', value: 'high-contrast' },
      { path: 'ui.accessibility.focusRing', value: true },
      { path: 'ui.colors.primary', value: '#000000' },
      { path: 'ui.colors.background', value: '#ffffff' },
      { path: 'ui.colors.text', value: '#000000' }
    ]);
  }, [batchSet]);
  
  const enableReducedMotion = useCallback(async () => {
    await batchSet([
      { path: 'ui.accessibility.reducedMotion', value: true },
      { path: 'ui.effects.animations', value: false },
      { path: 'ui.effects.transitions', value: false },
      { path: 'ui.effects.blur', value: false }
    ]);
  }, [batchSet]);
  
  const optimizeForColorblind = useCallback(async () => {
    await batchSet([
      { path: 'ui.colorScheme', value: 'colorblind' },
      { path: 'ui.colors.primary', value: '#0066cc' },  // Blue
      { path: 'ui.colors.secondary', value: '#ff9900' }, // Orange
      { path: 'ui.colors.success', value: '#0066cc' },  // Blue instead of green
      { path: 'ui.colors.error', value: '#cc0000' },    // Red
      { path: 'ui.colors.warning', value: '#ff9900' }   // Orange
    ]);
  }, [batchSet]);
  
  // Theme switching utilities
  const toggleTheme = useCallback(async () => {
    const currentTheme = theme;
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    await set('ui.theme', newTheme);
  }, [theme, set]);
  
  const setSystemTheme = useCallback(async () => {
    await set('ui.theme', 'system');
  }, [set]);
  
  return {
    // Core theme settings
    theme,
    accentColor,
    colorScheme,
    fontSize,
    fontFamily,
    borderRadius,
    
    // Layout settings
    layout: {
      sidebarPosition,
      sidebarWidth,
      headerHeight,
      contentMaxWidth,
      compactMode
    },
    
    // Visual effects
    effects: {
      animations,
      transitions,
      shadows,
      blur,
      glassmorphism
    },
    
    // Accessibility
    accessibility: {
      highContrast,
      reducedMotion,
      focusRing,
      keyboardNavigation,
      screenReaderSupport
    },
    
    // Colors
    colors: {
      primary: primaryColor,
      secondary: secondaryColor,
      background: backgroundColor,
      surface: surfaceColor,
      text: textColor,
      mutedText: mutedTextColor,
      border: borderColor,
      error: errorColor,
      warning: warningColor,
      success: successColor
    },
    
    // Update functions
    updateThemeSetting,
    updateLayoutSetting,
    updateEffectsSetting,
    updateAccessibilitySetting,
    updateColorSetting,
    
    // Preset functions
    setThemePreset,
    enableHighContrast,
    enableReducedMotion,
    optimizeForColorblind,
    
    // Utilities
    toggleTheme,
    setSystemTheme
  };
}

/**
 * Hook for responsive design settings
 */
export function useResponsiveSettings() {
  const { set } = useSettingSetter();
  
  const responsiveEnabled = useSelectiveSetting<boolean>('ui.responsive.enabled');
  const breakpoints = useSelectiveSetting<Record<string, number>>('ui.responsive.breakpoints');
  const mobileOptimized = useSelectiveSetting<boolean>('ui.responsive.mobileOptimized');
  const touchOptimized = useSelectiveSetting<boolean>('ui.responsive.touchOptimized');
  
  const updateResponsiveSetting = useCallback(async <T>(
    path: string,
    value: T
  ) => {
    const fullPath: SettingsPath = `ui.responsive.${path}`;
    await set(fullPath, value);
  }, [set]);
  
  return {
    responsiveEnabled,
    breakpoints,
    mobileOptimized,
    touchOptimized,
    updateResponsiveSetting
  };
}