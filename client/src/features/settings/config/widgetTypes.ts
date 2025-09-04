// Extended widget type definitions for comprehensive settings management

export type SettingWidgetType =
  | 'toggle'
  | 'slider'
  | 'numberInput'
  | 'textInput'
  | 'colorPicker'
  | 'select'
  | 'radioGroup' // For a small set of mutually exclusive choices
  | 'rangeSlider' // For [number, number] arrays
  | 'buttonAction'
  | 'dualColorPicker'; // Custom type for [string, string] color arrays

export interface UISettingDefinition {
  label: string;
  type: SettingWidgetType;
  path: string; // Full path in the SettingsStore, e.g., "visualisation.nodes.baseColor"
  description?: string; // Tooltip text
  options?: Array<{ value: string | number; label: string }>; // For select
  min?: number; // For slider, numberInput, rangeSlider
  max?: number; // For slider, numberInput, rangeSlider
  step?: number; // For slider, numberInput, rangeSlider
  unit?: string; // e.g., "px", "ms"
  isAdvanced?: boolean; // To hide behind an "Advanced" toggle if needed
  isPowerUserOnly?: boolean; // Only visible/editable by power users
  localStorage?: boolean; // Flag to indicate this uses localStorage instead of server sync
  required?: boolean; // Required field indicator
  action?: () => void; // For buttonAction type
}

export interface UISubsectionDefinition {
  label: string;
  description?: string;
  settings: Record<string, UISettingDefinition>;
}

export interface UICategoryDefinition {
  label: string;
  icon?: string; // Lucide icon name
  description?: string;
  subsections: Record<string, UISubsectionDefinition>;
}