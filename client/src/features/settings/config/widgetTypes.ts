// Extended widget type definitions for comprehensive settings management

export type SettingWidgetType =
  | 'toggle'
  | 'slider'
  | 'numberInput'
  | 'textInput'
  | 'colorPicker'
  | 'select'
  | 'radioGroup' 
  | 'rangeSlider' 
  | 'buttonAction'
  | 'dualColorPicker'; 

export interface UISettingDefinition {
  label: string;
  type: SettingWidgetType;
  path: string; 
  description?: string; 
  options?: Array<{ value: string | number; label: string }>; 
  min?: number; 
  max?: number; 
  step?: number; 
  unit?: string; 
  isAdvanced?: boolean; 
  isPowerUserOnly?: boolean; 
  localStorage?: boolean; 
  required?: boolean; 
  action?: () => void; 
}

export interface UISubsectionDefinition {
  label: string;
  description?: string;
  settings: Record<string, UISettingDefinition>;
}

export interface UICategoryDefinition {
  label: string;
  icon?: string; 
  description?: string;
  subsections: Record<string, UISubsectionDefinition>;
}