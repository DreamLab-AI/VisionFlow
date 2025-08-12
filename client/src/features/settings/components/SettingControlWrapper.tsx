/**
 * Wrapper component that chooses between localStorage or backend-synced setting controls
 */

import React from 'react';
import { SettingControlComponent } from './SettingControlComponent';
import { LocalStorageSettingControl } from './LocalStorageSettingControl';

interface SettingControlWrapperProps {
  setting: any;
  category: string;
  subsection: string;
}

export const SettingControlWrapper: React.FC<SettingControlWrapperProps> = ({ 
  setting, 
  category, 
  subsection 
}) => {
  // Check if this setting should use localStorage
  if (setting.localStorage) {
    return <LocalStorageSettingControl setting={setting} />;
  }
  
  // Otherwise use the regular backend-synced control
  return (
    <SettingControlComponent
      setting={setting}
      category={category}
      subsection={subsection}
    />
  );
};