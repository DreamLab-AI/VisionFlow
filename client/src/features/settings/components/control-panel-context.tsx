import React, { createContext, useContext, useState, ReactNode } from 'react';

interface ControlPanelContextType {
  advancedMode: boolean;
  toggleAdvancedMode: () => void;
}

const defaultContext: ControlPanelContextType = {
  advancedMode: false,
  toggleAdvancedMode: () => {},
};

const ControlPanelContext = createContext<ControlPanelContextType>(defaultContext);

interface ControlPanelProviderProps {
  children: ReactNode;
}

export const ControlPanelProvider: React.FC<ControlPanelProviderProps> = ({ children }) => {
  const [advancedMode, setAdvancedMode] = useState(false);

  const toggleAdvancedMode = () => {
    setAdvancedMode(prev => !prev);
  };

  return (
    <ControlPanelContext.Provider value={{ advancedMode, toggleAdvancedMode }}>
      {children}
    </ControlPanelContext.Provider>
  );
};

export const useControlPanelContext = (): ControlPanelContextType => {
  const context = useContext(ControlPanelContext);
  if (!context) {
    
    
    return defaultContext;
  }
  return context;
};
