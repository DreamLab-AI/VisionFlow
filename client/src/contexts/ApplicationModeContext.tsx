import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext, useState, useEffect } from 'react';
import { createLogger } from '../utils/loggerConfig';
const logger = createLogger('ApplicationModeContext');
const defaultContext = {
    mode: 'desktop',
    previousMode: null,
    isXRMode: false,
    isMobileView: false,
    setMode: () => { },
    layoutSettings: {
        showPanels: true,
        showViewport: true,
        showControls: true
    }
};
// Create the context
const ApplicationModeContext = createContext(defaultContext);

export const ApplicationModeProvider = ({ children }) => {
    const [mode, setMode] = useState('desktop');
    const [previousMode, setPreviousMode] = useState(null);
    const [isMobileView, setIsMobileView] = useState(false);
    
    useEffect(() => {
        const handleResize = () => {
            const isMobile = window.innerWidth < 768; 
            setIsMobileView(isMobile);
            
            
            if (isMobile && mode !== 'xr') {
                setMode('mobile');
            }
            else if (!isMobile && mode === 'mobile') {
                setMode('desktop');
            }
        };
        
        handleResize();
        
        window.addEventListener('resize', handleResize);
        
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, [mode]);
    
    const handleModeChange = (newMode) => {
        logger.info(`Changing mode: ${mode} -> ${newMode}`);
        setPreviousMode(mode);
        setMode(newMode);
    };
    
    const getLayoutSettings = () => {
        switch (mode) {
            case 'desktop':
                return {
                    showPanels: true,
                    showViewport: true,
                    showControls: true
                };
            case 'mobile':
                return {
                    showPanels: true,
                    showViewport: true,
                    showControls: true
                };
            case 'xr':
                return {
                    showPanels: false,
                    showViewport: true,
                    showControls: false
                };
            default:
                return {
                    showPanels: true,
                    showViewport: true,
                    showControls: true
                };
        }
    };
    const contextValue = {
        mode,
        previousMode,
        isXRMode: mode === 'xr',
        isMobileView,
        setMode: handleModeChange,
        layoutSettings: getLayoutSettings()
    };
    return (_jsx(ApplicationModeContext.Provider, { value: contextValue, children: children }));
};

export const useApplicationMode = () => {
    const context = useContext(ApplicationModeContext);
    if (!context) {
        throw new Error('useApplicationMode must be used within an ApplicationModeProvider');
    }
    return context;
};
