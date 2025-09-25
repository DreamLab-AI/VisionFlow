import React, { useEffect, useState } from 'react';
import GraphCanvas from './GraphCanvas';
import GraphCanvasTestMode from './GraphCanvasTestMode';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('GraphCanvasWrapper');

/**
 * Detects if we're running in a testing environment
 * Checks for:
 * - Headless browser indicators
 * - Missing WebGL support
 * - Test mode query parameters
 * - Environment variables
 */
const detectTestMode = (): boolean => {
    // Check for test mode query parameter
    if (typeof window !== 'undefined') {
        const params = new URLSearchParams(window.location.search);
        if (params.get('testMode') === 'true' || params.get('bypassWebGL') === 'true') {
            logger.info('Test mode enabled via query parameter');
            return true;
        }
    }

    // Check for headless browser indicators
    if (typeof navigator !== 'undefined') {
        const userAgent = navigator.userAgent.toLowerCase();

        // Common headless browser indicators
        if (userAgent.includes('headless') ||
            userAgent.includes('phantomjs') ||
            userAgent.includes('nightmare') ||
            userAgent.includes('electron')) {
            logger.info('Headless browser detected, enabling test mode');
            return true;
        }

        // Check for Playwright user agent
        if (userAgent.includes('playwright')) {
            logger.info('Playwright detected, enabling test mode');
            return true;
        }
    }

    // Check for WebGL availability
    if (typeof document !== 'undefined') {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

            if (!gl) {
                logger.warn('WebGL not available, enabling test mode');
                return true;
            }

            // Additional check for software rendering (common in containers)
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                if (renderer && typeof renderer === 'string') {
                    const rendererLower = renderer.toLowerCase();
                    if (rendererLower.includes('swiftshader') ||
                        rendererLower.includes('llvmpipe') ||
                        rendererLower.includes('software') ||
                        rendererLower.includes('mesa')) {
                        logger.info(`Software renderer detected (${renderer}), enabling test mode`);
                        return true;
                    }
                }
            }
        } catch (error) {
            logger.error('Error checking WebGL support:', error);
            return true;
        }
    }

    // Check for test environment variables (if exposed to client)
    if (typeof process !== 'undefined' && process.env) {
        if (process.env.NODE_ENV === 'test' ||
            process.env.VISIONFLOW_TEST_MODE === 'true' ||
            process.env.BYPASS_WEBGL === 'true') {
            logger.info('Test mode enabled via environment variable');
            return true;
        }
    }

    // Check for missing required browser APIs (common in test environments)
    if (typeof window !== 'undefined') {
        // Check if we're in a container or restricted environment
        if (!window.WebGLRenderingContext || !window.WebGL2RenderingContext) {
            logger.warn('WebGL rendering context not available, enabling test mode');
            return true;
        }

        // Check for automation indicators
        if ((window as any).navigator?.webdriver === true ||
            (window as any).__nightmare ||
            (window as any).__selenium_unwrapped ||
            (window as any).callPhantom) {
            logger.info('Automation tool detected, enabling test mode');
            return true;
        }
    }

    return false;
};

/**
 * Wrapper component that automatically switches between
 * full GraphCanvas (with WebGL) and GraphCanvasTestMode
 * based on environment detection
 */
const GraphCanvasWrapper: React.FC = () => {
    const [isTestMode, setIsTestMode] = useState<boolean>(false);
    const [detectionComplete, setDetectionComplete] = useState<boolean>(false);

    useEffect(() => {
        // Perform detection on mount
        const testMode = detectTestMode();
        setIsTestMode(testMode);
        setDetectionComplete(true);

        // Log the decision
        logger.info(`GraphCanvas mode: ${testMode ? 'TEST MODE (WebGL bypassed)' : 'NORMAL MODE (WebGL enabled)'}`);

        // Set a global flag for other components to check
        if (typeof window !== 'undefined') {
            (window as any).__VISIONFLOW_TEST_MODE = testMode;
        }
    }, []);

    // Don't render until detection is complete to avoid flashing
    if (!detectionComplete) {
        return (
            <div style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw',
                height: '100vh',
                backgroundColor: '#000033',
                zIndex: 0
            }} />
        );
    }

    // Render appropriate component based on test mode detection
    return isTestMode ? <GraphCanvasTestMode /> : <GraphCanvas />;
};

export default GraphCanvasWrapper;