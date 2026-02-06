import React, { useEffect, useState } from 'react';
import GraphCanvas from './GraphCanvas';
import { createLogger } from '../../../utils/loggerConfig';

class CanvasErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  state = { hasError: false, error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 20, color: '#ff6b6b', background: '#1a1a2e', borderRadius: 8, margin: 10 }}>
          <h3>3D Rendering Error</h3>
          <p>The graph visualization encountered an error. Try refreshing the page.</p>
          <details><summary>Details</summary><pre>{this.state.error?.message}</pre></details>
        </div>
      );
    }
    return this.props.children;
  }
}

const logger = createLogger('GraphCanvasWrapper');


const detectTestMode = (): boolean => {
    
    if (typeof window !== 'undefined') {
        const params = new URLSearchParams(window.location.search);
        if (params.get('testMode') === 'true' || params.get('bypassWebGL') === 'true') {
            logger.info('Test mode enabled via query parameter');
            return true;
        }
    }

    
    if (typeof navigator !== 'undefined') {
        const userAgent = navigator.userAgent.toLowerCase();

        
        if (userAgent.includes('headless') ||
            userAgent.includes('phantomjs') ||
            userAgent.includes('nightmare') ||
            userAgent.includes('electron')) {
            logger.info('Headless browser detected, enabling test mode');
            return true;
        }

        
        if (userAgent.includes('playwright')) {
            logger.info('Playwright detected, enabling test mode');
            return true;
        }
    }

    
    if (typeof document !== 'undefined') {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

            if (!gl) {
                logger.warn('WebGL not available, enabling test mode');
                return true;
            }

            
            const glContext = gl as WebGL2RenderingContext;
            const debugInfo = glContext.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                const renderer = glContext.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
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

    
    if (typeof process !== 'undefined' && process.env) {
        if (process.env.NODE_ENV === 'test' ||
            process.env.VISIONFLOW_TEST_MODE === 'true' ||
            process.env.BYPASS_WEBGL === 'true') {
            logger.info('Test mode enabled via environment variable');
            return true;
        }
    }

    
    if (typeof window !== 'undefined') {
        
        if (!window.WebGLRenderingContext || !window.WebGL2RenderingContext) {
            logger.warn('WebGL rendering context not available, enabling test mode');
            return true;
        }

        
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


const GraphCanvasWrapper: React.FC = () => {
    return (
        <CanvasErrorBoundary>
            <GraphCanvas />
        </CanvasErrorBoundary>
    );
};

export default GraphCanvasWrapper;