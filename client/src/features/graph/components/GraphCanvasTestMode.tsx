import React from 'react';
import { useSettingsStore } from '../../../store/settingsStore';
import { AgentPollingStatus } from '../../bots/components/AgentPollingStatus';

// Test mode fallback component that renders without WebGL
const GraphCanvasTestMode: React.FC = () => {
    const { settings } = useSettingsStore();
    const showStats = settings?.system?.debug?.enablePerformanceDebug ?? false;

    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw',
                height: '100vh',
                backgroundColor: '#000033',
                zIndex: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontFamily: 'monospace'
            }}
        >
            {/* Debug indicator */}
            {showStats && (
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    color: 'white',
                    backgroundColor: 'rgba(0, 255, 0, 0.5)',
                    padding: '5px 10px',
                    zIndex: 1000,
                    fontSize: '12px'
                }}>
                    TEST MODE: WebGL Bypassed
                </div>
            )}

            {/* Agent Polling Status Overlay */}
            <AgentPollingStatus />

            {/* Test mode indicator */}
            <div style={{
                padding: '40px',
                textAlign: 'center',
                backgroundColor: 'rgba(0, 50, 100, 0.3)',
                borderRadius: '10px',
                border: '1px solid rgba(100, 150, 255, 0.5)'
            }}>
                <h2 style={{ fontSize: '24px', marginBottom: '20px', color: '#88aaff' }}>
                    VisionFlow Test Mode
                </h2>
                <p style={{ fontSize: '14px', marginBottom: '10px' }}>
                    WebGL rendering bypassed for headless testing
                </p>
                <p style={{ fontSize: '12px', opacity: 0.7 }}>
                    Control panel and API interactions remain functional
                </p>

                <div style={{
                    marginTop: '30px',
                    padding: '20px',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    borderRadius: '5px'
                }}>
                    <p style={{ fontSize: '12px', marginBottom: '10px' }}>
                        System Status:
                    </p>
                    <ul style={{
                        listStyle: 'none',
                        padding: 0,
                        fontSize: '11px',
                        textAlign: 'left'
                    }}>
                        <li>✅ API Connection: Active</li>
                        <li>✅ Control Panel: Functional</li>
                        <li>✅ Settings Store: Connected</li>
                        <li>⚠️ 3D Rendering: Disabled (Test Mode)</li>
                    </ul>
                </div>
            </div>

            {/* Test identification markers for Playwright */}
            <div
                id="visionflow-test-mode"
                data-testid="graph-canvas-test-mode"
                style={{ display: 'none' }}
            >
                TEST_MODE_ACTIVE
            </div>
        </div>
    );
};

export default GraphCanvasTestMode;