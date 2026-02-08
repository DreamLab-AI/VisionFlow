/**
 * Health Dashboard
 *
 * Monitors system health via consolidated health API:
 * - Overall system status
 * - Component health (database, graph, physics, websocket)
 * - Physics simulation status
 * - MCP relay status and control
 * - Real-time health updates
 */

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Label } from '@/features/design-system/components/Label';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { useToast } from '@/features/design-system/components/Toast';
import { Activity, CheckCircle, XCircle, AlertCircle, RefreshCw, Play, FileText, Zap } from 'lucide-react';
import { useHealthService } from '../hooks/useHealthService';

interface HealthDashboardProps {
  className?: string;
}

export function HealthDashboard({ className }: HealthDashboardProps) {
  const { toast } = useToast();
  const { overallHealth, physicsHealth, loading, error, startMCPRelay, getMCPLogs, refreshHealth } =
    useHealthService();

  const [mcpLogs, setMcpLogs] = useState<string | null>(null);

  const handleStartMCPRelay = async () => {
    try {
      const result = await startMCPRelay();
      toast({
        title: 'MCP Relay Started',
        description: result.message || 'MCP relay has been started',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Start Failed',
        description: err.message || 'Failed to start MCP relay',
        variant: 'destructive',
      });
    }
  };

  const handleGetMCPLogs = async () => {
    try {
      const logs = await getMCPLogs();
      setMcpLogs(logs);
      toast({
        title: 'Logs Retrieved',
        description: 'MCP logs have been loaded',
        variant: 'default',
      });
    } catch (err: any) {
      toast({
        title: 'Logs Failed',
        description: err.message || 'Failed to get MCP logs',
        variant: 'destructive',
      });
    }
  };

  const handleRefresh = async () => {
    await refreshHealth();
    toast({
      title: 'Health Refreshed',
      description: 'System health status has been updated',
      variant: 'default',
    });
  };

  const StatusIcon = ({ healthy }: { healthy: boolean }) => {
    return healthy ? (
      <CheckCircle className="h-4 w-4 text-green-500" />
    ) : (
      <XCircle className="h-4 w-4 text-destructive" />
    );
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Health Monitor
            </CardTitle>
            <CardDescription>Real-time system health and component status</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={loading}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Overall Status */}
        {overallHealth && (
          <div className="rounded-lg border p-4 space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-base font-medium">Overall Status</Label>
              <div className="flex items-center gap-2">
                <StatusIcon healthy={overallHealth.healthy} />
                <Badge variant={overallHealth.healthy ? 'default' : 'destructive'}>
                  {overallHealth.healthy ? 'HEALTHY' : 'UNHEALTHY'}
                </Badge>
              </div>
            </div>

            {overallHealth.version && (
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Version:</span>
                <span className="font-medium">{overallHealth.version}</span>
              </div>
            )}

            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Last Check:</span>
              <span className="font-medium">{new Date(overallHealth.timestamp).toLocaleString()}</span>
            </div>
          </div>
        )}

        {/* Component Health */}
        {overallHealth?.components && Object.keys(overallHealth.components).length > 0 && (
          <div className="space-y-3">
            <Label className="text-sm font-medium">Component Health</Label>
            <div className="rounded-lg border divide-y">
              {Object.entries(overallHealth.components).map(([name, healthy]) => (
                <div key={name} className="flex items-center justify-between p-3">
                  <div className="flex items-center gap-2">
                    <StatusIcon healthy={healthy as boolean} />
                    <span className="text-sm font-medium capitalize">{name.replace('_', ' ')}</span>
                  </div>
                  <Badge variant={healthy ? 'default' : 'destructive'} className="text-xs">
                    {healthy ? 'OK' : 'FAILED'}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Physics Simulation Health */}
        {physicsHealth && (
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Physics Simulation
            </Label>
            <div className="rounded-lg border p-3 space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Status:</span>
                <Badge variant={physicsHealth.running ? 'default' : 'secondary'}>
                  {physicsHealth.running ? 'Running' : 'Stopped'}
                </Badge>
              </div>

              {physicsHealth.simulation_id && (
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Simulation ID:</span>
                  <span className="font-medium font-mono text-xs">{physicsHealth.simulation_id}</span>
                </div>
              )}

              {physicsHealth.statistics && (
                <>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Total Steps:</span>
                    <span className="font-medium">{physicsHealth.statistics.total_steps}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Avg Step Time:</span>
                    <span className="font-medium">
                      {physicsHealth.statistics.average_step_time_ms.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">GPU Memory:</span>
                    <span className="font-medium">{physicsHealth.statistics.gpu_memory_used_mb.toFixed(1)} MB</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* MCP Relay */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">MCP Relay</Label>
          <div className="flex gap-2">
            <Button onClick={handleStartMCPRelay} disabled={loading} size="sm">
              <Play className="mr-2 h-4 w-4" />
              Start Relay
            </Button>
            <Button onClick={handleGetMCPLogs} disabled={loading} variant="outline" size="sm">
              <FileText className="mr-2 h-4 w-4" />
              View Logs
            </Button>
          </div>

          {mcpLogs && (
            <div className="rounded-lg border p-3 bg-muted">
              <Label className="text-xs font-medium mb-2 block">Logs:</Label>
              <pre className="text-xs font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                {mcpLogs || 'No logs available'}
              </pre>
            </div>
          )}
        </div>

        {error && (
          <div className="rounded-lg border border-destructive bg-destructive/10 p-3 flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-destructive">Error</p>
              <p className="text-sm text-destructive/90">{error}</p>
            </div>
          </div>
        )}

        {!overallHealth && !error && (
          <div className="flex items-center justify-center p-8 text-muted-foreground">
            <Activity className="mr-2 h-4 w-4 animate-pulse" />
            <span>Loading health status...</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
