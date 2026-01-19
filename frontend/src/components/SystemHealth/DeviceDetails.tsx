'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Server, 
  Clock, 
  Settings, 
  Activity,
  HardDrive,
  Play,
  AlertTriangle
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { agentService } from '@/services/agent-service'
import { useAlerts } from '@/services/alert-service'
import { Skeleton } from '@/components/ui/skeleton'
import type { Agent } from '@/types/agent'
import type { Alert } from '@/types/alert'

interface DeviceDetailsProps {
  deviceId: string
}

export function DeviceDetails({ deviceId }: DeviceDetailsProps) {
  const { data: agent, isLoading: agentLoading, error: agentError } = useQuery({
    queryKey: ['agent', deviceId],
    queryFn: () => agentService.getAgentStatus(deviceId),
    enabled: !!deviceId,
    retry: 1,
  });

  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    agentId: deviceId,
    limit: 10,
    status: 'new'
  });

  const { data: processesData, isLoading: processesLoading } = useQuery({
    queryKey: ['agent-processes', deviceId],
    queryFn: () => agentService.getAgentProcesses(deviceId),
    enabled: !!deviceId,
  });

  if (agentLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (agentError) {
    return (
      <div className="text-center text-red-500 py-8">
        Error loading agent: {agentError instanceof Error ? agentError.message : 'Unknown error'}
      </div>
    );
  }

  if (!agent) {
    return (
      <div className="text-center text-gray-500 py-8">
        Agent not found (ID: {deviceId})
      </div>
    );
  }

  const alerts = alertsData ?? [];
  const processes = processesData ?? [];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Device Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium text-gray-500">Device Name</h3>
                <p className="mt-1">{agent.name}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">IP Address</h3>
                <p className="mt-1">{agent.ipAddress}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Hostname</h3>
                <p className="mt-1">{agent.hostname}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Type</h3>
                <p className="mt-1 capitalize">{agent.type || 'Unknown'}</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium text-gray-500">Operating System</h3>
                <p className="mt-1">{agent.operatingSystem || agent.os || 'Unknown'}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Agent Version</h3>
                <p className="mt-1">{agent.version || 'Unknown'}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Status</h3>
                <p className="mt-1">{agent.status}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Last Connected</h3>
                <p className="mt-1">
                  {agent.lastConnected || agent.lastHeartbeat 
                    ? new Date(agent.lastConnected || agent.lastHeartbeat!).toLocaleString()
                    : 'Never'}
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="services">
        <TabsList>
          <TabsTrigger value="services">Services</TabsTrigger>
          <TabsTrigger value="processes">Processes</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="services" className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center text-gray-500 py-4">
                Service information not available
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="processes" className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              {processesLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : processes.length === 0 ? (
                <div className="text-center text-gray-500 py-4">
                  No process information available
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-sm text-gray-500">
                        <th className="pb-4">PID</th>
                        <th className="pb-4">Name</th>
                        <th className="pb-4">CPU %</th>
                        <th className="pb-4">Memory</th>
                        <th className="pb-4">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {processes.map((process: any) => (
                        <tr key={process.pid} className="text-sm">
                          <td className="py-3">{process.pid}</td>
                          <td className="py-3">{process.name}</td>
                          <td className="py-3">{process.cpuUsage?.toFixed(1) || 0}%</td>
                          <td className="py-3">{process.memoryUsage ? `${(process.memoryUsage / 1024 / 1024).toFixed(0)} MB` : 'N/A'}</td>
                          <td className="py-3">
                            <span className="px-2 py-1 text-xs rounded-full bg-green-50 text-green-700">
                              {process.status || 'running'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              {alertsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : alerts.length === 0 ? (
                <div className="text-center text-gray-500 py-4">
                  No alerts for this agent
                </div>
              ) : (
                <div className="space-y-4">
                  {alerts.map((alert: Alert) => {
                    const severity = (alert.severity?.toLowerCase() || 'low') as 'critical' | 'high' | 'medium' | 'low';
                    const isAcknowledged = alert.status !== 'new';
                    
                    return (
                      <div 
                        key={alert.id} 
                        className={`
                          flex items-center justify-between p-4 border rounded
                          ${severity === 'critical' ? 'border-red-200 bg-red-50' :
                            severity === 'high' ? 'border-yellow-200 bg-yellow-50' :
                            'border-blue-200 bg-blue-50'}
                        `}
                      >
                        <div className="space-y-1">
                          <div className="flex items-center space-x-2">
                            <AlertTriangle className={`h-4 w-4 ${
                              severity === 'critical' ? 'text-red-500' :
                              severity === 'high' ? 'text-yellow-500' :
                              'text-blue-500'
                            }`} />
                            <span className="font-medium">{alert.title || alert.message || 'Alert'}</span>
                          </div>
                          <div className="text-sm text-gray-500">
                            <span>{alert.source || 'System'}</span>
                            <span className="mx-2">•</span>
                            <span>{alert.timestamp ? new Date(alert.timestamp).toLocaleString() : 'Unknown'}</span>
                          </div>
                        </div>
                        {isAcknowledged ? (
                          <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-600">
                            Acknowledged
                          </span>
                        ) : (
                          <button className="px-2 py-1 text-xs rounded-full bg-white border border-gray-300 hover:bg-gray-50">
                            Acknowledge
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 