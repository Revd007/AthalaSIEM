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
import type { SystemDevice, DeviceHealth } from '@/types/system-health'

interface DeviceDetailsProps {
  deviceId: string
}

// Mock data - replace with actual API call
const mockDevice: SystemDevice = {
  id: '1',
  name: 'Main Firewall',
  type: 'firewall',
  status: 'healthy',
  ipAddress: '192.168.1.1',
  location: 'Main DC',
  lastSeen: '2024-03-19T10:00:00Z',
  agentVersion: '2.0.0',
  manufacturer: 'Palo Alto',
  model: 'PA-3260',
  operatingSystem: 'PAN-OS 10.1.0'
}

// Add these mock data
const mockProcesses = [
  {
    pid: 1234,
    name: 'firewalld',
    cpu: 2.5,
    memory: 256,
    status: 'running',
    user: 'root',
    startTime: '2024-03-19T08:00:00Z'
  },
  {
    pid: 1235,
    name: 'ipsengine',
    cpu: 4.2,
    memory: 512,
    status: 'running',
    user: 'root',
    startTime: '2024-03-19T08:00:00Z'
  },
  {
    pid: 1236,
    name: 'vpnservice',
    cpu: 1.8,
    memory: 128,
    status: 'running',
    user: 'root',
    startTime: '2024-03-19T08:00:00Z'
  }
]

const mockAlerts = [
  {
    id: '1',
    severity: 'warning',
    message: 'High CPU usage detected',
    timestamp: '2024-03-19T09:45:00Z',
    source: 'System Monitor',
    acknowledged: false
  },
  {
    id: '2',
    severity: 'critical',
    message: 'Service restart required',
    timestamp: '2024-03-19T09:30:00Z',
    source: 'Service Monitor',
    acknowledged: true
  },
  {
    id: '3',
    severity: 'info',
    message: 'System update available',
    timestamp: '2024-03-19T09:00:00Z',
    source: 'Update Manager',
    acknowledged: false
  }
]

export function DeviceDetails({ deviceId }: DeviceDetailsProps) {
  // TODO: Replace with actual API call to fetch agent by ID
  // const { data: agent } = useQuery({
  //   queryKey: ['agent', deviceId],
  //   queryFn: () => agentService.getAgentStatus(deviceId)
  // });
  const device = mockDevice // Replace with API call

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
                <p className="mt-1">{device.name}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">IP Address</h3>
                <p className="mt-1">{device.ipAddress}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Location</h3>
                <p className="mt-1">{device.location}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Type</h3>
                <p className="mt-1 capitalize">{device.type}</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium text-gray-500">Manufacturer</h3>
                <p className="mt-1">{device.manufacturer}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Model</h3>
                <p className="mt-1">{device.model}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Operating System</h3>
                <p className="mt-1">{device.operatingSystem}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500">Agent Version</h3>
                <p className="mt-1">{device.agentVersion}</p>
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
              <div className="space-y-4">
                {mockServices.map(service => (
                  <div key={service.name} className="flex items-center justify-between p-2 border rounded">
                    <div className="flex items-center space-x-4">
                      <Play className={`h-4 w-4 ${
                        service.status === 'running' ? 'text-green-500' :
                        service.status === 'stopped' ? 'text-gray-500' : 'text-red-500'
                      }`} />
                      <span>{service.name}</span>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      service.status === 'running' ? 'bg-green-50 text-green-700' :
                      service.status === 'stopped' ? 'bg-gray-50 text-gray-700' : 'bg-red-50 text-red-700'
                    }`}>
                      {service.status}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="processes" className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-sm text-gray-500">
                      <th className="pb-4">PID</th>
                      <th className="pb-4">Name</th>
                      <th className="pb-4">CPU %</th>
                      <th className="pb-4">Memory</th>
                      <th className="pb-4">Status</th>
                      <th className="pb-4">User</th>
                      <th className="pb-4">Start Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {mockProcesses.map(process => (
                      <tr key={process.pid} className="text-sm">
                        <td className="py-3">{process.pid}</td>
                        <td className="py-3">{process.name}</td>
                        <td className="py-3">{process.cpu}%</td>
                        <td className="py-3">{process.memory} MB</td>
                        <td className="py-3">
                          <span className="px-2 py-1 text-xs rounded-full bg-green-50 text-green-700">
                            {process.status}
                          </span>
                        </td>
                        <td className="py-3">{process.user}</td>
                        <td className="py-3">
                          {new Date(process.startTime).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-4">
                {mockAlerts.map(alert => (
                  <div 
                    key={alert.id} 
                    className={`
                      flex items-center justify-between p-4 border rounded
                      ${alert.severity === 'critical' ? 'border-red-200 bg-red-50' :
                        alert.severity === 'warning' ? 'border-yellow-200 bg-yellow-50' :
                        'border-blue-200 bg-blue-50'}
                    `}
                  >
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <AlertTriangle className={`h-4 w-4 ${
                          alert.severity === 'critical' ? 'text-red-500' :
                          alert.severity === 'warning' ? 'text-yellow-500' :
                          'text-blue-500'
                        }`} />
                        <span className="font-medium">{alert.message}</span>
                      </div>
                      <div className="text-sm text-gray-500">
                        <span>{alert.source}</span>
                        <span className="mx-2">•</span>
                        <span>{new Date(alert.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                    {alert.acknowledged ? (
                      <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-600">
                        Acknowledged
                      </span>
                    ) : (
                      <button className="px-2 py-1 text-xs rounded-full bg-white border border-gray-300 hover:bg-gray-50">
                        Acknowledge
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

const mockServices = [
  { name: 'Firewall Service', status: 'running' as const },
  { name: 'IPS Engine', status: 'running' as const },
  { name: 'VPN Service', status: 'running' as const },
  { name: 'Update Service', status: 'stopped' as const }
] 