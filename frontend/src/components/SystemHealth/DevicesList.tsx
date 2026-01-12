'use client'

import { Server, Activity, AlertTriangle } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { agentService } from '@/services/agent-service'
import { Agent, AgentStatus } from '@/types/agent'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useState, useEffect } from 'react'

interface DevicesListProps {
  selectedDevice: string | null
  onDeviceSelect: (deviceId: string) => void
  typeFilter: string[]
  searchQuery: string
  statusFilter: string[]
}

const statusConfig: Record<AgentStatus, { color: string; bgColor: string; icon: any }> = {
  [AgentStatus.Online]: { 
    color: 'text-green-500', 
    bgColor: 'bg-green-50',
    icon: Activity 
  },
  [AgentStatus.Active]: { 
    color: 'text-green-500', 
    bgColor: 'bg-green-50',
    icon: Activity 
  },
  [AgentStatus.Pending]: { 
    color: 'text-yellow-500', 
    bgColor: 'bg-yellow-50',
    icon: Server 
  },
  [AgentStatus.Offline]: { 
    color: 'text-gray-500', 
    bgColor: 'bg-gray-50',
    icon: Server 
  },
  [AgentStatus.Error]: { 
    color: 'text-red-500', 
    bgColor: 'bg-red-50',
    icon: AlertTriangle 
  }
}

const getMetricColor = (value: number) => {
  if (value >= 90) return 'text-red-500'
  if (value >= 70) return 'text-yellow-500'
  return 'text-green-500'
}

export function DevicesList({
  selectedDevice,
  onDeviceSelect,
  typeFilter,
  searchQuery,
  statusFilter
}: DevicesListProps) {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ['agents'],
    queryFn: async (): Promise<Agent[]> => {
      console.log('Fetching agents...');
      // Check if the browser is online
      if (!navigator.onLine) {
        console.error('Browser is offline');
        return [];
      }
      
      try {
        const result = await agentService.getAgents();
        console.log('Agents fetched:', result);
        return result;
      } catch (error) {
        console.error('Error fetching agents:', error);
        // Return empty array instead of throwing to prevent component from crashing
        return [];
      }
    },
    refetchInterval: 30000,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    // Add stale time to prevent too frequent refetches
    staleTime: 10000,
  });

  // Ensure agents is always an array
  const agents = data as Agent[] || [];

  // Add online status detection
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    // Update online status
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (!isOnline) {
    return (
      <Card>
        <div className="p-4">
          <div className="text-yellow-500">
            You are currently offline. Agent data cannot be loaded. Please check your internet connection.
          </div>
        </div>
      </Card>
    );
  }

  if (error) {
    console.error('Error in DevicesList:', error);
    return (
      <Card>
        <div className="p-4">
          <div className="text-red-500">
            Error loading agents. Please check your connection and try again.
          </div>
        </div>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card>
        <div className="p-4 space-y-4">
          <h2 className="text-lg font-semibold">Devices</h2>
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="w-full p-4 rounded-lg border">
                <div className="space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                  <div className="flex items-center space-x-2">
                    <Skeleton className="h-5 w-16 rounded-full" />
                    <Skeleton className="h-3 w-24" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>
    )
  }

  const filteredAgents = agents.filter(agent => {
    if (typeFilter.length && !typeFilter.includes(agent.type)) return false
    if (statusFilter.length && !statusFilter.includes(agent.status)) return false
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      return (
        agent.name.toLowerCase().includes(query) ||
        agent.ipAddress.includes(query) ||
        agent.hostname.toLowerCase().includes(query)
      )
    }
    return true
  });

  return (
    <Card>
      <div className="p-4">
        <h2 className="text-lg font-semibold mb-4">Agents ({filteredAgents.length})</h2>
        <ScrollArea className="h-[calc(100vh-300px)]">
          <div className="space-y-2 pr-4">
            {filteredAgents.map(agent => {
              const StatusIcon = statusConfig[agent.status]?.icon || Server
              const agentId = agent.id || agent.agentId;
              return (
                <div
                  key={agentId}
                  onClick={() => onDeviceSelect(agentId)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onDeviceSelect(agentId)
                    }
                  }}
                  role="button"
                  tabIndex={0}
                  aria-selected={selectedDevice === agentId}
                  className={`w-full p-4 rounded-lg border transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    selectedDevice === agentId
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 hover:border-blue-500 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <div className="font-medium">{agent.name}</div>
                        <Badge variant="outline" className="text-xs">
                          {agent.type}
                        </Badge>
                      </div>
                      <div className="text-sm text-gray-500 flex items-center space-x-2">
                        <span>{agent.ipAddress}</span>
                        <span>•</span>
                        <span>{agent.hostname}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span 
                          className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            statusConfig[agent.status]?.bgColor || statusConfig[AgentStatus.Offline].bgColor
                          } ${statusConfig[agent.status]?.color || statusConfig[AgentStatus.Offline].color}`}
                        >
                          <StatusIcon className="w-3 h-3 mr-1" />
                          {agent.status}
                        </span>
                      </div>
                      {agent.cpuUsage !== undefined && (
                        <div className="grid grid-cols-3 gap-4 mt-2">
                          <div className="text-sm">
                            <span className="text-gray-500">CPU: </span>
                            <span className={getMetricColor(agent.cpuUsage)}>{agent.cpuUsage}%</span>
                          </div>
                          <div className="text-sm">
                            <span className="text-gray-500">Memory: </span>
                            <span className={getMetricColor(agent.memoryUsage || 0)}>{agent.memoryUsage}%</span>
                          </div>
                          <div className="text-sm">
                            <span className="text-gray-500">Disk: </span>
                            <span className={getMetricColor(agent.diskUsage || 0)}>{agent.diskUsage}%</span>
                          </div>
                        </div>
                      )}
                      <div className="text-xs text-gray-400 flex items-center space-x-1">
                        <span>Last seen:</span>
                        <time dateTime={agent.lastHeartbeat}>
                          {new Date(agent.lastHeartbeat).toLocaleString()}
                        </time>
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </ScrollArea>
      </div>
    </Card>
  )
}