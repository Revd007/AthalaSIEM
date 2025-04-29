'use client'

import { Server, Activity, AlertTriangle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { agentService } from '@/services/agent-service'
import { Agent, AgentStatus } from '@/types/agent'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { AlertCircle, RefreshCw, Search } from 'lucide-react'
import { useToast } from '@/components/ui/use-toast'
import type { Device } from '@/types/system-health'

interface DevicesListProps {
  devices: Device[]
  onDeviceClick: (device: Device) => void
  onRefresh: () => Promise<void>
  onSearch: (query: string) => void
  onFilter: (type: string) => void
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

export function DevicesList({ devices, onDeviceClick, onRefresh, onSearch, onFilter }: DevicesListProps) {
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

  const { toast } = useToast()

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

  const [searchQuery, setSearchQuery] = useState('')
  const [selectedType, setSelectedType] = useState('all')
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = async () => {
    try {
      setIsRefreshing(true)
      await onRefresh()
      toast({
        title: 'Devices refreshed',
        description: 'The device list has been updated.',
      })
    } catch (error) {
      toast({
        title: 'Refresh failed',
        description: 'Failed to refresh the device list.',
        variant: 'destructive',
      })
    } finally {
      setIsRefreshing(false)
    }
  }

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    onSearch(query)
  }

  const handleFilter = (type: string) => {
    setSelectedType(type)
    onFilter(type)
  }

  const convertAgentToDevice = (agent: Agent): Device => {
    return {
      id: agent.id,
      name: agent.name,
      type: agent.type,
      status: agent.status,
      lastSeen: agent.lastSeen,
      version: agent.version,
      os: agent.os,
      ip: agent.ip,
      location: agent.location,
      resources: agent.resources
    }
  }

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
    if (devices.length && !devices.includes(agent)) return false
    return true
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Input
            placeholder="Search devices..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="w-64"
          />
          <Select value={selectedType} onValueChange={handleFilter}>
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="collector">Collector</SelectItem>
              <SelectItem value="analyzer">Analyzer</SelectItem>
              <SelectItem value="responder">Responder</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button
          variant="outline"
          size="icon"
          onClick={handleRefresh}
          disabled={isRefreshing}
        >
          <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {devices.map((device) => (
          <Card key={device.id}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>{device.name}</span>
                <Badge variant={device.status === 'online' ? 'success' : 'destructive'}>
                  {device.status}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Type</span>
                  <span className="text-sm font-medium">{device.type}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Version</span>
                  <span className="text-sm font-medium">{device.version}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">OS</span>
                  <span className="text-sm font-medium">{device.os}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">IP</span>
                  <span className="text-sm font-medium">{device.ip}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Location</span>
                  <span className="text-sm font-medium">{device.location}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Last Seen</span>
                  <span className="text-sm font-medium">
                    {new Date(device.lastSeen).toLocaleString()}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}