'use client'

import { Server, Activity, AlertTriangle, Edit2, Trash2, MoreVertical, X } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query'
import { agentService } from '@/services/agent-service'
import { Agent, AgentStatus } from '@/types/agent'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { toast } from 'sonner'
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
        return [];
      }
    },
    refetchInterval: 30000,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    staleTime: 10000,
  });

  // Ensure agents is always an array
  const agents = data as Agent[] || [];

  // Edit/Delete dialog state
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
  const [editForm, setEditForm] = useState({
    name: '',
    hostname: '',
    ipAddress: '',
    isEnabled: true,
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (agentId: string) => agentService.deleteAgent(agentId),
    onSuccess: () => {
      toast.success('Agent deleted successfully');
      queryClient.invalidateQueries({ queryKey: ['agents'] });
      setDeleteDialogOpen(false);
      setEditingAgent(null);
      if (selectedDevice === editingAgent?.id) {
        onDeviceSelect('');
      }
    },
    onError: (error: any) => {
      toast.error(`Failed to delete agent: ${error.message}`);
    },
  });

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: Partial<Agent> }) => 
      agentService.configureAgent(agentId, data),
    onSuccess: () => {
      toast.success('Agent updated successfully');
      queryClient.invalidateQueries({ queryKey: ['agents'] });
      setEditDialogOpen(false);
      setEditingAgent(null);
    },
    onError: (error: any) => {
      toast.error(`Failed to update agent: ${error.message}`);
    },
  });

  const handleEditClick = (agent: Agent, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingAgent(agent);
    setEditForm({
      name: agent.name || '',
      hostname: agent.hostname || '',
      ipAddress: agent.ipAddress || '',
      isEnabled: agent.isEnabled ?? agent.enabled ?? true,
    });
    setEditDialogOpen(true);
  };

  const handleDeleteClick = (agent: Agent, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingAgent(agent);
    setDeleteDialogOpen(true);
  };

  const handleEditSubmit = () => {
    if (!editingAgent?.id) return;
    
    updateMutation.mutate({
      agentId: editingAgent.id,
      data: {
        name: editForm.name,
        hostname: editForm.hostname,
        ipAddress: editForm.ipAddress,
        isEnabled: editForm.isEnabled,
        enabled: editForm.isEnabled,
      },
    });
  };

  const handleDeleteConfirm = () => {
    if (!editingAgent?.id) return;
    
    deleteMutation.mutate(editingAgent.id);
  };

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
      <div className="p-3 sm:p-4">
        <h2 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">Agents ({filteredAgents.length})</h2>
        <ScrollArea className="h-[300px] sm:h-[400px] lg:h-[calc(100vh-350px)]">
          <div className="space-y-2 pr-2 sm:pr-4">
            {filteredAgents.map(agent => {
              const StatusIcon = statusConfig[agent.status]?.icon || Server
              const agentId = agent.id;
              if (!agentId) {
                console.warn('Agent missing ID:', agent);
                return null;
              }
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
                  className={`w-full p-2.5 sm:p-4 rounded-lg border transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    selectedDevice === agentId
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 hover:border-blue-500 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="space-y-1.5 sm:space-y-2 flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
                        <div className="font-medium text-sm sm:text-base truncate">{agent.name}</div>
                        <Badge variant="outline" className="text-[10px] sm:text-xs">
                          {agent.type}
                        </Badge>
                      </div>
                      <div className="text-xs sm:text-sm text-gray-500 flex flex-wrap items-center gap-1 sm:gap-2">
                        <span className="truncate">{agent.ipAddress}</span>
                        <span className="hidden sm:inline">•</span>
                        <span className="truncate">{agent.hostname}</span>
                      </div>
                      <div className="flex items-center">
                        <span 
                          className={`inline-flex items-center px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-full text-[10px] sm:text-xs font-medium ${
                            statusConfig[agent.status]?.bgColor || statusConfig[AgentStatus.Offline].bgColor
                          } ${statusConfig[agent.status]?.color || statusConfig[AgentStatus.Offline].color}`}
                        >
                          <StatusIcon className="w-2.5 h-2.5 sm:w-3 sm:h-3 mr-0.5 sm:mr-1" />
                          {agent.status}
                        </span>
                      </div>
                      {agent.cpuUsage !== undefined && (
                        <div className="grid grid-cols-3 gap-1 sm:gap-4 mt-1 sm:mt-2">
                          <div className="text-[10px] sm:text-sm">
                            <span className="text-gray-500">CPU: </span>
                            <span className={getMetricColor(agent.cpuUsage)}>{agent.cpuUsage}%</span>
                          </div>
                          <div className="text-[10px] sm:text-sm">
                            <span className="text-gray-500">Mem: </span>
                            <span className={getMetricColor(agent.memoryUsage || 0)}>{agent.memoryUsage}%</span>
                          </div>
                          <div className="text-[10px] sm:text-sm">
                            <span className="text-gray-500">Disk: </span>
                            <span className={getMetricColor(agent.diskUsage || 0)}>{agent.diskUsage}%</span>
                          </div>
                        </div>
                      )}
                      <div className="text-[10px] sm:text-xs text-gray-400 flex items-center gap-1">
                        <span>Last:</span>
                        <time dateTime={agent.lastHeartbeat || agent.lastConnected || ''} className="truncate">
                          {agent.lastHeartbeat || agent.lastConnected 
                            ? new Date(agent.lastHeartbeat || agent.lastConnected!).toLocaleString()
                            : 'Never'}
                        </time>
                      </div>
                    </div>
                    
                    {/* Edit/Delete Actions */}
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild onClick={(e: React.MouseEvent) => e.stopPropagation()}>
                        <Button variant="ghost" size="sm" className="h-6 w-6 sm:h-8 sm:w-8 p-0 flex-shrink-0">
                          <MoreVertical className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={(e: React.MouseEvent) => handleEditClick(agent, e)}>
                          <Edit2 className="h-4 w-4 mr-2" />
                          Edit Agent
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={(e: React.MouseEvent) => handleDeleteClick(agent, e)}
                          className="text-red-600 focus:text-red-600"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete Agent
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              )
            })}
          </div>
          </ScrollArea>
      </div>

      {/* Edit Agent Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Edit Agent</DialogTitle>
            <DialogDescription>
              Update the agent configuration. Changes will be applied immediately.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">
                Name
              </Label>
              <Input
                id="name"
                value={editForm.name}
                onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="hostname" className="text-right">
                Hostname
              </Label>
              <Input
                id="hostname"
                value={editForm.hostname}
                onChange={(e) => setEditForm({ ...editForm, hostname: e.target.value })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="ipAddress" className="text-right">
                IP Address
              </Label>
              <Input
                id="ipAddress"
                value={editForm.ipAddress}
                onChange={(e) => setEditForm({ ...editForm, ipAddress: e.target.value })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="enabled" className="text-right">
                Enabled
              </Label>
              <div className="col-span-3 flex items-center space-x-2">
                <Switch
                  id="enabled"
                  checked={editForm.isEnabled}
                  onCheckedChange={(checked) => setEditForm({ ...editForm, isEnabled: checked })}
                />
                <span className="text-sm text-gray-500">
                  {editForm.isEnabled ? 'Agent is active' : 'Agent is disabled'}
                </span>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleEditSubmit} disabled={updateMutation.isPending}>
              {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Agent</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete the agent &quot;{editingAgent?.name}&quot;? 
              This action cannot be undone and will remove all associated data.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteConfirm}
              className="bg-red-600 hover:bg-red-700"
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? 'Deleting...' : 'Delete Agent'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}