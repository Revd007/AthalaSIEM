'use client'

import { Card } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Users,
  Shield,
  Settings,
  FileText,
  HardDrive,
  Network,
  AlertTriangle,
  AlertCircle,
  RefreshCw,
  Play,
  Square,
  Download,
  Copy,
  Terminal,
  Loader2,
  Pencil,
  Trash,
  Monitor,
  AppleIcon
} from 'lucide-react'
import { useState, useEffect } from 'react'
import { CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { agentService } from '@/services/agent-service'
import { toast } from 'sonner'
import { AgentStatus, NewAgentConfig, Agent } from '@/types/agent'
import { Badge } from '@/components/ui/badge'
import { useQuery, UseQueryOptions } from '@tanstack/react-query'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useRouter } from 'next/navigation'
import { authService } from '@/services/auth-service'
import { Checkbox } from "@/components/ui/checkbox"

interface AgentResponse {
  id: string;
  name: string;
  status: string;
  installationCommand: string;
  message: string;
  apiKey?: string;
}

export function AgentSettings() {
  const router = useRouter()
  const [newAgent, setNewAgent] = useState<NewAgentConfig>({
    name: '',
    hostname: '',
    ipAddress: '',
    port: 514,
    os: ''
  })
  const [selectedOS, setSelectedOS] = useState<'windows' | 'linux-rpm' | 'linux-deb' | 'macos'>('windows')
  const [serverAddress, setServerAddress] = useState('')
  const [rememberAddress, setRememberAddress] = useState(false)
  const [agentName, setAgentName] = useState('')
  const [selectedGroup, setSelectedGroup] = useState('default')
  const [installationCommand, setInstallationCommand] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showInstallDialog, setShowInstallDialog] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [agents, setAgents] = useState<Agent[]>([])
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null)
  const [editForm, setEditForm] = useState<Partial<Agent>>({})
  const [isEditModalOpen, setIsEditModalOpen] = useState(false)
  const [serviceStatus, setServiceStatus] = useState<AgentStatus>(AgentStatus.Offline)

  const queryOptions = {
    queryKey: ['agents'] as const,
    queryFn: () => agentService.getAgents(),
    retry: 1,
    staleTime: 30000,
    enabled: true,
    onError: (err: Error) => {
      if (err.message.includes('401')) {
        router.push('/login');
        toast.error('Please login to continue');
      } else {
        toast.error('Failed to load agents');
      }
    }
  };

  const { data: agentsData, isLoading: queryLoading, error, refetch: refetchAgents } = useQuery(queryOptions);

  // Check service status when agents data changes
  useEffect(() => {
    if (agentsData && agentsData.length > 0) {
      setAgents(agentsData)
      const firstAgentId = agentsData[0].id || agentsData[0].agentId;
      if (firstAgentId) {
        checkServiceStatus(firstAgentId);
      }
    }
  }, [agentsData]);

  const checkServiceStatus = async (agentId: string) => {
    try {
      const response = await agentService.getAgentStatus(agentId);
      setServiceStatus(response.status);
    } catch (error) {
      console.error('Failed to check service status:', error);
      setServiceStatus(AgentStatus.Error);
    }
  };

  const handleServiceAction = async (action: 'start' | 'stop') => {
    try {
      const agentId = agentsData?.[0]?.id || agentsData?.[0]?.agentId;
      if (!agentId) {
        toast.error('No agent selected');
        return;
      }
      await agentService.configureAgent(agentId, { 
        status: action === 'start' ? AgentStatus.Online : AgentStatus.Offline 
      })
      toast.success(`Service ${action}ed successfully`)
      await checkServiceStatus(agentId)
    } catch (error) {
      toast.error(`Failed to ${action} service`)
    }
  }

  const handleCopyCommand = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast.success('Command copied to clipboard')
    } catch (err) {
      toast.error('Failed to copy command')
    }
  }

  const handleDownloadInstaller = async () => {
    try {
      setIsLoading(true);
      await agentService.downloadAgentInstaller(selectedOS);
      toast.success('Installer downloaded successfully');
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to download installer');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddAgent = async (e: React.FormEvent) => {
    e.preventDefault()

    try {
      const token = localStorage.getItem('token')
      if (!token) {
        toast.error('Please login to continue')
        return
      }

      const agentData = {
        ...newAgent,
        port: newAgent.port || 514,
        os: newAgent.os || 'Windows'
      }

      const result = await agentService.addAgent(agentData) as AgentResponse

      if (result.installationCommand && result.apiKey) {
        setInstallationCommand(result.installationCommand)
        setApiKey(result.apiKey)
        setShowInstallDialog(true)
        toast.success('Agent added successfully')
        refetchAgents()
      } else {
        toast.error('Installation information not received')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      toast.error(`Failed to add agent: ${errorMessage}`)
    }
  }

  const handleToggleAgent = async (agentId: string, enabled: boolean) => {
    try {
      await agentService.configureAgent(agentId, { isEnabled: enabled })
      toast.success(`Agent ${enabled ? 'enabled' : 'disabled'} successfully`)
      refetchAgents() // Refresh list
    } catch (error) {
      toast.error('Failed to update agent status')
    }
  }

  const handleEditAgent = (agent: Agent) => {
    setEditingAgent(agent);
    setEditForm({
      name: agent.name,
      hostname: agent.hostname,
      ipAddress: agent.ipAddress,
      port: agent.port,
      os: agent.os,
      isEnabled: agent.isEnabled
    });
    setIsEditModalOpen(true);
  };

  const handleDeleteAgent = async (agent: Agent) => {
    if (window.confirm(`Are you sure you want to delete agent ${agent.name}?`)) {
      setIsLoading(true);
      try {
        const agentId = agent.id || agent.agentId;
        if (!agentId) {
          throw new Error('Agent ID is required');
        }
        await agentService.deleteAgent(agentId);
        setAgents(agents.filter(a => (a.id || a.agentId) !== agentId));
        toast.success(`Agent ${agent.name} deleted successfully`);
      } catch (error) {
        console.error('Error deleting agent:', error);
        toast.error('Failed to delete agent');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleToggleAgentStatus = async (agent: Agent) => {
    setIsLoading(true);
    try {
      const agentId = agent.id || agent.agentId;
      if (!agentId) {
        throw new Error('Agent ID is required');
      }
      const updatedAgent = await agentService.configureAgent(agentId, { isEnabled: !agent.isEnabled });
      const updatedAgentId = updatedAgent.id || updatedAgent.agentId;
      setAgents(agents.map(a => (a.id || a.agentId) === updatedAgentId ? updatedAgent : a));
      toast.success(`Agent ${agent.name} ${updatedAgent.isEnabled ? 'enabled' : 'disabled'} successfully`);
    } catch (error) {
      console.error('Error toggling agent status:', error);
      toast.error('Failed to update agent status');
    } finally {
      setIsLoading(false);
    }
  };

  const generateInstallCommand = (os: string) => {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9595';
    const downloadUrl = `${baseUrl}/api/agents/download/${os}`;
    const defaultName = os === 'windows' ? 'Windows' : '$(hostname)';
    
    switch (os) {
      case 'windows':
        return `Invoke-WebRequest -Uri "${downloadUrl}" -OutFile $(env:tmp)\\athala-agent.msi; msiexec.exe /i $(env:tmp)\\athala-agent.msi /q ATHALA_MANAGER="${serverAddress}" ATHALA_AGENT_GROUP="${selectedGroup}" ATHALA_AGENT_NAME="${agentName || 'Windows'}"`;
      case 'linux-rpm':
        return `curl -o athala-agent.rpm "${downloadUrl}" && sudo rpm -i athala-agent.rpm && sudo ATHALA_MANAGER="${serverAddress}" ATHALA_AGENT_GROUP="${selectedGroup}" ATHALA_AGENT_NAME="${agentName || '`hostname`'}" /etc/init.d/athala-agent start`;
      case 'linux-deb':
        return `curl -o athala-agent.deb "${downloadUrl}" && sudo dpkg -i athala-agent.deb && sudo ATHALA_MANAGER="${serverAddress}" ATHALA_AGENT_GROUP="${selectedGroup}" ATHALA_AGENT_NAME="${agentName || '`hostname`'}" /etc/init.d/athala-agent start`;
      case 'macos':
        return `curl -o athala-agent.pkg "${downloadUrl}" && sudo installer -pkg athala-agent.pkg -target / && sudo ATHALA_MANAGER="${serverAddress}" ATHALA_AGENT_GROUP="${selectedGroup}" ATHALA_AGENT_NAME="${agentName || '`hostname`'}" launchctl load /Library/LaunchDaemons/com.athala.agent.plist`;
      default:
        return '';
    }
  };

  useEffect(() => {
    const command = generateInstallCommand(selectedOS);
    setInstallationCommand(command);
  }, [selectedOS, serverAddress, selectedGroup, agentName]);

  if (queryLoading) {
    return (
      <div className="flex items-center justify-center min-h-[200px]">
        <div className="animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-blue-500" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Deploy New Agent</CardTitle>
          <CardDescription>Download and install the SIEM agent on your systems</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Step 1: Select Package */}
          <div className="space-y-4">
            <h3 className="flex items-center text-lg font-semibold">
              <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-600 text-white text-sm mr-2">1</span>
              Select the package to download and install on your system:
                      </h3>
            <div className="grid grid-cols-3 gap-4">
              {/* Linux Options */}
              <div className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <FileText className="h-6 w-6 mr-2" />
                    <span className="font-medium">LINUX</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="radio"
                      id="rpm-amd64"
                      name="os"
                      value="linux-rpm"
                      checked={selectedOS === 'linux-rpm'}
                      onChange={(e) => setSelectedOS(e.target.value as 'linux-rpm')}
                      className="mr-2"
                    />
                    <label htmlFor="rpm-amd64">RPM amd64</label>
                    </div>
                  <div className="flex items-center">
                    <input
                      type="radio"
                      id="deb-amd64"
                      name="os"
                      value="linux-deb"
                      checked={selectedOS === 'linux-deb'}
                      onChange={(e) => setSelectedOS(e.target.value as 'linux-deb')}
                      className="mr-2"
                    />
                    <label htmlFor="deb-amd64">DEB amd64</label>
                    </div>
                      </div>
                    </div>
                    
              {/* Windows Options */}
              <div className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <Monitor className="h-6 w-6 mr-2" />
                    <span className="font-medium">WINDOWS</span>
                      </div>
                    </div>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="radio"
                      id="windows"
                      name="os"
                      value="windows"
                      checked={selectedOS === 'windows'}
                      onChange={(e) => setSelectedOS(e.target.value as 'windows')}
                      className="mr-2"
                    />
                    <label htmlFor="windows">MSI 32/64 bits</label>
                    </div>
                      </div>
                    </div>
                    
              {/* macOS Options */}
              <div className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <AppleIcon className="h-6 w-6 mr-2" />
                    <span className="font-medium">macOS</span>
                        </div>
                        </div>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="radio"
                      id="macos-intel"
                      name="os"
                      value="macos"
                      checked={selectedOS === 'macos'}
                      onChange={(e) => setSelectedOS(e.target.value as 'macos')}
                      className="mr-2"
                    />
                    <label htmlFor="macos-intel">Intel</label>
                        </div>
                      </div>
                    </div>
                  </div>
            <div className="bg-blue-50 p-4 rounded-lg flex items-start">
              <AlertCircle className="h-5 w-5 text-blue-500 mr-2 mt-0.5" />
              <p className="text-sm text-blue-700">
                For additional systems and architectures, please check our{" "}
                <a href="#" className="text-blue-600 hover:underline">documentation</a>.
              </p>
            </div>
          </div>
          
          {/* Step 2: Download Button */}
          <div className="space-y-4">
            <h3 className="flex items-center text-lg font-semibold">
              <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-600 text-white text-sm mr-2">2</span>
              Download and Install:
            </h3>
            <div className="space-y-4">
              <Button 
                onClick={handleDownloadInstaller}
                disabled={isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="mr-2 h-4 w-4" />
                    Download Agent Installer
                  </>
                )}
              </Button>
              
              <div className="bg-blue-50 p-4 rounded-lg space-y-2">
                <h4 className="font-medium flex items-center">
                  <AlertCircle className="h-4 w-4 mr-2" />
                  Installation Requirements
                </h4>
                <ul className="text-sm text-muted-foreground pl-6 list-disc space-y-1">
                  <li>Administrator privileges are required for installation</li>
                  {selectedOS === 'windows' && <li>Windows 2008 Server or later is required</li>}
                  {selectedOS.includes('linux') && <li>systemd-based Linux distribution is required</li>}
                  {selectedOS === 'macos' && <li>macOS 10.15 or later is required</li>}
                </ul>
              </div>
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={() => setShowInstallDialog(false)}>Close</Button>
          </div>
        </CardContent>
      </Card>

        <Card>
          <CardHeader>
            <CardTitle>Add New Agent</CardTitle>
            <CardDescription>Register a new agent in your SIEM system</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleAddAgent} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Agent Name</Label>
                  <Input
                    id="name"
                    value={newAgent.name}
                    onChange={(e) => setNewAgent({ ...newAgent, name: e.target.value })}
                    required
                  placeholder="Enter agent name"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="hostname">Hostname</Label>
                  <Input
                    id="hostname"
                    value={newAgent.hostname}
                    onChange={(e) => setNewAgent({ ...newAgent, hostname: e.target.value })}
                    required
                  placeholder="Enter hostname"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="ipAddress">IP Address</Label>
                  <Input
                    id="ipAddress"
                    type="text"
                    value={newAgent.ipAddress}
                    onChange={(e) => setNewAgent({ ...newAgent, ipAddress: e.target.value })}
                    required
                  placeholder="Enter IP address"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="port">Port</Label>
                  <Input
                    id="port"
                    type="number"
                    value={newAgent.port}
                    onChange={(e) => setNewAgent({ ...newAgent, port: Number(e.target.value) })}
                    required
                  placeholder="Enter port"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="os">Operating System</Label>
                <Select
                  value={newAgent.os}
                  onValueChange={(value) => setNewAgent({ ...newAgent, os: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select OS" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Windows">Windows</SelectItem>
                    <SelectItem value="Linux">Linux</SelectItem>
                    <SelectItem value="MacOS">MacOS</SelectItem>
                  </SelectContent>
                </Select>
              </div>

            <Button type="submit" className="w-full">Add Agent</Button>
            </form>
          </CardContent>
        </Card>

      <Card>
        <CardHeader>
          <CardTitle>Registered Agents</CardTitle>
          <CardDescription>Manage your registered SIEM agents</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Array.isArray(agentsData) && agentsData.length > 0 ? (
              agentsData.map((agent) => {
                const agentId = agent.id || agent.agentId;
                const lastSeen = agent.lastConnected || agent.lastHeartbeat;
                return (
                <Card key={agentId} className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium">{agent.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        {agent.hostname} ({agent.ipAddress})
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant={agent.status === 'Online' ? 'success' : 'destructive'}>
                          {agent.status}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          Last seen: {lastSeen ? new Date(lastSeen).toLocaleString() : 'Never'}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch
                        checked={agent.status === 'Online'}
                        onCheckedChange={(checked) => {
                          if (agentId) {
                            handleToggleAgent(agentId, checked);
                          }
                        }}
                      />
                      <Button variant="ghost" size="sm" onClick={() => {
                        if (agentId) {
                          checkServiceStatus(agentId);
                        }
                      }}>
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </Card>
                );
              })
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No agents registered yet. Add your first agent above.
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 