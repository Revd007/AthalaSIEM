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
  Trash
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
  const [installationCommand, setInstallationCommand] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [serviceStatus, setServiceStatus] = useState<AgentStatus>(AgentStatus.Pending)
  const [showInstallDialog, setShowInstallDialog] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [agents, setAgents] = useState<Agent[]>([])
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null)
  const [editForm, setEditForm] = useState<Partial<Agent>>({})
  const [isEditModalOpen, setIsEditModalOpen] = useState(false)

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
      checkServiceStatus(agentsData[0].agentId);
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
      await agentService.configureAgent(agentsData?.[0]?.agentId || '', { 
        status: action === 'start' ? AgentStatus.Online : AgentStatus.Offline 
      })
      toast.success(`Service ${action}ed successfully`)
      if (agentsData?.[0]?.agentId) {
        await checkServiceStatus(agentsData[0].agentId)
      }
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

  const handleDownloadAgent = async (agent: Agent) => {
    setIsLoading(true);
    try {
      await agentService.downloadAgentInstaller(agent.agentId);
      toast.success(`Agent installer for ${agent.name} downloaded successfully`);
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to download agent installer');
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
        // Call API to delete agent
        await agentService.deleteAgent(agent.agentId);
        
        // Remove agent from state
        setAgents(agents.filter(a => a.agentId !== agent.agentId));
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
      const updatedAgent = await agentService.configureAgent(agent.agentId, { isEnabled: !agent.isEnabled });
      
      // Update agent in state
      setAgents(agents.map(a => a.agentId === updatedAgent.agentId ? updatedAgent : a));
      toast.success(`Agent ${agent.name} ${updatedAgent.isEnabled ? 'enabled' : 'disabled'} successfully`);
    } catch (error) {
      console.error('Error toggling agent status:', error);
      toast.error('Failed to update agent status');
    } finally {
      setIsLoading(false);
    }
  };

  if (queryLoading) {
    return (
      <div className="flex items-center justify-center min-h-[200px]">
        <div className="animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-blue-500" />
      </div>
    )
  }

  // if (error) {
  //   return (
  //     <div className="flex flex-col items-center justify-center min-h-[200px] space-y-4">
  //       <AlertCircle className="h-8 w-8 text-red-500" />
  //       <p className="text-sm text-muted-foreground">Failed to load agents</p>
  //       <Button onClick={() => refetchAgents()} variant="outline" size="sm">
  //         <RefreshCw className="h-4 w-4 mr-2" />
  //         Retry
  //       </Button>
  //     </div>
  //   )
  // }

  // if (!agentsData || agentsData.length === 0) {
  //   return (
  //     <div className="flex flex-col items-center justify-center min-h-[200px] space-y-4">
  //       <Shield className="h-8 w-8 text-gray-400" />
  //       <p className="text-sm text-muted-foreground">No agents found. Add your first agent to get started.</p>
  //     </div>
  //   )
  // }

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium">Agent Service Status</h3>
            <Badge variant={
              serviceStatus === AgentStatus.Online ? 'success' :
              serviceStatus === AgentStatus.Offline ? 'destructive' : 'secondary'
            }>
              {serviceStatus}
            </Badge>
          </div>
          <div className="space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleServiceAction('start')}
              disabled={serviceStatus === AgentStatus.Online}
            >
              <Play className="h-4 w-4 mr-2" />
              Start
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleServiceAction('stop')}
              disabled={serviceStatus === AgentStatus.Offline}
            >
              <Square className="h-4 w-4 mr-2" />
              Stop
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                if (agentsData && agentsData.length > 0) {
                  checkServiceStatus(agentsData[0].agentId)
                }
              }}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Agent Deployment</CardTitle>
          <CardDescription>Download and install the SIEM agent on your Windows systems</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col md:flex-row gap-4">
            <Button 
              onClick={() => handleDownloadAgent(agentsData?.[0] || {} as Agent)} 
              disabled={isLoading} 
              className="bg-blue-600 hover:bg-blue-700 text-white flex-1"
              size="lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Downloading...
                </>
              ) : (
                <>
                  <Download className="h-5 w-5 mr-2" />
                  Download Agent Installer
                </>
              )}
            </Button>
            
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="lg" className="flex-1">
                  <Terminal className="h-5 w-5 mr-2" />
                  View Installation Guide
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-3xl">
                <DialogHeader>
                  <DialogTitle className="text-xl">Athala SIEM Agent Installation Guide</DialogTitle>
                  <DialogDescription>
                    Follow these steps to install and configure the SIEM agent on your Windows systems
                  </DialogDescription>
                </DialogHeader>
                <ScrollArea className="h-[500px] rounded-md border p-6 text-sm">
                  <div className="space-y-6">
                    <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-md border border-blue-200 dark:border-blue-800">
                      <h3 className="font-semibold text-blue-800 dark:text-blue-300 flex items-center">
                        <AlertCircle className="h-4 w-4 mr-2" />
                        Prerequisites
                      </h3>
                      <ul className="list-disc pl-5 mt-2 space-y-1 text-sm text-muted-foreground">
                        <li>Windows 10/11 or Windows Server 2016/2019/2022</li>
                        <li>Administrator privileges on the target system</li>
                        <li>Outbound connectivity to the SIEM server</li>
                        <li>.NET Runtime 8.0 or later (will be installed automatically if missing)</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-semibold mb-2 flex items-center">
                        <span className="bg-primary text-primary-foreground rounded-full w-6 h-6 inline-flex items-center justify-center mr-2">1</span>
                        Download the Agent Installer
                      </h3>
                      <p className="text-muted-foreground ml-8">
                        Click the "Download Agent Installer" button to get the latest version of the Athala SIEM Agent installer (AthalaAgent-Setup.exe).
                        The installer is a self-contained executable that includes everything needed to install and configure the agent.
                      </p>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-semibold mb-2 flex items-center">
                        <span className="bg-primary text-primary-foreground rounded-full w-6 h-6 inline-flex items-center justify-center mr-2">2</span>
                        Run the Installer as Administrator
                      </h3>
                      <p className="text-muted-foreground ml-8">
                        Right-click on the downloaded AthalaAgent-Setup.exe file and select "Run as administrator". 
                        This is required because the installer needs to create a Windows service.
                      </p>
                      <div className="ml-8 mt-2 bg-muted p-3 rounded-md">
                        <p className="text-sm font-medium">If you see a Windows security warning:</p>
                        <ol className="list-decimal pl-5 mt-1 text-sm text-muted-foreground">
                          <li>Click "More info" if shown</li>
                          <li>Click "Run anyway" to proceed with the installation</li>
                        </ol>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-semibold mb-2 flex items-center">
                        <span className="bg-primary text-primary-foreground rounded-full w-6 h-6 inline-flex items-center justify-center mr-2">3</span>
                        Configure Server Connection
                      </h3>
                      <p className="text-muted-foreground ml-8">
                        When prompted, enter the URL of your SIEM server. This is typically the same URL you use to access the web interface.
                        For example: <code className="bg-muted px-1 py-0.5 rounded">http://your-siem-server:5135</code>
                      </p>
                      <div className="ml-8 mt-2 bg-muted p-3 rounded-md">
                        <p className="text-sm font-medium">The server URL will be pre-filled if you downloaded the installer from this interface.</p>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-semibold mb-2 flex items-center">
                        <span className="bg-primary text-primary-foreground rounded-full w-6 h-6 inline-flex items-center justify-center mr-2">4</span>
                        Complete Installation
                      </h3>
                      <p className="text-muted-foreground ml-8">
                        Click the "Install Service" button to complete the installation. The agent will be installed as a Windows service 
                        named "Athala SIEM Agent" and will start automatically. The service is configured to start automatically when the system boots.
                      </p>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-semibold mb-2 flex items-center">
                        <span className="bg-primary text-primary-foreground rounded-full w-6 h-6 inline-flex items-center justify-center mr-2">5</span>
                        Verify Installation
                      </h3>
                      <p className="text-muted-foreground ml-8">
                        After installation, you can verify that the agent is running by checking the "Agents" tab in the SIEM web interface.
                        The agent should appear in the list with a status of "Online" within a few minutes.
                      </p>
                      <div className="ml-8 mt-2 bg-muted p-3 rounded-md">
                        <p className="text-sm font-medium">You can also verify the service status in Windows:</p>
                        <ol className="list-decimal pl-5 mt-1 text-sm text-muted-foreground">
                          <li>Open Services (services.msc)</li>
                          <li>Look for "Athala SIEM Agent" in the list</li>
                          <li>Verify that the status is "Running" and the startup type is "Automatic"</li>
                        </ol>
                      </div>
                    </div>
                    
                    <div className="bg-amber-50 dark:bg-amber-950 p-4 rounded-md border border-amber-200 dark:border-amber-800 mt-8">
                      <h3 className="font-semibold text-amber-800 dark:text-amber-300 flex items-center">
                        <AlertTriangle className="h-4 w-4 mr-2" />
                        Troubleshooting
                      </h3>
                      <div className="mt-2 space-y-3">
                        <div>
                          <p className="font-medium">Administrator Privileges:</p>
                          <p className="text-sm text-muted-foreground">If you see "Access denied" errors, make sure you're running the installer as Administrator.</p>
                        </div>
                        <div>
                          <p className="font-medium">Connection Issues:</p>
                          <p className="text-sm text-muted-foreground">If the agent fails to connect, check that the server URL is correct and that the server is running.</p>
                        </div>
                        <div>
                          <p className="font-medium">Firewall Settings:</p>
                          <p className="text-sm text-muted-foreground">Ensure that your firewall allows outbound connections from the agent to the SIEM server.</p>
                        </div>
                        <div>
                          <p className="font-medium">Event Logs:</p>
                          <p className="text-sm text-muted-foreground">Check Windows Event Viewer (Application and System logs) for any error messages related to the Athala SIEM Agent service.</p>
                        </div>
                        <div>
                          <p className="font-medium">Log Files:</p>
                          <p className="text-sm text-muted-foreground">Agent logs are stored in C:\ProgramData\AthalaSIEM\Agent\logs and can provide additional troubleshooting information.</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </ScrollArea>
              </DialogContent>
            </Dialog>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-md border border-blue-200 dark:border-blue-800">
            <h3 className="font-semibold text-blue-800 dark:text-blue-300 flex items-center mb-2">
              <AlertCircle className="h-4 w-4 mr-2" />
              System Requirements
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-muted-foreground">
              <div>
                <p className="font-medium">Operating Systems:</p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Windows 10/11</li>
                  <li>Windows Server 2016/2019/2022</li>
                </ul>
              </div>
              <div>
                <p className="font-medium">Hardware Requirements:</p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>2GB RAM minimum</li>
                  <li>500MB free disk space</li>
                  <li>Network connectivity to SIEM server</li>
                </ul>
              </div>
            </div>
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
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="hostname">Hostname</Label>
                  <Input
                    id="hostname"
                    value={newAgent.hostname}
                    onChange={(e) => setNewAgent({ ...newAgent, hostname: e.target.value })}
                    required
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

      <Dialog open={showInstallDialog} onOpenChange={setShowInstallDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Agent Registration Successful</DialogTitle>
            <DialogDescription>
              Your agent has been registered. Use these details to configure the agent.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>API Key</Label>
              <div className="rounded bg-muted p-4 flex justify-between items-center mt-2">
                <code className="text-sm break-all">{apiKey}</code>
                <Button variant="ghost" size="sm" onClick={() => handleCopyCommand(apiKey)}>
                  <Copy className="h-4 w-4" />
                </Button>
        </div>
            </div>
            <div>
              <Label>Installation Command</Label>
              <div className="rounded bg-muted p-4 flex justify-between items-center mt-2">
                <code className="text-sm break-all">{installationCommand}</code>
                <Button variant="ghost" size="sm" onClick={() => handleCopyCommand(installationCommand)}>
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <Button onClick={() => setShowInstallDialog(false)}>Close</Button>
          </div>
        </DialogContent>
      </Dialog>

      <Card>
        <CardHeader>
          <CardTitle>Registered Agents</CardTitle>
          <CardDescription>Manage your registered SIEM agents</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Array.isArray(agentsData) && agentsData.length > 0 ? (
              agentsData.map((agent) => (
                <Card key={agent.agentId} className="p-4">
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
                          Last seen: {new Date(agent.lastHeartbeat).toLocaleString()}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch
                        checked={agent.status === 'Online'}
                        onCheckedChange={(checked) => handleToggleAgent(agent.agentId, checked)}
                      />
                      <Button variant="ghost" size="sm" onClick={() => checkServiceStatus(agent.agentId)}>
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </Card>
              ))
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