'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Checkbox } from '@/components/ui/checkbox'
import { Copy, Download, QrCode, Terminal, ArrowRight, Clock } from 'lucide-react'
import { toast } from 'sonner'
import { agentService } from '@/services/agent-service'
import { QRCodeSVG } from 'qrcode.react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'

interface CollectorConfig {
  id: string;
  name: string;
  type: string;
  enabled: boolean;
}

interface DeploymentToken {
  token: string;
  expiresAt: string;
  downloadUrl: string;
}

export function AgentDeployment() {
  // Installation type state
  const [installType, setInstallType] = useState<string>('windows')
  
  // Agent configuration state
  const [serverUrl, setServerUrl] = useState<string>('')
  const [port, setPort] = useState<number>(443)
  const [agentName, setAgentName] = useState<string>('')
  const [useSSL, setUseSSL] = useState<boolean>(true)
  
  // Collectors configuration state
  const [collectors, setCollectors] = useState<CollectorConfig[]>([
    { id: '1', name: 'Windows Event Logs', type: 'windows', enabled: true },
    { id: '2', name: 'System Metrics', type: 'metrics', enabled: true },
    { id: '3', name: 'File Integrity Monitoring', type: 'fim', enabled: false },
    { id: '4', name: 'Network Monitoring', type: 'network', enabled: false },
  ])
  
  // Deployment token state
  const [token, setToken] = useState<DeploymentToken | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [deployMethod, setDeployMethod] = useState<string>('download')
  
  // Function to handle collector toggle
  const handleCollectorToggle = (id: string, enabled: boolean) => {
    setCollectors(
      collectors.map(collector => 
        collector.id === id ? { ...collector, enabled } : collector
      )
    )
  }
  
  // Function to generate deployment token
  const handleGenerateToken = async () => {
    setIsLoading(true)
    try {
      // Create configuration object
      const agentConfig = {
        serverUrl: serverUrl || window.location.origin,
        port: port || 443,
        name: agentName || `Agent-${Math.floor(Math.random() * 10000)}`,
        useSSL,
        collectors: collectors.filter(c => c.enabled).map(c => c.type)
      }
      
      // Call API to generate token with this configuration
      const response = await agentService.generateDeploymentToken(installType, agentConfig)
      setToken(response)
      toast.success('Deployment token generated successfully')
    } catch (error) {
      console.error('Error generating token:', error)
      toast.error('Failed to generate deployment token')
    } finally {
      setIsLoading(false)
    }
  }
  
  // Function to copy to clipboard
  const copyToClipboard = async (text: string, message: string = 'Copied to clipboard') => {
    try {
      await navigator.clipboard.writeText(text)
      toast.success(message)
    } catch (error) {
      toast.error('Failed to copy to clipboard')
    }
  }
  
  // Function to construct download URL
  const getDownloadUrl = () => {
    if (!token) return ''
    return `${token.downloadUrl}?token=${token.token}&type=${installType}`
  }
  
  // Function to get deployment command
  const getDeploymentCommand = () => {
    if (!token) return ''
    
    if (installType === 'windows') {
      return `powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '${getDownloadUrl()}' -OutFile 'AthalaAgent-Setup.exe'; Start-Process -Wait -FilePath 'AthalaAgent-Setup.exe' -ArgumentList '/quiet', '/norestart', 'TOKEN=${token.token}'"`
    } else {
      return `curl -sSL "${getDownloadUrl()}" | sudo bash -s -- --token=${token.token}`
    }
  }
  
  // Function to calculate expiration time
  const getExpirationTime = () => {
    if (!token) return ''
    
    const expiration = new Date(token.expiresAt)
    const now = new Date()
    const diff = expiration.getTime() - now.getTime()
    
    // Convert to hours
    const hours = Math.floor(diff / (1000 * 60 * 60))
    
    return `Expires in ${hours} hours`
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Agent Deployment</CardTitle>
        <CardDescription>
          Configure and deploy new agents to your systems
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <Tabs defaultValue="configure" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="configure">1. Configure</TabsTrigger>
            <TabsTrigger value="generate">2. Generate Token</TabsTrigger>
            <TabsTrigger value="deploy" disabled={!token}>3. Deploy</TabsTrigger>
          </TabsList>
          
          {/* Configuration Tab */}
          <TabsContent value="configure" className="space-y-4 py-4">
            <div className="space-y-4">
              <div>
                <Label>Select Installer Type</Label>
                <Select
                  value={installType}
                  onValueChange={(value) => setInstallType(value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select installer type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="windows">Windows</SelectItem>
                    <SelectItem value="linux">Linux</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <Separator className="my-4" />
              
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Agent Configuration</h3>
                <p className="text-sm text-muted-foreground">
                  Pre-configure the agent settings
                </p>
              </div>
              
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="serverUrl">Server URL (optional)</Label>
                  <Input
                    id="serverUrl"
                    placeholder="https://your-server.com"
                    value={serverUrl}
                    onChange={(e) => setServerUrl(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Leave blank to use current server
                  </p>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="port">Port</Label>
                  <Input
                    id="port"
                    type="number"
                    placeholder="443"
                    value={port}
                    onChange={(e) => setPort(parseInt(e.target.value))}
                  />
                </div>
              </div>
              
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="agentName">Agent Name (optional)</Label>
                  <Input
                    id="agentName"
                    placeholder="Production-Server-01"
                    value={agentName}
                    onChange={(e) => setAgentName(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Leave blank to generate automatically
                  </p>
                </div>
                
                <div className="flex items-center justify-start space-x-2 pt-8">
                  <Switch
                    id="useSSL"
                    checked={useSSL}
                    onCheckedChange={setUseSSL}
                  />
                  <Label htmlFor="useSSL">Use SSL</Label>
                </div>
              </div>
              
              <Separator className="my-4" />
              
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Enable Collectors</h3>
                <p className="text-sm text-muted-foreground">
                  Select which data to collect from this agent
                </p>
              </div>
              
              <div className="space-y-4">
                {collectors.map((collector) => (
                  <div key={collector.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={`collector-${collector.id}`}
                      checked={collector.enabled}
                      onCheckedChange={(checked: boolean | 'indeterminate') => 
                        handleCollectorToggle(collector.id, checked === true)
                      }
                    />
                    <Label htmlFor={`collector-${collector.id}`}>{collector.name}</Label>
                    {collector.type === 'windows' && installType !== 'windows' && (
                      <Badge variant="outline" className="ml-2">
                        Windows only
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
              
              <div className="pt-4">
                <Button onClick={() => document.querySelector('[value="generate"]')?.dispatchEvent(new Event('click'))}>
                  Next: Generate Token <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </div>
          </TabsContent>
          
          {/* Generate Token Tab */}
          <TabsContent value="generate" className="space-y-4 py-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Generate Deployment Token</h3>
                <p className="text-sm text-muted-foreground">
                  Create a secure token that will be used to authenticate the agent installation
                </p>
              </div>
              
              <div className="bg-muted p-4 rounded-md">
                <h4 className="font-medium mb-2">Configuration Summary</h4>
                <div className="text-sm space-y-1">
                  <p><span className="font-medium">Installer Type:</span> {installType}</p>
                  <p><span className="font-medium">Server URL:</span> {serverUrl || 'Auto (current server)'}</p>
                  <p><span className="font-medium">Port:</span> {port}</p>
                  <p><span className="font-medium">Agent Name:</span> {agentName || 'Auto-generated'}</p>
                  <p><span className="font-medium">Use SSL:</span> {useSSL ? 'Yes' : 'No'}</p>
                  <p><span className="font-medium">Enabled Collectors:</span> {collectors.filter(c => c.enabled).map(c => c.name).join(', ') || 'None'}</p>
                </div>
              </div>
              
              <Button 
                onClick={handleGenerateToken} 
                disabled={isLoading}
                className="w-full"
              >
                {isLoading ? 'Generating...' : 'Generate Deployment Token'}
              </Button>
              
              {token && (
                <div className="pt-4 space-y-4">
                  <div className="p-4 border rounded-md bg-green-50 dark:bg-green-950">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium text-green-700 dark:text-green-300">Token Generated</h4>
                        <p className="text-sm text-green-600 dark:text-green-400">
                          <Clock className="inline-block mr-1 h-3 w-3" />
                          {getExpirationTime()}
                        </p>
                      </div>
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => copyToClipboard(token.token, 'Token copied to clipboard')}
                      >
                        <Copy className="h-4 w-4 mr-1" /> Copy
                      </Button>
                    </div>
                    <div className="mt-2 bg-white dark:bg-gray-800 p-2 rounded font-mono text-xs break-all">
                      {token.token}
                    </div>
                  </div>
                  
                  <Button 
                    onClick={() => document.querySelector('[value="deploy"]')?.dispatchEvent(new Event('click'))}
                    className="w-full"
                  >
                    Next: Deploy Agent <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          </TabsContent>
          
          {/* Deploy Tab */}
          <TabsContent value="deploy" className="space-y-4 py-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Deploy Agent</h3>
                <p className="text-sm text-muted-foreground">
                  Choose how you want to deploy the agent
                </p>
              </div>
              
              <Tabs value={deployMethod} onValueChange={setDeployMethod} className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="download">
                    <Download className="h-4 w-4 mr-2" /> Direct Download
                  </TabsTrigger>
                  <TabsTrigger value="command">
                    <Terminal className="h-4 w-4 mr-2" /> Command Line
                  </TabsTrigger>
                  <TabsTrigger value="qrcode">
                    <QrCode className="h-4 w-4 mr-2" /> QR Code
                  </TabsTrigger>
                </TabsList>
                
                {/* Direct Download */}
                <TabsContent value="download" className="pt-4">
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-center space-y-4">
                        <p>Download the installer and run it on your system</p>
                        <Button onClick={() => window.location.href = getDownloadUrl()}>
                          <Download className="h-4 w-4 mr-2" /> Download {installType === 'windows' ? 'Windows' : 'Linux'} Installer
                        </Button>
                        <p className="text-sm text-muted-foreground">
                          The token is included in the download URL
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                {/* Command Line */}
                <TabsContent value="command" className="pt-4">
                  <Card>
                    <CardContent className="pt-6">
                      <div className="space-y-4">
                        <p>Run this command on your target system:</p>
                        <div className="bg-black text-white p-4 rounded-md font-mono text-sm overflow-x-auto">
                          {getDeploymentCommand()}
                        </div>
                        <Button 
                          variant="outline" 
                          onClick={() => copyToClipboard(getDeploymentCommand(), 'Command copied to clipboard')}
                        >
                          <Copy className="h-4 w-4 mr-2" /> Copy command
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                {/* QR Code */}
                <TabsContent value="qrcode" className="pt-4">
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-center space-y-4">
                        <p>Scan this QR code to download the installer:</p>
                        <div className="flex justify-center py-4">
                          <QRCodeSVG 
                            value={getDownloadUrl()} 
                            size={200}
                            bgColor={"#ffffff"}
                            fgColor={"#000000"}
                            level={"L"}
                            includeMargin={false}
                          />
                        </div>
                        <p className="text-sm text-muted-foreground">
                          This QR code contains the download URL with embedded token
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
              
              <div className="pt-4 bg-muted p-4 rounded-md">
                <h4 className="font-medium mb-2">Next Steps</h4>
                <ol className="list-decimal list-inside space-y-2 text-sm">
                  <li>Install the agent using your preferred method</li>
                  <li>The agent will automatically register with your provided configuration</li>
                  <li>Once registered, the agent will appear in your Agent Management dashboard</li>
                  <li>You can further configure the agent from the dashboard if needed</li>
                </ol>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
} 