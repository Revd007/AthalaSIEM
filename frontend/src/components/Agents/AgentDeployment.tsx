'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { AlertCircle, Download, Copy, CheckCircle2, XCircle } from 'lucide-react'
import { useToast } from '@/components/ui/use-toast'

interface CollectorConfig {
  id: string
  name: string
  type: string
  status: string
  lastSeen: string
  version: string
  os: string
  ip: string
  location: string
  resources: {
    cpu: number
    memory: number
    disk: number
  }
  logs: Array<{
    timestamp: string
    level: string
    message: string
  }>
  alerts: Array<{
    id: string
    severity: string
    message: string
    timestamp: string
  }>
}

interface DeploymentToken {
  token: string
  expiresAt: string
  downloadUrl: string
}

interface AgentDeploymentProps {
  agentId: string
  onDeploy: (config: CollectorConfig) => Promise<void>
}

export function AgentDeployment({ agentId, onDeploy }: AgentDeploymentProps) {
  const [config, setConfig] = useState<CollectorConfig>({
    id: agentId,
    name: '',
    type: 'collector',
    status: 'pending',
    lastSeen: new Date().toISOString(),
    version: '1.0.0',
    os: '',
    ip: '',
    location: '',
    resources: {
      cpu: 0,
      memory: 0,
      disk: 0
    },
    logs: [],
    alerts: []
  })
  const [deploymentToken, setDeploymentToken] = useState<DeploymentToken | null>(null)
  const [isDeploying, setIsDeploying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { toast } = useToast()

  const handleDeploy = async () => {
    try {
      setIsDeploying(true)
      setError(null)
      await onDeploy(config)
      // Simulate getting deployment token
      setDeploymentToken({
        token: 'mock-token-123',
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
        downloadUrl: 'https://example.com/download'
      })
      toast({
        title: 'Agent deployed successfully',
        description: 'The agent has been deployed and is ready to use.',
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to deploy agent')
      toast({
        title: 'Deployment failed',
        description: 'There was an error deploying the agent.',
        variant: 'destructive',
      })
    } finally {
      setIsDeploying(false)
    }
  }

  const getDownloadUrl = () => {
    if (!deploymentToken) return ''
    return deploymentToken.downloadUrl
  }

  const getDeploymentCommand = () => {
    if (!deploymentToken) return ''
    return `curl -sSL ${deploymentToken.downloadUrl} | bash -s -- --token ${deploymentToken.token}`
  }

  const getExpirationTime = () => {
    if (!deploymentToken) return ''
    return new Date(deploymentToken.expiresAt).toLocaleString()
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    toast({
      title: 'Copied to clipboard',
      description: 'The command has been copied to your clipboard.',
    })
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Deploy Agent</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="name">Agent Name</Label>
                <Input
                  id="name"
                  value={config.name}
                  onChange={(e) => setConfig({ ...config, name: e.target.value })}
                  placeholder="Enter agent name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="type">Agent Type</Label>
                <Select
                  value={config.type}
                  onValueChange={(value) => setConfig({ ...config, type: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="collector">Collector</SelectItem>
                    <SelectItem value="analyzer">Analyzer</SelectItem>
                    <SelectItem value="responder">Responder</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="os">Operating System</Label>
                <Input
                  id="os"
                  value={config.os}
                  onChange={(e) => setConfig({ ...config, os: e.target.value })}
                  placeholder="Enter OS"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="location">Location</Label>
                <Input
                  id="location"
                  value={config.location}
                  onChange={(e) => setConfig({ ...config, location: e.target.value })}
                  placeholder="Enter location"
                />
              </div>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button
              onClick={handleDeploy}
              disabled={isDeploying}
              className="w-full"
            >
              {isDeploying ? 'Deploying...' : 'Deploy Agent'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {deploymentToken && (
        <Card>
          <CardHeader>
            <CardTitle>Deployment Instructions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Deployment Token</p>
                  <p className="text-sm text-muted-foreground">
                    Expires: {getExpirationTime()}
                  </p>
                </div>
                <Badge variant="secondary">{deploymentToken.token}</Badge>
              </div>

              <Tabs defaultValue="command">
                <TabsList>
                  <TabsTrigger value="command">Command</TabsTrigger>
                  <TabsTrigger value="manual">Manual</TabsTrigger>
                </TabsList>
                <TabsContent value="command" className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Input
                      value={getDeploymentCommand()}
                      readOnly
                      className="font-mono text-sm"
                    />
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => copyToClipboard(getDeploymentCommand())}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Run this command on the target system to deploy the agent.
                  </p>
                </TabsContent>
                <TabsContent value="manual" className="space-y-4">
                  <div className="space-y-2">
                    <p className="text-sm font-medium">Download URL</p>
                    <div className="flex items-center space-x-2">
                      <Input
                        value={getDownloadUrl()}
                        readOnly
                        className="font-mono text-sm"
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => copyToClipboard(getDownloadUrl())}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm font-medium">Token</p>
                    <div className="flex items-center space-x-2">
                      <Input
                        value={deploymentToken.token}
                        readOnly
                        className="font-mono text-sm"
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => copyToClipboard(deploymentToken.token)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 