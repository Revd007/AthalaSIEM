"use client"

import { useEffect, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { useToast } from "@/components/ui/use-toast"
import { AgentConfiguration } from "./AgentConfiguration"
import { agentService } from "@/services/agent-service"
import type { Agent } from "@/types/agent"

export function AgentManagement() {
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [showConfig, setShowConfig] = useState(false)
  const { toast } = useToast()

  const { data: agents, isLoading, error } = useQuery({
    queryKey: ["agents"],
    queryFn: agentService.getAgents,
    refetchInterval: 30000, // Refetch every 30 seconds
  })

  const handleConfigClick = (agent: Agent) => {
    setSelectedAgent(agent)
    setShowConfig(true)
  }

  const handleCloseConfig = () => {
    setSelectedAgent(null)
    setShowConfig(false)
  }

  const handleConfigUpdate = async (agentId: string, config: Partial<Agent>) => {
    try {
      await agentService.configureAgent(agentId, config)
      toast({
        title: "Configuration Updated",
        description: "Agent configuration has been updated successfully.",
      })
      handleCloseConfig()
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update agent configuration.",
        variant: "destructive",
      })
    }
  }

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-6 w-3/4" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-4 w-full mb-4" />
              <Skeleton className="h-4 w-2/3" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          Failed to load agents. Please try again later.
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {agents?.map((agent) => {
          const agentId = agent.id || agent.agentId;
          const lastSeen = agent.lastConnected || agent.lastHeartbeat;
          return (
          <Card key={agentId}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {agent.name}
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleConfigClick(agent)}
              >
                Configure
              </Button>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{agent.status}</div>
              <div className="text-xs text-muted-foreground">
                Last seen: {lastSeen ? new Date(lastSeen).toLocaleString() : 'Never'}
              </div>
              <div className="mt-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">CPU Usage</span>
                  <span className="text-sm font-medium">
                    {agent.cpuUsage ?? 0}%
                  </span>
                </div>
                <Progress value={agent.cpuUsage ?? 0} />
                <div className="flex items-center justify-between">
                  <span className="text-sm">Memory Usage</span>
                  <span className="text-sm font-medium">
                    {agent.memoryUsage ?? 0}%
                  </span>
                </div>
                <Progress value={agent.memoryUsage ?? 0} />
                <div className="flex items-center justify-between">
                  <span className="text-sm">Disk Usage</span>
                  <span className="text-sm font-medium">
                    {agent.diskUsage ?? 0}%
                  </span>
                </div>
                <Progress value={agent.diskUsage ?? 0} />
              </div>
              {agent.alerts?.count && agent.alerts.count > 0 && (
                <Alert variant="destructive" className="mt-4">
                  <AlertTitle>Active Alerts</AlertTitle>
                  <AlertDescription>
                    {agent.alerts.count} active alerts detected
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      <Dialog open={showConfig} onOpenChange={setShowConfig}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Configure Agent</DialogTitle>
          </DialogHeader>
          {selectedAgent && (
            <AgentConfiguration
              agent={selectedAgent}
              onClose={handleCloseConfig}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}