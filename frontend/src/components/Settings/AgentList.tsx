 'use client'

import { Card } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { useEffect, useState } from 'react'
import { agentService } from '@/services/agent-service'
import { Agent } from '@/types/agent'
import { RefreshCw, AlertCircle } from 'lucide-react'
import { toast } from 'sonner'
import { AgentStatus } from '@/types/agent'
import { Badge } from '@/components/ui/badge'

export function AgentList() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const fetchAgents = async () => {
    setIsLoading(true)
    try {
      const data = await agentService.getAgents()
      setAgents(data)
    } catch (error) {
      toast.error('Failed to fetch agents')
    } finally {
      setIsLoading(false)
    }
  }

  const handleToggleAgent = async (agentId: string, enabled: boolean) => {
    try {
      await agentService.configureAgent(agentId, { isEnabled: enabled, enabled: enabled })
      toast.success(`Agent ${enabled ? 'enabled' : 'disabled'} successfully`)
      fetchAgents() // Refresh list
    } catch (error) {
      toast.error('Failed to update agent status')
    }
  }

  useEffect(() => {
    fetchAgents()
    // Set up polling for status updates
    const interval = setInterval(fetchAgents, 30000) // Poll every 30 seconds
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Agent List</h3>
        <button 
          onClick={fetchAgents}
          className="p-2 hover:bg-gray-100 rounded-full"
          disabled={isLoading}
        >
          <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="grid gap-4">
        {agents.map((agent) => {
          const agentId = agent.id || agent.agentId;
          const isEnabled = agent.enabled !== undefined ? agent.enabled : agent.isEnabled;
          return (
          <Card key={agentId} className="p-4">
            <div className="flex justify-between items-start">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h4 className="font-medium">{agent.name}</h4>
                  <Badge variant={agent.status === AgentStatus.Online ? 'success' : 'destructive'}>
                    {agent.status}
                  </Badge>
                </div>
                <p className="text-sm text-gray-500">{agent.hostname}</p>
                <p className="text-sm text-gray-500">{agent.ipAddress}</p>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-sm font-medium">Events</div>
                  <div className="text-2xl font-bold">{agent.eventCount || 0}</div>
                </div>
                
                <Switch
                  checked={isEnabled ?? false}
                  onCheckedChange={(checked) => {
                    if (agentId) {
                      handleToggleAgent(agentId, checked);
                    }
                  }}
                />
              </div>
            </div>

            {agent.status === AgentStatus.Error && (
              <div className="mt-2 flex items-center gap-2 text-red-500 text-sm">
                <AlertCircle className="h-4 w-4" />
                <span>Connection error detected</span>
              </div>
            )}
          </Card>
          );
        })}

        {agents.length === 0 && (
          <Card className="p-4 text-center text-gray-500">
            No agents found. Add an agent to get started.
          </Card>
        )}
      </div>
    </div>
  )
}