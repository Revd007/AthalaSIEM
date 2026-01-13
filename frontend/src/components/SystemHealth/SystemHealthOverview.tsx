'use client'

import { Server, Shield, AlertTriangle } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { useQuery } from '@tanstack/react-query'
import { agentService } from '@/services/agent-service'
import { AgentStatus } from '@/types/agent'

export function SystemHealthOverview() {
  const { data: agents, isLoading } = useQuery({
    queryKey: ['agents'],
    queryFn: agentService.getAgents,
    refetchInterval: 30000, // Refetch every 30 seconds
  })

  const calculateMetrics = () => {
    if (!agents) return []

    const totalAgents = agents.length
    const healthyAgents = agents.filter(agent => agent.status === AgentStatus.Online).length
    const warningAgents = agents.filter(agent => agent.status === AgentStatus.Pending).length
    const criticalAgents = agents.filter(agent => 
      agent.status === AgentStatus.Offline || agent.status === AgentStatus.Error
    ).length

    return [
      {
        title: 'Total Agents',
        value: totalAgents.toString(),
        icon: Server,
        color: 'blue' as const,
        loading: isLoading
      },
      {
        title: 'Healthy',
        value: healthyAgents.toString(),
        icon: Shield,
        color: 'green' as const,
        loading: isLoading
      },
      {
        title: 'Warning',
        value: warningAgents.toString(),
        icon: AlertTriangle,
        color: 'yellow' as const,
        loading: isLoading
      },
      {
        title: 'Critical',
        value: criticalAgents.toString(),
        icon: AlertTriangle,
        color: 'red' as const,
        loading: isLoading
      }
    ]
  }

  const metrics = calculateMetrics()

  return (
    <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 lg:gap-6">
      {metrics.map((metric) => (
        <StatsCard key={metric.title} {...metric} />
      ))}
    </div>
  )
} 