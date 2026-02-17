'use client'

import React, { useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { agentService } from '@/services/agent-service'
import { logService } from '@/services/log-service'
import { useAlerts, useDeleteAlert } from '@/services/alert-service'
import { useDashboardSummary } from '@/services/analytics-service'
import { Activity, AlertTriangle, Server, Wifi, WifiOff, RefreshCw, X, Radio } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { startSiemHub, getSiemHubConnection } from '@/lib/signalr'
import { StructuredLogCard } from './StructuredLogCard'
import type { Agent } from '@/types/agent'
import type { LogEntry } from '@/types/agent'
import type { Alert } from '@/types/alert'

const POLLING_INTERVAL = 5000
const LOG_PAGE_SIZE = 100

export function RealTimeDashboard() {
  const queryClient = useQueryClient()
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [isAutoRefresh, setIsAutoRefresh] = useState(true)
  const [signalRConnected, setSignalRConnected] = useState(false)
  const [logFilter, setLogFilter] = useState('')
  const [dashboardState, setDashboardState] = useState<any>(null)
  const signalRInitialized = useRef(false)

  // Real aggregated counters from the backend — no client-side counting
  const { data: summary } = useDashboardSummary()

  const { data: agents, isLoading: agentsLoading, error: agentsError } = useQuery({
    queryKey: ['agents'],
    queryFn: () => agentService.getAgents(),
    refetchInterval: isAutoRefresh ? POLLING_INTERVAL : false,
    retry: 3,
    staleTime: 3000,
  })

  const { data: recentLogs, isLoading: logsLoading } = useQuery({
    queryKey: ['recent-logs'],
    queryFn: async () => {
      const result = await logService.getLogs({
        limit: LOG_PAGE_SIZE,
        sortField: 'timestamp',
        sortDirection: 'desc'
      })
      return result.items
    },
    refetchInterval: isAutoRefresh ? POLLING_INTERVAL : false,
    retry: 3,
    staleTime: 2000,
  })

  // Filter logs based on search input (defined after recentLogs)
  const filteredLogs = React.useMemo(() => {
    if (!recentLogs || !logFilter) return recentLogs || []

    const filterLower = logFilter.toLowerCase()
    return recentLogs.filter(log => {
      const message = (log.message || '').toLowerCase()
      const source = (log.source || '').toLowerCase()
      const severity = (log.severity || log.level || '').toLowerCase()
      const agentId = (log.agentId || '').toLowerCase()

      return message.includes(filterLower) ||
             source.includes(filterLower) ||
             severity.includes(filterLower) ||
             agentId.includes(filterLower) ||
             (log.eventId && log.eventId.toString().includes(filterLower)) ||
             (log.processName && log.processName.toLowerCase().includes(filterLower)) ||
             (log.username && log.username.toLowerCase().includes(filterLower)) ||
             (log.ipAddress && log.ipAddress.toLowerCase().includes(filterLower))
    })
  }, [recentLogs, logFilter])

  const { data: alerts, isLoading: alertsLoading } = useAlerts({
    limit: 10,
    status: 'new',
    enabled: true
  })
  const deleteAlertMutation = useDeleteAlert()

  // SignalR: connect once and listen for real-time log pushes
  useEffect(() => {
    if (signalRInitialized.current) return
    signalRInitialized.current = true

    startSiemHub()
      .then((connected) => {
        setSignalRConnected(connected)
        if (connected) {
          const conn = getSiemHubConnection()
          conn.on('ReceiveLogBatch', () => {
            queryClient.invalidateQueries({ queryKey: ['recent-logs'] })
            queryClient.invalidateQueries({ queryKey: ['alerts'] })
            setLastUpdate(new Date())
          })
          conn.on('ReceiveDashboardState', (state: any) => {
            setDashboardState(state)
            setLastUpdate(new Date())
          })
          conn.on('ReceiveAlert', () => {
            queryClient.invalidateQueries({ queryKey: ['alerts'] })
          })
          conn.on('AgentStatusChange', () => {
            queryClient.invalidateQueries({ queryKey: ['agents'] })
          })
          conn.onclose(() => setSignalRConnected(false))
          conn.onreconnected(() => setSignalRConnected(true))
        }
      })
      .catch(() => setSignalRConnected(false))
  }, [queryClient])

  // Polling fallback for when SignalR is not connected
  useEffect(() => {
    if (isAutoRefresh) {
      const interval = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ['agents'] })
        queryClient.invalidateQueries({ queryKey: ['recent-logs'] })
        queryClient.invalidateQueries({ queryKey: ['alerts'] })
        setLastUpdate(new Date())
      }, POLLING_INTERVAL)

      return () => clearInterval(interval)
    }
  }, [isAutoRefresh, queryClient])

  const onlineAgents = agents?.filter(a => a.status === 'Online' || a.status === 'Active') ?? []
  const offlineAgents = agents?.filter(a => a.status === 'Offline') ?? []
  const totalLogsCount = summary?.totalLogs24h ?? recentLogs?.length ?? 0
  const criticalAlerts = alerts?.filter(a => a.severity?.toLowerCase() === 'critical') ?? []

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffSecs = Math.floor(diffMs / 1000)
    const diffMins = Math.floor(diffSecs / 60)

    if (diffSecs < 10) return 'Just now'
    if (diffSecs < 60) return `${diffSecs}s ago`
    if (diffMins < 60) return `${diffMins}m ago`
    return date.toLocaleTimeString()
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Real-Time SIEM Dashboard</h1>
        <div className="flex items-center gap-4">
          {signalRConnected && (
            <Badge variant="outline" className="flex items-center gap-1 text-emerald-600 border-emerald-300">
              <Radio className="h-3 w-3" />
              Real-Time
            </Badge>
          )}
          <Badge variant={isAutoRefresh ? "default" : "secondary"} className="flex items-center gap-2">
            {isAutoRefresh ? (
              <>
                <Activity className="h-3 w-3 animate-pulse" />
                Live
              </>
            ) : (
              <>
                <WifiOff className="h-3 w-3" />
                Paused
              </>
            )}
          </Badge>
          <button
            onClick={() => setIsAutoRefresh(!isAutoRefresh)}
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
          >
            {isAutoRefresh ? 'Pause' : 'Resume'}
          </button>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Updated: {formatTime(lastUpdate)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {agentsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <>
                <div className="text-2xl font-bold">{agents?.length ?? 0}</div>
                <p className="text-xs text-muted-foreground">
                  {onlineAgents.length} online, {offlineAgents.length} offline
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Online Agents</CardTitle>
            <Wifi className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            {agentsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <>
                <div className="text-2xl font-bold text-green-600">{onlineAgents.length}</div>
                <p className="text-xs text-muted-foreground">
                  {agents && agents.length > 0 
                    ? `${Math.round((onlineAgents.length / agents.length) * 100)}% active`
                    : 'No agents'}
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Events (24h)</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {logsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <>
                <div className="text-2xl font-bold">
                  {totalLogsCount > 1000 ? `${(totalLogsCount / 1000).toFixed(1)}K` : totalLogsCount}
                </div>
                <p className="text-xs text-muted-foreground">
                  {summary?.eventsPerSecond ?? 0}/s &middot; {summary?.criticalCount ?? 0} critical
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Critical Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            {alertsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <>
                <div className="text-2xl font-bold text-red-600">{criticalAlerts.length}</div>
                <p className="text-xs text-muted-foreground">
                  {alerts?.length ?? 0} total alerts
                </p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Agent Status</CardTitle>
          </CardHeader>
          <CardContent>
            {agentsLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map(i => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : agentsError ? (
              <div className="text-center text-red-500 py-4">
                Error loading agents
              </div>
            ) : !agents || agents.length === 0 ? (
              <div className="text-center text-gray-500 py-4">
                No agents registered
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {agents.map(agent => {
                  const isOnline = agent.status === 'Online' || agent.status === 'Active'
                  const lastHeartbeat = agent.lastConnected || agent.lastHeartbeat
                  const isRecent = lastHeartbeat 
                    ? (new Date().getTime() - new Date(lastHeartbeat).getTime()) < 60000
                    : false

                  return (
                    <div
                      key={agent.id}
                      className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <div className="flex items-center gap-3">
                        {isOnline && isRecent ? (
                          <Wifi className="h-4 w-4 text-green-500" />
                        ) : (
                          <WifiOff className="h-4 w-4 text-gray-400" />
                        )}
                        <div>
                          <div className="font-medium">{agent.name}</div>
                          <div className="text-xs text-gray-500">{agent.hostname} • {agent.ipAddress}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant={isOnline ? "default" : "secondary"} className="mb-1">
                          {agent.status}
                        </Badge>
                        {lastHeartbeat && (
                          <div className="text-xs text-gray-500">
                            {formatTimestamp(lastHeartbeat)}
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Recent Logs Stream</CardTitle>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Filter by severity, source, or search..."
                  className="px-3 py-1 text-sm border rounded-md w-64"
                  value={logFilter}
                  onChange={(e) => setLogFilter(e.target.value)}
                />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {logsLoading ? (
              <div className="space-y-2">
                {[1, 2, 3, 4, 5].map(i => (
                  <Skeleton key={i} className="h-24 w-full" />
                ))}
              </div>
            ) : !filteredLogs || filteredLogs.length === 0 ? (
              <div className="text-center text-gray-500 py-4">
                {recentLogs && recentLogs.length > 0 
                  ? `No logs match filter "${logFilter}"`
                  : 'No recent logs'}
              </div>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {filteredLogs.map(log => (
                  <StructuredLogCard key={log.id} log={log} />
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Alerts Feed</CardTitle>
          </CardHeader>
          <CardContent>
            {alertsLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map(i => (
                  <Skeleton key={i} className="h-20 w-full" />
                ))}
              </div>
            ) : !alerts || alerts.length === 0 ? (
              <div className="text-center text-gray-500 py-4">
                No alerts
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {alerts.map(alert => {
                  const severity = (alert.severity?.toLowerCase() || 'low') as 'low' | 'medium' | 'high' | 'critical'
                  const severityColors = {
                    critical: 'border-red-500 bg-red-50 dark:bg-red-950',
                    high: 'border-orange-500 bg-orange-50 dark:bg-orange-950',
                    medium: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950',
                    low: 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                  }

                  return (
                    <div
                      key={alert.id}
                      className={`p-3 border-l-4 rounded ${severityColors[severity] || severityColors.low}`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <AlertTriangle className={`h-4 w-4 shrink-0 ${
                              severity === 'critical' ? 'text-red-600' :
                              severity === 'high' ? 'text-orange-600' :
                              severity === 'medium' ? 'text-yellow-600' :
                              'text-blue-600'
                            }`} />
                            <span className="font-medium truncate">{alert.title || alert.message}</span>
                            <Badge variant={
                              severity === 'critical' ? "destructive" :
                              severity === 'high' ? "default" : "secondary"
                            }>
                              {alert.severity}
                            </Badge>
                          </div>
                          {alert.description && (
                            <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                              {alert.description}
                            </p>
                          )}
                          <div className="text-xs text-gray-500">
                            {alert.timestamp && formatTimestamp(alert.timestamp)} • {alert.source || 'System'}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="shrink-0 h-8 w-8 text-gray-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950"
                          onClick={() => deleteAlertMutation.mutate(alert.id)}
                          disabled={deleteAlertMutation.isPending}
                          title="Hapus alert"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
