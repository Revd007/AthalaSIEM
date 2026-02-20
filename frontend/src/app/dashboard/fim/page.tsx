'use client'

import { useState, useMemo } from 'react'
import { FIMStatsCards } from '@/components/FIM/FIMStatsCards'
import { FIMEventTable } from '@/components/FIM/FIMEventTable'
import { FIMRulesList } from '@/components/FIM/FIMRulesList'
import { FIMBaselineManager } from '@/components/FIM/FIMBaselineManager'
import type { FIMEventTableFilters } from '@/components/FIM/FIMEventTable'
import type { FIMEvent } from '@/types/fim'
import { useFIMStats } from '@/services/fim-service'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'

const TIME_RANGES = [
  { value: '24h', label: 'Last 24 hours', days: 1 },
  { value: '7d', label: 'Last 7 days', days: 7 },
  { value: '30d', label: 'Last 30 days', days: 30 },
] as const

export default function FIMDashboardPage() {
  const [agentId, setAgentId] = useState<string | undefined>(undefined)
  const [timeRange, setTimeRange] = useState<string>('7d')
  const [severity, setSeverity] = useState<string | undefined>(undefined)
  const [changeType, setChangeType] = useState<string | undefined>(undefined)
  const [acknowledged, setAcknowledged] = useState<boolean | undefined>(undefined)
  const [selectedEvent, setSelectedEvent] = useState<FIMEvent | null>(null)

  const days = useMemo(
    () => TIME_RANGES.find((r) => r.value === timeRange)?.days ?? 7,
    [timeRange]
  )

  const { data: stats } = useFIMStats(agentId, days)
  const agentOptions = useMemo(() => {
    const list = stats?.eventsByAgent ?? []
    return [{ agentId: 'all', agentName: 'All agents' }, ...list]
  }, [stats?.eventsByAgent])

  const filters: FIMEventTableFilters = useMemo(() => {
    const end = new Date()
    const start = new Date()
    start.setDate(start.getDate() - days)
    return {
      agentId: agentId || undefined,
      severity: severity || undefined,
      changeType: changeType || undefined,
      acknowledged,
      startDate: start.toISOString(),
      endDate: end.toISOString(),
    }
  }, [agentId, severity, changeType, acknowledged, days])

  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          File Integrity Monitoring (FIM)
        </h1>
      </div>

      <FIMStatsCards agentId={agentId} days={days} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>FIM events</CardTitle>
              <div className="flex flex-wrap gap-4 pt-2">
                <div className="space-y-1">
                  <Label className="text-xs">Agent</Label>
                  <Select
                    value={agentId ?? 'all'}
                    onValueChange={(v) => setAgentId(v === 'all' ? undefined : v)}
                  >
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="All agents" />
                    </SelectTrigger>
                    <SelectContent>
                      {agentOptions.map((a) => (
                        <SelectItem key={a.agentId} value={a.agentId}>
                          {a.agentName || a.agentId}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Time range</Label>
                  <Select value={timeRange} onValueChange={setTimeRange}>
                    <SelectTrigger className="w-[140px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {TIME_RANGES.map((r) => (
                        <SelectItem key={r.value} value={r.value}>
                          {r.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Severity</Label>
                  <Select
                    value={severity ?? 'all'}
                    onValueChange={(v) => setSeverity(v === 'all' ? undefined : v)}
                  >
                    <SelectTrigger className="w-[120px]">
                      <SelectValue placeholder="All" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      <SelectItem value="Critical">Critical</SelectItem>
                      <SelectItem value="High">High</SelectItem>
                      <SelectItem value="Medium">Medium</SelectItem>
                      <SelectItem value="Low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Status</Label>
                  <Select
                    value={
                      acknowledged === undefined
                        ? 'all'
                        : acknowledged
                          ? 'ack'
                          : 'new'
                    }
                    onValueChange={(v) =>
                      setAcknowledged(
                        v === 'all' ? undefined : v === 'ack'
                      )
                    }
                  >
                    <SelectTrigger className="w-[120px]">
                      <SelectValue placeholder="All" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      <SelectItem value="new">New</SelectItem>
                      <SelectItem value="ack">Acknowledged</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <FIMEventTable
                filters={filters}
                pageSize={20}
                onViewDetails={setSelectedEvent}
              />
            </CardContent>
          </Card>
        </div>
        <div className="space-y-4">
          <FIMRulesList />
          <FIMBaselineManager />
          {selectedEvent && (
            <Card>
              <CardHeader>
                <CardTitle>Event details</CardTitle>
                <button
                  type="button"
                  className="text-sm text-muted-foreground hover:underline"
                  onClick={() => setSelectedEvent(null)}
                >
                  Close
                </button>
              </CardHeader>
              <CardContent className="text-sm space-y-2">
                <p><span className="font-medium">Path:</span> {selectedEvent.filePath}</p>
                <p><span className="font-medium">Change:</span> {selectedEvent.changeType}</p>
                <p><span className="font-medium">Severity:</span> {selectedEvent.severity}</p>
                {selectedEvent.baselineHash != null && (
                  <p><span className="font-medium">Baseline hash:</span> {selectedEvent.baselineHash}</p>
                )}
                {selectedEvent.currentHash != null && (
                  <p><span className="font-medium">Current hash:</span> {selectedEvent.currentHash}</p>
                )}
                {selectedEvent.details && (
                  <p><span className="font-medium">Details:</span> {selectedEvent.details}</p>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
