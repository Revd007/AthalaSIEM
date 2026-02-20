'use client'

import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useHuntDashboard, useHuntBehavior, useIocScan, useLiveHuntStart, useLiveHuntResults } from '@/hooks/useAiData'
import { MitreBarChart } from '@/components/charts/MitreBarChart'
import { ThreatHuntingDashboard } from '@/components/ThreatHunting/ThreatHuntingDashboard'
import { IOCScanner } from '@/components/ThreatHunting/IOCScanner'
import { BehaviorAnalysis } from '@/components/ThreatHunting/BehaviorAnalysis'
import { YARARules } from '@/components/ThreatHunting/YARARules'
import { SIGMARules } from '@/components/ThreatHunting/SIGMARules'
import { ThreatIntelligence } from '@/components/ThreatHunting/ThreatIntelligence'
import { HuntingPlaybooks } from '@/components/ThreatHunting/HuntingPlaybooks'
import { Skeleton } from '@/components/ui/skeleton'
import { format } from 'date-fns'

const TABS = [
  { id: 'dashboard', name: 'Dashboard' },
  { id: 'ioc-scanner', name: 'IOC Scanner' },
  { id: 'behavior', name: 'Behavior Analysis' },
  { id: 'yara', name: 'YARA Rules' },
  { id: 'sigma', name: 'SIGMA Rules' },
  { id: 'intel', name: 'Threat Intel' },
  { id: 'playbooks', name: 'Hunting Playbooks' },
  { id: 'live', name: 'Live Hunting' },
] as const

export default function ThreatHuntingPage() {
  const [activeTab, setActiveTab] = useState<string>('dashboard')

  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Threat Hunting</h1>
      </div>

      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex flex-wrap gap-2">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                border-b-2 py-4 px-1 text-sm font-medium whitespace-nowrap
                ${activeTab === tab.id
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }
              `}
            >
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      <div className="mt-6">
        {activeTab === 'dashboard' && <HuntDashboardTab />}
        {activeTab === 'ioc-scanner' && <IOCScanner />}
        {activeTab === 'behavior' && <HuntBehaviorTab />}
        {activeTab === 'yara' && <YARARules />}
        {activeTab === 'sigma' && <SIGMARules />}
        {activeTab === 'intel' && <ThreatIntelligence />}
        {activeTab === 'playbooks' && <HuntingPlaybooks />}
        {activeTab === 'live' && <LiveHuntTab />}
      </div>
    </div>
  )
}

function HuntDashboardTab() {
  const { data, isLoading, isError } = useHuntDashboard()

  if (isLoading) return <Skeleton className="h-96 w-full rounded-lg" />
  if (isError) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load hunt dashboard.
      </div>
    )
  }

  const d = data ?? {}
  const activity = d.huntActivityLast7Days ?? []
  const recentFindings = d.recentFindings ?? []

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Active Hunts" value={String(d.activeHunts ?? 0)} />
        <MetricCard label="Total Findings" value={(d.totalFindings ?? 0).toLocaleString()} />
        <MetricCard label="Avg Duration" value={String(d.avgHuntDuration ?? 0)} />
        <MetricCard label="Success Rate" value={`${(d.successRate ?? 0).toFixed(1)}%`} />
      </div>
      {activity.length > 0 && (
        <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
          <h3 className="font-semibold mb-4">Hunt Activity (7 days)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={activity.map((a) => ({ ...a, date: a.date ?? '' }))} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value: number) => [value, 'Findings']} />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden">
        <h3 className="font-semibold p-4 border-b border-gray-200 dark:border-gray-700">Recent Findings</h3>
        {recentFindings.length === 0 ? (
          <p className="p-4 text-gray-500">No recent findings.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-gray-50 dark:bg-gray-800/50">
                  <th className="text-left p-3">Description</th>
                  <th className="text-left p-3">Severity</th>
                  <th className="text-left p-3">Time</th>
                </tr>
              </thead>
              <tbody>
                {recentFindings.map((f) => (
                  <tr key={f.id} className="border-b border-gray-100 dark:border-gray-700/50">
                    <td className="p-3">{f.description}</td>
                    <td className="p-3">{f.severity}</td>
                    <td className="p-3 text-gray-500">{f.createdAt ? format(new Date(f.createdAt), 'PPp') : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function HuntBehaviorTab() {
  const { data, isLoading, isError } = useHuntBehavior()

  if (isLoading) return <Skeleton className="h-96 w-full rounded-lg" />
  if (isError) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load behavior data.
      </div>
    )
  }

  const d = data ?? {}
  const mitre = d.mitreTechniqueCounts ?? []

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="font-semibold mb-4">MITRE ATT&CK Technique Counts</h3>
        <MitreBarChart data={mitre} height={320} />
      </div>
      {(d.processBehavior?.length || d.networkBehavior?.length || d.userBehavior?.length) ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {d.processBehavior?.length > 0 && <BehaviorCard title="Process" items={d.processBehavior} />}
          {d.networkBehavior?.length > 0 && <BehaviorCard title="Network" items={d.networkBehavior} />}
          {d.userBehavior?.length > 0 && <BehaviorCard title="User" items={d.userBehavior} />}
        </div>
      ) : (
        <p className="text-gray-500">No process/network/user behavior data. Data comes from AI predictions.</p>
      )}
    </div>
  )
}

function BehaviorCard({ title, items }: { title: string; items: unknown[] }) {
  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <h4 className="font-medium mb-2">{title}</h4>
      <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
        {items.slice(0, 5).map((item: unknown, i: number) => (
          <li key={i}>{typeof item === 'object' ? JSON.stringify(item) : String(item)}</li>
        ))}
      </ul>
    </div>
  )
}

function LiveHuntTab() {
  const [query, setQuery] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const startMutation = useLiveHuntStart()
  const { data: results } = useLiveHuntResults(sessionId)

  const handleStart = async () => {
    if (!query.trim()) return
    try {
      const res = await startMutation.mutateAsync({ query: query.trim(), timeRangeMinutes: 15 })
      setSessionId(res.sessionId)
    } catch {
      setSessionId(null)
    }
  }

  const findings = results?.findings ?? []

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="font-semibold mb-4">Live Hunt</h3>
        <div className="flex gap-2 flex-wrap">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search query (e.g. keyword or pattern)"
            className="flex-1 min-w-[200px] rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-sm"
          />
          <button
            onClick={handleStart}
            disabled={startMutation.isPending}
            className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            {startMutation.isPending ? 'Running…' : 'Run Hunt'}
          </button>
        </div>
        {sessionId && (
          <p className="mt-2 text-sm text-gray-500">
            Session: {sessionId} · Status: {results?.status ?? '—'} · Findings: {results?.findingsCount ?? 0}
          </p>
        )}
      </div>
      {findings.length > 0 && (
        <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden">
          <h3 className="font-semibold p-4 border-b border-gray-200 dark:border-gray-700">Findings</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-gray-50 dark:bg-gray-800/50">
                  <th className="text-left p-3">Description</th>
                  <th className="text-left p-3">Severity</th>
                </tr>
              </thead>
              <tbody>
                {findings.map((f) => (
                  <tr key={f.id} className="border-b border-gray-100 dark:border-gray-700/50">
                    <td className="p-3">{f.description}</td>
                    <td className="p-3">{f.severity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <p className="text-sm text-gray-600 dark:text-gray-400">{label}</p>
      <p className="text-xl font-semibold mt-1 text-gray-900 dark:text-white">{value}</p>
    </div>
  )
}
