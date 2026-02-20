'use client'

import { Brain, Activity, AlertTriangle, Shield, Zap, LineChart, Globe } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AiOverviewCards } from '@/components/dashboard/AiOverviewCards'
import { AnomalyTimeline } from '@/components/charts/AnomalyTimeline'
import { PredictionTimeline } from '@/components/charts/PredictionTimeline'
import { MitreBarChart } from '@/components/charts/MitreBarChart'
import {
  useAiOverview,
  useAiAnomalies,
  useAiBehavior,
  useAiPredictive,
  useAiAutomatedResponse,
  useAiOsint,
} from '@/hooks/useAiData'
import { Skeleton } from '@/components/ui/skeleton'
import { format } from 'date-fns'

export default function AIAnalysisPage() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold tracking-tight">AI Security Analysis</h1>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">
            <Brain className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="anomalies">
            <AlertTriangle className="w-4 h-4 mr-2" />
            Anomaly Detection
          </TabsTrigger>
          <TabsTrigger value="behavior">
            <Activity className="w-4 h-4 mr-2" />
            Behavioral Analytics
          </TabsTrigger>
          <TabsTrigger value="predictive">
            <LineChart className="w-4 h-4 mr-2" />
            Predictive Analysis
          </TabsTrigger>
          <TabsTrigger value="response">
            <Zap className="w-4 h-4 mr-2" />
            Automated Response
          </TabsTrigger>
          <TabsTrigger value="osint">
            <Globe className="w-4 h-4 mr-2" />
            OSINT Analysis
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <AiOverviewCards />
          <LatestInsightsTable />
        </TabsContent>

        <TabsContent value="anomalies" className="space-y-6">
          <AnomalyDetectionTab />
        </TabsContent>

        <TabsContent value="behavior" className="space-y-6">
          <BehaviorTab />
        </TabsContent>

        <TabsContent value="predictive" className="space-y-6">
          <PredictiveTab />
        </TabsContent>

        <TabsContent value="response" className="space-y-6">
          <AutomatedResponseTab />
        </TabsContent>

        <TabsContent value="osint" className="space-y-6">
          <OsintTab />
        </TabsContent>
      </Tabs>
    </div>
  )
}

function LatestInsightsTable() {
  const { data, isLoading, isError } = useAiOverview()

  if (isLoading) return <Skeleton className="h-64 w-full rounded-lg" />
  if (isError || !data?.latestInsights?.length) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Latest Insights</h2>
        <p className="text-gray-500 dark:text-gray-400">No insights yet.</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden">
      <h2 className="text-lg font-semibold p-4 border-b border-gray-200 dark:border-gray-700">Latest Insights</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
              <th className="text-left p-3 font-medium">Class</th>
              <th className="text-left p-3 font-medium">Confidence</th>
              <th className="text-left p-3 font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {data.latestInsights.map((row) => (
              <tr key={row.id} className="border-b border-gray-100 dark:border-gray-700/50">
                <td className="p-3">{row.predictedClass}</td>
                <td className="p-3">{(row.confidence * 100).toFixed(1)}%</td>
                <td className="p-3 text-gray-500">
                  {row.createdAt ? format(new Date(row.createdAt), 'PPp') : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function AnomalyDetectionTab() {
  const { data, isLoading, isError } = useAiAnomalies()

  if (isLoading) return <Skeleton className="h-96 w-full rounded-lg" />
  if (isError) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load anomaly data.
      </div>
    )
  }

  const d = data ?? {}
  const timeline = d.anomalyTimeline24h ?? []
  const anomalies = d.detectedAnomalies ?? []

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Anomaly Score" value={d.anomalyScore?.toFixed(2) ?? '0'} />
        <MetricCard label="Detected Today" value={String(d.detectedToday ?? 0)} />
        <MetricCard label="High Severity" value={String(d.highSeverityAlerts ?? 0)} />
        <MetricCard label="Logs Analyzed" value={(d.totalLogsAnalyzed ?? 0).toLocaleString()} />
      </div>
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="font-semibold mb-4">Anomaly Timeline (24h)</h3>
        <AnomalyTimeline data={timeline} />
      </div>
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden">
        <h3 className="font-semibold p-4 border-b border-gray-200 dark:border-gray-700">Detected Anomalies</h3>
        {anomalies.length === 0 ? (
          <p className="p-4 text-gray-500">No anomalies detected in the last 24h.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-gray-50 dark:bg-gray-800/50">
                  <th className="text-left p-3">Severity</th>
                  <th className="text-left p-3">Score</th>
                  <th className="text-left p-3">Time</th>
                </tr>
              </thead>
              <tbody>
                {anomalies.map((a) => (
                  <tr key={a.id} className="border-b border-gray-100 dark:border-gray-700/50">
                    <td className="p-3">{a.severity}</td>
                    <td className="p-3">{a.score?.toFixed(2)}</td>
                    <td className="p-3 text-gray-500">
                      {a.createdAt ? format(new Date(a.createdAt), 'PPp') : '—'}
                    </td>
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

function BehaviorTab() {
  const { data, isLoading, isError } = useAiBehavior()

  if (isLoading) return <Skeleton className="h-96 w-full rounded-lg" />
  if (isError) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load behavior data.
      </div>
    )
  }

  const d = data ?? {}
  const timeline = d.userActivityTimeline ?? []

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Users Monitored" value={String(d.usersMonitored ?? 0)} />
        <MetricCard label="Anomalies Today" value={String(d.anomaliesToday ?? 0)} />
        <MetricCard label="Avg Risk Score" value={String(d.avgRiskScore ?? 0)} />
      </div>
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="font-semibold mb-4">User Activity Timeline</h3>
        {timeline.length === 0 ? (
          <p className="text-gray-500 py-8 text-center">No activity data.</p>
        ) : (
          <div className="h-64 flex items-center justify-center text-gray-500">
            Timeline: {timeline.length} data points (chart can be wired to userScore/normalScore)
          </div>
        )}
      </div>
    </div>
  )
}

function PredictiveTab() {
  const { data, isLoading, isError } = useAiPredictive()

  if (isLoading) return <Skeleton className="h-96 w-full rounded-lg" />
  if (isError) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load predictive data.
      </div>
    )
  }

  const d = data ?? {}
  const timeline = d.predictionTimeline ?? []
  const predictions = d.activePredictions ?? []

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Active Predictions" value={String(d.activePredictionsCount ?? 0)} />
        <MetricCard label="Critical Alerts" value={String(d.criticalAlerts ?? 0)} />
        <MetricCard label="Total (24h)" value={String(d.totalAlerts24h ?? 0)} />
        <MetricCard label="High Risk" value={String(d.highRiskPredictions ?? 0)} />
      </div>
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="font-semibold mb-4">Prediction Timeline</h3>
        <PredictionTimeline data={timeline} />
      </div>
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden">
        <h3 className="font-semibold p-4 border-b border-gray-200 dark:border-gray-700">Active Predictions</h3>
        {predictions.length === 0 ? (
          <p className="p-4 text-gray-500">No predictions in the last 24h.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-gray-50 dark:bg-gray-800/50">
                  <th className="text-left p-3">Class</th>
                  <th className="text-left p-3">Confidence</th>
                  <th className="text-left p-3">Time</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p) => (
                  <tr key={p.id} className="border-b border-gray-100 dark:border-gray-700/50">
                    <td className="p-3">{p.predictedClass}</td>
                    <td className="p-3">{(p.confidence * 100).toFixed(1)}%</td>
                    <td className="p-3 text-gray-500">
                      {p.createdAt ? format(new Date(p.createdAt), 'PPp') : '—'}
                    </td>
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

function AutomatedResponseTab() {
  const { data, isLoading, isError } = useAiAutomatedResponse()

  if (isLoading) return <Skeleton className="h-64 w-full rounded-lg" />
  if (isError) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load automated response data.
      </div>
    )
  }

  const actions = data?.recentAutomatedActions ?? []

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden">
      <h2 className="text-lg font-semibold p-4 border-b border-gray-200 dark:border-gray-700">Recent Automated Actions</h2>
      {actions.length === 0 ? (
        <p className="p-4 text-gray-500">No playbook executions in the last 7 days.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-gray-50 dark:bg-gray-800/50">
                <th className="text-left p-3">Playbook</th>
                <th className="text-left p-3">Status</th>
                <th className="text-left p-3">Started</th>
                <th className="text-left p-3">Completed</th>
              </tr>
            </thead>
            <tbody>
              {actions.map((a) => (
                <tr key={a.id} className="border-b border-gray-100 dark:border-gray-700/50">
                  <td className="p-3">{a.playbookId}</td>
                  <td className="p-3">{a.status}</td>
                  <td className="p-3 text-gray-500">
                    {a.startedAt ? format(new Date(a.startedAt), 'PPp') : '—'}
                  </td>
                  <td className="p-3 text-gray-500">
                    {a.completedAt ? format(new Date(a.completedAt), 'PPp') : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function OsintTab() {
  const { data, isLoading, isError } = useAiOsint()

  if (isLoading) return <Skeleton className="h-48 w-full rounded-lg" />
  if (isError) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load OSINT data.
      </div>
    )
  }

  const total = data?.totalPredictions ?? 0
  const correlation = data?.osintPredictionCorrelation ?? []

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">OSINT Correlation</h2>
      <p className="text-gray-600 dark:text-gray-400">
        Total predictions (24h): <strong>{total.toLocaleString()}</strong>
      </p>
      {Array.isArray(correlation) && correlation.length > 0 ? (
        <ul className="mt-4 space-y-2">
          {correlation.map((item: unknown, i: number) => (
            <li key={i} className="text-sm text-gray-600 dark:text-gray-400">
              {JSON.stringify(item)}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-4 text-gray-500">No OSINT correlation data yet.</p>
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
