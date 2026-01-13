'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Zap, Check, AlertTriangle, Clock, Settings, Shield, Activity, RefreshCw } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { 
  useAutomatedActions, 
  useAutomatedRules, 
  useAutomatedResponseStats,
  useAutomatedResponseMetrics,
  type AutomatedAction 
} from '@/services/automated-response-service'
import { Skeleton } from '@/components/ui/skeleton'

export function AutomatedResponse() {
  const [selectedAction, setSelectedAction] = useState<AutomatedAction | null>(null)

  const { data: actionsData, isLoading: actionsLoading } = useAutomatedActions({ pageSize: 20 })
  const { data: rules, isLoading: rulesLoading } = useAutomatedRules('active')
  const { data: stats, isLoading: statsLoading } = useAutomatedResponseStats()
  const { data: metrics, isLoading: metricsLoading } = useAutomatedResponseMetrics(24)

  const actions = actionsData?.items || []

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statsLoading ? (
          <>
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-28 w-full" />
            ))}
          </>
        ) : (
          <>
            <StatsCard
              title="Actions Today"
              value={stats?.actionsToday?.toString() || '0'}
              change="+0"
              trend="up"
              icon={Zap}
              color="blue"
            />
            <StatsCard
              title="Success Rate"
              value={`${stats?.successRate?.toFixed(1) || 0}%`}
              change="+0%"
              trend="up"
              icon={Check}
              color="green"
            />
            <StatsCard
              title="Avg Response Time"
              value={`${stats?.averageResponseTime?.toFixed(1) || 0}s`}
              change="-0s"
              trend="down"
              icon={Clock}
              color="yellow"
            />
            <StatsCard
              title="Active Rules"
              value={stats?.activeRules?.toString() || '0'}
              change="+0"
              trend="up"
              icon={Settings}
              color="blue"
            />
          </>
        )}
      </div>

      {/* Main Content */}
      <Tabs defaultValue="actions" className="space-y-4">
        <TabsList>
          <TabsTrigger value="actions">
            <Activity className="w-4 h-4 mr-2" />
            Recent Actions
          </TabsTrigger>
          <TabsTrigger value="metrics">
            <RefreshCw className="w-4 h-4 mr-2" />
            Performance Metrics
          </TabsTrigger>
          <TabsTrigger value="rules">
            <Shield className="w-4 h-4 mr-2" />
            Active Rules
          </TabsTrigger>
        </TabsList>

        {/* Recent Actions Tab */}
        <TabsContent value="actions">
          <DashboardCard title="Recent Automated Actions" icon={Zap}>
            <div className="space-y-4">
              {actionsLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))}
                </div>
              ) : actions.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  No automated actions recorded yet
                </div>
              ) : (
                actions.map((action) => (
                <div
                  key={action.id}
                  className={`p-4 rounded-lg border cursor-pointer ${
                    selectedAction?.id === action.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                  onClick={() => setSelectedAction(action)}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          action.status === 'success'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200'
                            : action.status === 'failed'
                            ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                            : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                        }`}>
                          {action.status}
                        </span>
                        <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
                          {new Date(action.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <h3 className="font-medium text-gray-900 dark:text-white mt-2">
                        {action.trigger}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        {action.details}
                      </p>
                    </div>
                  </div>

                  {selectedAction?.id === action.id && (
                    <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <span className="text-sm text-gray-500 dark:text-gray-400">Target</span>
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            {action.target}
                          </p>
                        </div>
                        <div>
                          <span className="text-sm text-gray-500 dark:text-gray-400">Result</span>
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            {action.result || 'Pending'}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )))}
            </div>
          </DashboardCard>
        </TabsContent>

        {/* Performance Metrics Tab */}
        <TabsContent value="metrics">
          <DashboardCard title="Response Performance" icon={Activity}>
            <div className="h-[300px]">
              {metricsLoading ? (
                <Skeleton className="h-full w-full" />
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metrics || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="actions"
                      stroke="#3b82f6"
                      fill="#3b82f6"
                      fillOpacity={0.3}
                      name="Actions"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="responseTime"
                      stroke="#10b981"
                      name="Response Time (s)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </DashboardCard>
        </TabsContent>

        {/* Active Rules Tab */}
        <TabsContent value="rules">
          <DashboardCard title="Automated Response Rules" icon={Shield}>
            <div className="space-y-4">
              {rulesLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-28 w-full" />
                  ))}
                </div>
              ) : !rules || rules.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  No active rules configured
                </div>
              ) : (
                rules.map((rule) => (
                <div
                  key={rule.id}
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">
                        {rule.name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        {rule.description}
                      </p>
                    </div>
                    <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200">
                      {rule.status}
                    </span>
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        Triggers Today
                      </span>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {rule.triggers}
                      </p>
                    </div>
                    <div>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        Last Triggered
                      </span>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {rule.lastTriggered 
                          ? new Date(rule.lastTriggered).toLocaleString()
                          : 'Never'}
                      </p>
                    </div>
                  </div>
                </div>
              )))}
            </div>
          </DashboardCard>
        </TabsContent>
      </Tabs>
    </div>
  )
}
