'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Zap, Check, AlertTriangle, Clock, Settings, Shield, Activity, RefreshCw } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface AutomatedAction {
  id: string
  type: 'block' | 'isolate' | 'scan' | 'alert'
  trigger: string
  status: 'success' | 'failed' | 'in-progress'
  timestamp: string
  target: string
  details: string
  result?: string
}

const mockActions: AutomatedAction[] = [
  {
    id: '1',
    type: 'block',
    trigger: 'Malicious IP Detection',
    status: 'success',
    timestamp: new Date().toISOString(),
    target: '192.168.1.100',
    details: 'Blocked malicious IP after multiple failed login attempts',
    result: 'IP blocked for 24 hours'
  },
  {
    id: '2',
    type: 'isolate',
    trigger: 'Ransomware Behavior',
    status: 'success',
    timestamp: new Date(Date.now() - 1800000).toISOString(),
    target: 'WORKSTATION-01',
    details: 'Isolated endpoint showing ransomware indicators',
    result: 'Endpoint isolated from network'
  }
]

const mockMetrics = Array.from({ length: 24 }, (_, i) => ({
  time: `${i}:00`,
  actions: Math.floor(Math.random() * 30),
  responseTime: Math.random() * 2
}))

const mockRules = [
  {
    id: '1',
    name: 'Suspicious Login Block',
    description: 'Block IPs after multiple failed logins',
    status: 'active',
    triggers: 5,
    lastTriggered: new Date().toISOString()
  },
  {
    id: '2',
    name: 'Endpoint Isolation',
    description: 'Isolate endpoints showing malware indicators',
    status: 'active',
    triggers: 2,
    lastTriggered: new Date(Date.now() - 3600000).toISOString()
  }
]

export function AutomatedResponse() {
  const [selectedAction, setSelectedAction] = useState<AutomatedAction | null>(null)

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Actions Today"
          value="24"
          change="+5"
          trend="up"
          icon={Zap}
          color="blue"
        />
        <StatsCard
          title="Success Rate"
          value="94.2%"
          change="+2.1%"
          trend="up"
          icon={Check}
          color="green"
        />
        <StatsCard
          title="Avg Response Time"
          value="1.2s"
          change="-0.3s"
          trend="down"
          icon={Clock}
          color="yellow"
        />
        <StatsCard
          title="Active Rules"
          value="15"
          change="+2"
          trend="up"
          icon={Settings}
          color="blue"
        />
      </div>

      {/* Main Content */}
      <Tabs defaultValue="actions" className="space-y-4">
        <TabsList>
          <TabsTrigger value="actions">
            <Activity className="w-4 h-4 mr-2" />
            Recent Actions
          </TabsTrigger>
          <TabsTrigger value="metrics">
            <LineChart className="w-4 h-4 mr-2" />
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
              {mockActions.map((action) => (
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
                            {action.result}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </DashboardCard>
        </TabsContent>

        {/* Performance Metrics Tab */}
        <TabsContent value="metrics">
          <DashboardCard title="Response Performance" icon={Activity}>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="actions"
                    stroke="#3b82f6"
                    name="Actions"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="responseTime"
                    stroke="#10b981"
                    name="Response Time (s)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </DashboardCard>
        </TabsContent>

        {/* Active Rules Tab */}
        <TabsContent value="rules">
          <DashboardCard title="Automated Response Rules" icon={Shield}>
            <div className="space-y-4">
              {mockRules.map((rule) => (
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
                        {new Date(rule.lastTriggered).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </DashboardCard>
        </TabsContent>
      </Tabs>
    </div>
  )
} 