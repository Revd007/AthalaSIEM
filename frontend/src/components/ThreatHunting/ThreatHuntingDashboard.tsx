'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { Target, Activity, Search, AlertTriangle, Clock } from 'lucide-react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const mockHuntMetrics = Array.from({ length: 7 }, (_, i) => ({
  date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toLocaleDateString(),
  threats: Math.floor(Math.random() * 50),
  hunts: Math.floor(Math.random() * 20),
  findings: Math.floor(Math.random() * 30)
})).reverse()

const activeHunts = [
  {
    id: 1,
    name: 'Lateral Movement Detection',
    analyst: 'John Doe',
    status: 'in-progress',
    progress: 65,
    findings: 3
  },
  {
    id: 2,
    name: 'Data Exfiltration Hunt',
    analyst: 'Jane Smith',
    status: 'in-progress',
    progress: 42,
    findings: 1
  }
]

interface ThreatData {
  id: string
  timestamp: string
  type: string
  severity: string
  source: string
  message: string
  details: Record<string, unknown>
}

interface ThreatHuntingDashboardProps {
  data: ThreatData[]
  onThreatClick: (threat: ThreatData) => void
}

export function ThreatHuntingDashboard({ data, onThreatClick }: ThreatHuntingDashboardProps) {
  return (
    <div className="space-y-6">
      {/* Hunt Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Active Hunts"
          value="8"
          change="+2"
          trend="up"
          icon={Search}
        />
        <MetricCard
          title="Total Findings"
          value="47"
          change="+12"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <MetricCard
          title="Avg. Hunt Duration"
          value="4.2h"
          change="-0.5h"
          trend="down"
          icon={Clock}
          color="green"
        />
        <MetricCard
          title="Success Rate"
          value="76%"
          change="+5%"
          trend="up"
          icon={Target}
          color="blue"
        />
      </div>

      {/* Hunt Activity Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DashboardCard title="Hunt Activity" icon={Activity}>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockHuntMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="hunts" stroke="#3b82f6" name="Hunts" />
                <Line type="monotone" dataKey="findings" stroke="#ef4444" name="Findings" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </DashboardCard>

        {/* Active Hunts */}
        <DashboardCard title="Active Hunts" icon={Search}>
          <div className="space-y-4">
            {activeHunts.map(hunt => (
              <div key={hunt.id} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-medium text-gray-900 dark:text-white">
                      {hunt.name}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Analyst: {hunt.analyst}
                    </p>
                  </div>
                  <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200">
                    {hunt.findings} findings
                  </span>
                </div>
                <div className="mt-4">
                  <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mb-1">
                    <span>Progress</span>
                    <span>{hunt.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${hunt.progress}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </DashboardCard>
      </div>

      {/* Recent Findings */}
      <DashboardCard title="Recent Findings" icon={AlertTriangle}>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Finding
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Hunt
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Severity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {/* Add mock findings data here */}
            </tbody>
          </table>
        </div>
      </DashboardCard>
    </div>
  )
}

interface MetricCardProps {
  title: string
  value: string
  change: string
  trend: 'up' | 'down'
  icon: any
  color?: 'blue' | 'red' | 'green'
}

function MetricCard({ title, value, change, trend, icon: Icon, color = 'blue' }: MetricCardProps) {
  const colors = {
    blue: 'bg-blue-50 dark:bg-blue-900/20',
    red: 'bg-red-50 dark:bg-red-900/20',
    green: 'bg-green-50 dark:bg-green-900/20',
  }

  return (
    <div className={`${colors[color]} rounded-lg p-6`}>
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</span>
        <Icon className="h-5 w-5 text-gray-400" />
      </div>
      <div className="mt-4">
        <span className="text-2xl font-bold text-gray-900 dark:text-white">{value}</span>
        <span className={`
          ml-2 text-sm font-medium
          ${trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
        `}>
          {change}
        </span>
      </div>
    </div>
  )
}