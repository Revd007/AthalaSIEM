'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Activity, Users, Network, Brain, Search } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const mockTactics = [
  { name: 'Initial Access', value: 12 },
  { name: 'Execution', value: 18 },
  { name: 'Persistence', value: 8 },
  { name: 'Privilege Escalation', value: 15 },
  { name: 'Defense Evasion', value: 22 },
  { name: 'Credential Access', value: 14 },
  { name: 'Discovery', value: 25 },
  { name: 'Lateral Movement', value: 16 },
  { name: 'Collection', value: 11 },
  { name: 'Exfiltration', value: 7 },
]

export function BehaviorAnalysis() {
  const [selectedTactic, setSelectedTactic] = useState<string | null>(null)

  return (
    <div className="space-y-6">
      {/* MITRE ATT&CK Overview */}
      <DashboardCard title="MITRE ATT&CK Coverage" icon={Activity}>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={mockTactics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis />
              <Tooltip />
              <Bar 
                dataKey="value" 
                fill="#3b82f6"
                onClick={(data) => setSelectedTactic(data.name)}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </DashboardCard>

      {/* Analysis Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Process Behavior */}
        <DashboardCard title="Process Behavior" icon={Activity}>
          <div className="space-y-4">
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <h3 className="font-medium text-yellow-800 dark:text-yellow-200">Suspicious Process</h3>
              <p className="text-sm text-yellow-600 dark:text-yellow-300 mt-1">
                PowerShell.exe executing encoded commands
              </p>
              <div className="mt-3 flex justify-between text-sm">
                <span className="text-yellow-600 dark:text-yellow-300">PID: 4528</span>
                <span className="text-yellow-600 dark:text-yellow-300">5 min ago</span>
              </div>
            </div>
            {/* Add more process alerts */}
          </div>
        </DashboardCard>

        {/* Network Behavior */}
        <DashboardCard title="Network Behavior" icon={Network}>
          <div className="space-y-4">
            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <h3 className="font-medium text-red-800 dark:text-red-200">Anomalous Traffic</h3>
              <p className="text-sm text-red-600 dark:text-red-300 mt-1">
                Large data transfer to unknown external IP
              </p>
              <div className="mt-3 flex justify-between text-sm">
                <span className="text-red-600 dark:text-red-300">192.168.1.100</span>
                <span className="text-red-600 dark:text-red-300">2 min ago</span>
              </div>
            </div>
            {/* Add more network alerts */}
          </div>
        </DashboardCard>

        {/* User Behavior */}
        <DashboardCard title="User Behavior" icon={Users}>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h3 className="font-medium text-blue-800 dark:text-blue-200">Unusual Activity</h3>
              <p className="text-sm text-blue-600 dark:text-blue-300 mt-1">
                Multiple failed login attempts from new location
              </p>
              <div className="mt-3 flex justify-between text-sm">
                <span className="text-blue-600 dark:text-blue-300">user.admin</span>
                <span className="text-blue-600 dark:text-blue-300">1 min ago</span>
              </div>
            </div>
            {/* Add more user behavior alerts */}
          </div>
        </DashboardCard>
      </div>

      {/* ML-Based Analysis */}
      <DashboardCard title="Machine Learning Analysis" icon={Brain}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900 dark:text-white">Anomaly Detection</h3>
            {/* Add anomaly scores and charts */}
          </div>
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900 dark:text-white">Behavior Clustering</h3>
            {/* Add clustering visualization */}
          </div>
        </div>
      </DashboardCard>
    </div>
  )
} 