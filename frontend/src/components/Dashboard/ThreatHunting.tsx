'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { Shield, AlertTriangle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const mockData = [
  { name: 'Malware', value: 45 },
  { name: 'Phishing', value: 32 },
  { name: 'Data Exfil', value: 28 },
  { name: 'Lateral Movement', value: 15 },
  { name: 'Privilege Esc', value: 12 }
]

export function ThreatHunting() {
  return (
    <DashboardCard title="Threat Hunting Overview" icon={Shield}>
      <div className="space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Active Hunts</div>
            <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">8</div>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Findings Today</div>
            <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">12</div>
          </div>
        </div>

        {/* Chart */}
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={mockData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Findings */}
        <div>
          <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Recent Findings
          </h3>
          <div className="space-y-2">
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <div className="flex items-center">
                <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2" />
                <span className="text-sm text-yellow-800 dark:text-yellow-200">
                  Suspicious PowerShell Activity
                </span>
              </div>
              <span className="text-xs text-yellow-600 dark:text-yellow-300 mt-1 block">
                2 minutes ago
              </span>
            </div>
          </div>
        </div>
      </div>
    </DashboardCard>
  )
} 