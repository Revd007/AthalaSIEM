'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { 
  Activity, 
  AlertTriangle, 
  Check, 
  Clock, 
  Play,
  Shield 
} from 'lucide-react'

export function PlaybooksOverviewTab() {
  const stats = {
    activePlaybooks: 12,
    totalExecutions: 1458,
    successRate: 94.5,
    averageRuntime: '2m 15s',
    incidentsResolved: 286,
    automationRate: 78.3
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <DashboardCard>
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-blue-100 rounded-lg">
              <Play className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">Active Playbooks</p>
              <h3 className="text-2xl font-bold">{stats.activePlaybooks}</h3>
            </div>
          </div>
        </DashboardCard>

        <DashboardCard>
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-green-100 rounded-lg">
              <Check className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">Success Rate</p>
              <h3 className="text-2xl font-bold">{stats.successRate}%</h3>
            </div>
          </div>
        </DashboardCard>

        <DashboardCard>
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-purple-100 rounded-lg">
              <Shield className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">Incidents Resolved</p>
              <h3 className="text-2xl font-bold">{stats.incidentsResolved}</h3>
            </div>
          </div>
        </DashboardCard>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DashboardCard title="Recent Activity" icon={Activity}>
          <div className="space-y-4">
            {/* Recent activity items */}
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <AlertTriangle className="w-5 h-5 text-orange-500" />
                  <div>
                    <p className="font-medium">Malware Detection Response</p>
                    <p className="text-sm text-gray-500">Triggered by alert #1234</p>
                  </div>
                </div>
                <span className="text-sm text-gray-500">2 min ago</span>
              </div>
            </div>
          </div>
        </DashboardCard>

        <DashboardCard title="Performance Metrics" icon={Clock}>
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Average Runtime</span>
                <span className="text-sm text-gray-500">{stats.averageRuntime}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Automation Rate</span>
                <span className="text-sm text-gray-500">{stats.automationRate}%</span>
              </div>
            </div>
          </div>
        </DashboardCard>
      </div>
    </div>
  )
} 