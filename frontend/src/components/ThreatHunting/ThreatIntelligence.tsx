'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Globe, Shield, AlertTriangle, RefreshCw, Search, Filter, Download, ExternalLink } from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'
import { StatsCard } from '../SecurityEvents/StatsCard'

interface ThreatFeed {
  id: string
  name: string
  provider: string
  type: 'ip' | 'domain' | 'hash' | 'url'
  lastUpdate: string
  status: 'active' | 'disabled'
  indicators: number
  matches: number
}

const mockFeeds: ThreatFeed[] = [
  {
    id: '1',
    name: 'Emerging Threats',
    provider: 'ProofPoint',
    type: 'ip',
    lastUpdate: new Date().toISOString(),
    status: 'active',
    indicators: 25000,
    matches: 142
  },
  {
    id: '2',
    name: 'Abuse.ch Blocklist',
    provider: 'Abuse.ch',
    type: 'domain',
    lastUpdate: new Date().toISOString(),
    status: 'active',
    indicators: 15000,
    matches: 89
  }
]

const mockThreatTypes = [
  { name: 'Malware', value: 45 },
  { name: 'C2', value: 25 },
  { name: 'Phishing', value: 20 },
  { name: 'Ransomware', value: 10 }
]

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981']

export function ThreatIntelligence() {
  const [selectedFeed, setSelectedFeed] = useState<ThreatFeed | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = async () => {
    setIsRefreshing(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsRefreshing(false)
  }

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Active Feeds"
          value="12"
          change="+2"
          trend="up"
          icon={Globe}
        />
        <StatsCard
          title="Total Indicators"
          value="156,893"
          change="+12.5k"
          trend="up"
          icon={Shield}
          color="blue"
        />
        <StatsCard
          title="Matches Today"
          value="247"
          change="+18"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Enrichment Rate"
          value="98.2%"
          change="+0.5%"
          trend="up"
          icon={RefreshCw}
          color="green"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Threat Feeds */}
        <div className="lg:col-span-2">
          <DashboardCard title="Threat Feeds" icon={Globe}>
            <div className="space-y-4">
              {/* Actions */}
              <div className="flex justify-between">
                <div className="flex space-x-2">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search feeds..."
                      className="pl-10 pr-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
                    />
                    <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                  </div>
                  <button className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                    <Filter className="h-5 w-5 text-gray-500" />
                  </button>
                </div>
                <button 
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center"
                  onClick={handleRefresh}
                >
                  {isRefreshing ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4 mr-2" />
                  )}
                  Refresh All
                </button>
              </div>

              {/* Feeds Table */}
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead>
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Feed
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Provider
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Indicators
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Matches
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Last Update
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {mockFeeds.map(feed => (
                      <tr key={feed.id}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="font-medium text-gray-900 dark:text-white">
                            {feed.name}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {feed.provider}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200">
                            {feed.type}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {feed.indicators.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {feed.matches.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {new Date(feed.lastUpdate).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <button className="text-blue-500 hover:text-blue-600">
                            <ExternalLink className="h-4 w-4" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </DashboardCard>
        </div>

        {/* Threat Distribution */}
        <div className="lg:col-span-1">
          <DashboardCard title="Threat Distribution" icon={AlertTriangle}>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={mockThreatTypes}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {mockThreatTypes.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4">
              <div className="grid grid-cols-2 gap-4">
                {mockThreatTypes.map((type, index) => (
                  <div key={type.name} className="flex items-center">
                    <div 
                      className="w-3 h-3 rounded-full mr-2"
                      style={{ backgroundColor: COLORS[index] }}
                    />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {type.name} ({type.value}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </DashboardCard>
        </div>
      </div>
    </div>
  )
} 