'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { StatsCard } from '@/components/ui/StatsCard'
import { Shield, AlertTriangle, Target, Zap, RefreshCw, Search, Filter } from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'
import { toast } from 'react-hot-toast'

interface ThreatEvent {
  id: string
  type: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  source: string
  target: string
  description: string
  timestamp: string
  confidence: number
  mitreTactic: string
  mitreId: string
  details: Record<string, any>
}

interface ThreatAnalysisError {
  message: string
  code: string
  details?: Record<string, unknown>
}

const mockThreatData = {
  byType: [
    { name: 'Malware', value: 35 },
    { name: 'APT', value: 25 },
    { name: 'Zero Day', value: 15 },
    { name: 'Insider', value: 25 }
  ],
  byTactic: [
    { name: 'Initial Access', value: 28 },
    { name: 'Execution', value: 22 },
    { name: 'Persistence', value: 18 },
    { name: 'Privilege Escalation', value: 15 },
    { name: 'Defense Evasion', value: 12 }
  ]
}

const mockThreats: ThreatEvent[] = [
  {
    id: '1',
    type: 'APT',
    severity: 'critical',
    source: '192.168.1.100',
    target: 'internal-server',
    description: 'Advanced persistent threat activity detected',
    timestamp: new Date().toISOString(),
    confidence: 92,
    mitreTactic: 'Initial Access',
    mitreId: 'T1190',
    details: {
      technique: 'Exploit Public-Facing Application',
      indicators: ['Unusual Process Creation', 'Network Scanning']
    }
  },
  {
    id: '2',
    type: 'Malware',
    severity: 'high',
    source: 'external-endpoint',
    target: '192.168.1.50',
    description: 'Ransomware behavior pattern identified',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    confidence: 88,
    mitreTactic: 'Execution',
    mitreId: 'T1204',
    details: {
      technique: 'User Execution',
      indicators: ['File Encryption', 'Registry Modification']
    }
  }
]

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981']

export function AIThreatAnalyzer() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [selectedThreat, setSelectedThreat] = useState<ThreatEvent | null>(null)

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsRefreshing(false)
  }

  const handleError = (error: ThreatAnalysisError) => {
    console.error('Threat analysis error:', error)
    toast.error(error.message)
  }

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Active Threats"
          value="12"
          change="+3"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Threat Score"
          value="85.2"
          change="+5.4"
          trend="up"
          icon={Target}
          color="blue"
        />
        <StatsCard
          title="Detection Rate"
          value="94.8%"
          change="+2.1%"
          trend="up"
          icon={Shield}
          color="green"
        />
        <StatsCard
          title="Response Time"
          value="1.2m"
          change="-0.3m"
          trend="down"
          icon={Zap}
          color="yellow"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Threat Distribution */}
        <div className="lg:col-span-1">
          <DashboardCard title="Threat Distribution" icon={AlertTriangle}>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={mockThreatData.byType}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {mockThreatData.byType.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4">
              <div className="grid grid-cols-2 gap-4">
                {mockThreatData.byType.map((type, index) => (
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

        {/* MITRE ATT&CK Distribution */}
        <div className="lg:col-span-2">
          <DashboardCard title="MITRE ATT&CK Coverage" icon={Target}>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mockThreatData.byTactic}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </DashboardCard>
        </div>
      </div>

      {/* Detected Threats */}
      <DashboardCard title="Detected Threats" icon={Shield}>
        <div className="space-y-4">
          {/* Controls */}
          <div className="flex justify-between">
            <div className="relative flex-1 max-w-sm">
              <input
                type="text"
                placeholder="Search threats..."
                className="w-full pl-10 pr-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
              />
              <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
            </div>
            <div className="flex space-x-2">
              <select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <button
                onClick={handleRefresh}
                className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
              >
                <RefreshCw className={`h-5 w-5 ${isRefreshing ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>

          {/* Threats List */}
          <div className="space-y-4">
            {mockThreats.map((threat) => (
              <div
                key={threat.id}
                onClick={() => setSelectedThreat(threat)}
                className={`p-4 rounded-lg cursor-pointer border ${
                  selectedThreat?.id === threat.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        threat.severity === 'critical'
                          ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                          : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                      }`}>
                        {threat.severity}
                      </span>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {threat.type}
                      </span>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        [{threat.mitreId}]
                      </span>
                    </div>
                    <p className="text-sm text-gray-900 dark:text-white mt-1">
                      {threat.description}
                    </p>
                    <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                      <span>Source: {threat.source}</span>
                      <span>Target: {threat.target}</span>
                      <span>Confidence: {threat.confidence}%</span>
                    </div>
                  </div>
                  <div className="ml-4 text-right">
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {new Date(threat.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>

                {selectedThreat?.id === threat.id && (
                  <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Threat Details
                    </h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          MITRE Tactic
                        </span>
                        <span className="text-sm text-gray-900 dark:text-white">
                          {threat.mitreTactic}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          Technique
                        </span>
                        <span className="text-sm text-gray-900 dark:text-white">
                          {threat.details.technique}
                        </span>
                      </div>
                      <div>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          Indicators:
                        </span>
                        <ul className="mt-1 list-disc list-inside">
                          {threat.details.indicators.map((indicator: string, index: number) => (
                            <li key={index} className="text-sm text-gray-900 dark:text-white">
                              {indicator}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </DashboardCard>
    </div>
  )
} 