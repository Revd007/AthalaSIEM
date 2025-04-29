'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Activity, AlertTriangle, Brain, LineChart, Settings, RefreshCw } from 'lucide-react'
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { StatsCard } from '@/components/ui/StatsCard'
import { toast } from 'react-hot-toast'

interface AnomalyScore {
  timestamp: string
  score: number
  threshold: number
  category: string
}

interface AnomalyDetectionError {
  message: string
  code: string
  details?: Record<string, unknown>
}

const mockTimeSeriesData = Array.from({ length: 24 }, (_, i) => ({
  timestamp: new Date(Date.now() - i * 3600000).toISOString(),
  score: Math.random() * 100,
  threshold: 75,
  category: Math.random() > 0.8 ? 'anomaly' : 'normal'
})).reverse()

const mockAnomalies = [
  {
    id: '1',
    type: 'Network',
    description: 'Unusual outbound data transfer pattern detected',
    severity: 'high',
    timestamp: new Date().toISOString(),
    score: 89,
    details: {
      source_ip: '192.168.1.100',
      destination: 'unknown-domain.com',
      data_volume: '1.2GB'
    }
  },
  {
    id: '2',
    type: 'Authentication',
    description: 'Multiple failed login attempts from new location',
    severity: 'medium',
    timestamp: new Date(Date.now() - 1000000).toISOString(),
    score: 76,
    details: {
      username: 'admin',
      location: 'Unknown',
      attempts: 5
    }
  }
]

export function AnomalyDetection() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [selectedAnomaly, setSelectedAnomaly] = useState<any>(null)

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsRefreshing(false)
  }

  const handleError = (error: AnomalyDetectionError) => {
    console.error('Anomaly detection error:', error)
    toast.error(error.message)
  }

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Anomaly Score"
          value="78.5"
          change="+12.3"
          trend="up"
          icon={Activity}
          color="blue"
        />
        <StatsCard
          title="Detected Today"
          value="24"
          change="+5"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="False Positives"
          value="3.2%"
          change="-0.5%"
          trend="down"
          icon={Brain}
          color="green"
        />
        <StatsCard
          title="Model Accuracy"
          value="96.8%"
          change="+0.3%"
          trend="up"
          icon={LineChart}
          color="blue"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Anomaly Timeline */}
        <div className="lg:col-span-2">
          <DashboardCard title="Anomaly Timeline" icon={Activity}>
            <div className="space-y-4">
              {/* Controls */}
              <div className="flex justify-between">
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

              {/* Chart */}
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mockTimeSeriesData}>
                    <defs>
                      <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                              <p className="text-sm text-gray-500 dark:text-gray-400">
                                {new Date(payload[0].payload.timestamp).toLocaleString()}
                              </p>
                              <p className="text-sm font-medium text-gray-900 dark:text-white">
                                Score: {typeof payload[0]?.value === 'number' ? payload[0]?.value.toFixed(2) : 'N/A'}
                              </p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="score"
                      stroke="#3b82f6"
                      fillOpacity={1}
                      fill="url(#scoreGradient)"
                    />
                    <Line
                      type="monotone"
                      dataKey="threshold"
                      stroke="#ef4444"
                      strokeDasharray="5 5"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </DashboardCard>
        </div>

        {/* Detected Anomalies */}
        <div className="lg:col-span-1">
          <DashboardCard title="Detected Anomalies" icon={AlertTriangle}>
            <div className="space-y-4">
              {mockAnomalies.map((anomaly) => (
                <div
                  key={anomaly.id}
                  onClick={() => setSelectedAnomaly(anomaly)}
                  className={`p-4 rounded-lg cursor-pointer border ${
                    selectedAnomaly?.id === anomaly.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          anomaly.severity === 'high'
                            ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                            : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                        }`}>
                          {anomaly.severity}
                        </span>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          {anomaly.type}
                        </span>
                      </div>
                      <p className="text-sm text-gray-900 dark:text-white mt-1">
                        {anomaly.description}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-semibold text-gray-900 dark:text-white">
                        {anomaly.score}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        score
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    {new Date(anomaly.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </DashboardCard>
        </div>
      </div>

      {/* Model Configuration */}
      <DashboardCard title="Model Configuration" icon={Settings}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white">
              Detection Parameters
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Sensitivity Threshold
                </span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  defaultValue="75"
                  className="w-32"
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Learning Rate
                </span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  defaultValue="85"
                  className="w-32"
                />
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white">
              Model Performance
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  True Positive Rate
                </span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  96.8%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  False Positive Rate
                </span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  3.2%
                </span>
              </div>
            </div>
          </div>
        </div>
      </DashboardCard>
    </div>
  )
} 