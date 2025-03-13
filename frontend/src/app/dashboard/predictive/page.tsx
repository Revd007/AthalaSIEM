'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { 
  LineChart as LineChartIcon, Brain, AlertTriangle, Target, 
  Clock, Shield, Activity, TrendingUp, Search, Filter, 
  RefreshCw, ChevronDown, ArrowUpRight, Check 
} from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell 
} from 'recharts'

interface DetailedPrediction {
  id: string
  type: string
  probability: number
  impact: 'critical' | 'high' | 'medium' | 'low'
  timeframe: string
  details: string
  indicators: string[]
  mitigation: string[]
  affectedAssets: string[]
  confidence: number
  mlModel: string
  dataPoints: number
  lastUpdated: string
  status: 'active' | 'monitoring' | 'resolved'
  category: 'attack' | 'anomaly' | 'vulnerability' | 'threat'
  tags: string[]
  relatedIncidents?: string[]
}

const mockPredictions: DetailedPrediction[] = [
  {
    id: '1',
    type: 'Advanced Persistent Threat',
    probability: 89.5,
    impact: 'critical',
    timeframe: '12-24 hours',
    details: 'Sophisticated attack pattern detected targeting critical infrastructure',
    indicators: [
      'Command & Control communication patterns',
      'Lateral movement attempts',
      'Data staging activities'
    ],
    mitigation: [
      'Isolate affected systems',
      'Block suspicious IPs',
      'Enable enhanced monitoring'
    ],
    affectedAssets: ['Database Servers', 'Domain Controllers'],
    confidence: 92,
    mlModel: 'APT Detection v2.1',
    dataPoints: 15420,
    lastUpdated: new Date().toISOString(),
    status: 'active',
    category: 'attack',
    tags: ['apt', 'lateral-movement', 'data-theft']
  },
  // Add more mock predictions...
]

const mockTrendData = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toLocaleDateString(),
  predictions: Math.floor(Math.random() * 100),
  accuracy: 75 + Math.random() * 20,
  threats: Math.floor(Math.random() * 50)
}))

const mockModels = [
  {
    id: '1',
    name: 'APT Detection',
    version: '2.1',
    type: 'Anomaly Detection',
    accuracy: 94.2,
    lastTrained: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    status: 'active',
    predictions: 1240,
    successRate: 92.5
  },
  {
    id: '2',
    name: 'Ransomware Predictor',
    version: '1.8',
    type: 'Behavior Analysis',
    accuracy: 91.8,
    lastTrained: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    status: 'active',
    predictions: 856,
    successRate: 89.3
  }
]

const mockCategories = [
  { name: 'APT', value: 35 },
  { name: 'Ransomware', value: 25 },
  { name: 'Data Breach', value: 20 },
  { name: 'DDoS', value: 20 }
]

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981']

export default function PredictivePage() {
  const [selectedPrediction, setSelectedPrediction] = useState<DetailedPrediction | null>(null)
  const [timeRange, setTimeRange] = useState('7d')

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold tracking-tight">Predictive Analysis</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Time Range:</span>
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="text-sm border rounded-md px-2 py-1"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
          <button
            onClick={() => {/* Implement refresh */}}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full"
          >
            <RefreshCw className="h-5 w-5" />
          </button>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">
            <Brain className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="predictions">
            <Target className="w-4 h-4 mr-2" />
            Active Predictions
          </TabsTrigger>
          <TabsTrigger value="trends">
            <TrendingUp className="w-4 h-4 mr-2" />
            Trends Analysis
          </TabsTrigger>
          <TabsTrigger value="models">
            <Activity className="w-4 h-4 mr-2" />
            ML Models
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview">
          <div className="space-y-6">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatsCard
                title="Active Predictions"
                value="24"
                change="+5"
                trend="up"
                icon={Brain}
                color="blue"
              />
              <StatsCard
                title="Prediction Accuracy"
                value="94.2%"
                change="+2.1%"
                trend="up"
                icon={Target}
                color="green"
              />
              <StatsCard
                title="Critical Threats"
                value="3"
                change="+1"
                trend="up"
                icon={AlertTriangle}
                color="red"
              />
              <StatsCard
                title="ML Models Active"
                value="8"
                change="0"
                trend="stable"
                icon={Activity}
                color="yellow"
              />
            </div>

            {/* Recent Predictions */}
            <DashboardCard title="Recent Predictions" icon={Target}>
              <div className="space-y-4">
                {mockPredictions.map((prediction) => (
                  <div
                    key={prediction.id}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">
                          {prediction.type}
                        </h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                          {prediction.details}
                        </p>
                      </div>
                      <span
                        className={`px-2 py-1 text-xs rounded-full ${
                          prediction.impact === 'critical'
                            ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                            : 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200'
                        }`}
                      >
                        {prediction.probability}% probability
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </DashboardCard>
          </div>
        </TabsContent>

        {/* Replace the predictions tab content */}
        <TabsContent value="predictions">
          <div className="space-y-6">
            {/* Search and Filter */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search predictions..."
                    className="pl-10 pr-4 py-2 border rounded-md w-64"
                  />
                </div>
                <select className="border rounded-md px-3 py-2">
                  <option value="all">All Categories</option>
                  <option value="attack">Attacks</option>
                  <option value="anomaly">Anomalies</option>
                  <option value="vulnerability">Vulnerabilities</option>
                </select>
                <select className="border rounded-md px-3 py-2">
                  <option value="all">All Statuses</option>
                  <option value="active">Active</option>
                  <option value="monitoring">Monitoring</option>
                  <option value="resolved">Resolved</option>
                </select>
              </div>
              <button className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600">
                <Filter className="h-4 w-4" />
                Advanced Filters
              </button>
            </div>

            {/* Detailed Predictions List */}
            <div className="grid grid-cols-1 gap-6">
              {mockPredictions.map((prediction) => (
                <DashboardCard key={prediction.id} icon={AlertTriangle}>
                  <div className="space-y-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                            {prediction.type}
                          </h3>
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            prediction.impact === 'critical'
                              ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                              : 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200'
                          }`}>
                            {prediction.impact.toUpperCase()}
                          </span>
                        </div>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                          {prediction.details}
                        </p>
                      </div>
                      <div className="flex flex-col items-end">
                        <span className="text-2xl font-bold text-blue-500">
                          {prediction.probability}%
                        </span>
                        <span className="text-sm text-gray-500">probability</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                          Key Indicators
                        </h4>
                        <ul className="space-y-1">
                          {prediction.indicators.map((indicator, index) => (
                            <li key={index} className="text-sm text-gray-600 dark:text-gray-300 flex items-center gap-2">
                              <AlertTriangle className="h-4 w-4 text-yellow-500" />
                              {indicator}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                          Affected Assets
                        </h4>
                        <ul className="space-y-1">
                          {prediction.affectedAssets.map((asset, index) => (
                            <li key={index} className="text-sm text-gray-600 dark:text-gray-300 flex items-center gap-2">
                              <Shield className="h-4 w-4 text-blue-500" />
                              {asset}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                          Mitigation Steps
                        </h4>
                        <ul className="space-y-1">
                          {prediction.mitigation.map((step, index) => (
                            <li key={index} className="text-sm text-gray-600 dark:text-gray-300 flex items-center gap-2">
                              <Check className="h-4 w-4 text-green-500" />
                              {step}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-4 border-t">
                      <div className="flex items-center gap-4">
                        <span className="text-sm text-gray-500">
                          ML Model: {prediction.mlModel}
                        </span>
                        <span className="text-sm text-gray-500">
                          Confidence: {prediction.confidence}%
                        </span>
                        <span className="text-sm text-gray-500">
                          Data Points: {prediction.dataPoints.toLocaleString()}
                        </span>
                      </div>
                      <button className="flex items-center gap-2 text-blue-500 hover:text-blue-600">
                        <ArrowUpRight className="h-4 w-4" />
                        View Details
                      </button>
                    </div>
                  </div>
                </DashboardCard>
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="trends">
          <div className="space-y-6">
            {/* Trend Analysis Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <DashboardCard title="Prediction Trends" icon={TrendingUp}>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={mockTrendData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="date" 
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis 
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip
                        content={({ active, payload, label }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                                <p className="text-sm font-medium text-gray-900 dark:text-white">
                                  Date: {label}
                                </p>
                                <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                                  Predictions: {payload[0].value}
                                </p>
                                <p className="text-sm font-medium text-green-600 dark:text-green-400">
                                  Accuracy: {payload[1].value.toFixed(1)}%
                                </p>
                              </div>
                            )
                          }
                          return null
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="predictions"
                        stroke="#3b82f6"
                        fill="#3b82f6"
                        fillOpacity={0.1}
                        name="Predictions"
                      />
                      <Area
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#10b981"
                        fill="#10b981"
                        fillOpacity={0.1}
                        name="Accuracy"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </DashboardCard>

              <DashboardCard title="Threat Distribution" icon={PieChart}>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={mockCategories}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        fill="#8884d8"
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {mockCategories.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                                <p className="text-sm font-medium text-gray-900 dark:text-white">
                                  {payload[0].name}
                                </p>
                                <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                                  {payload[0].value}% of threats
                                </p>
                              </div>
                            )
                          }
                          return null
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-4 mt-4">
                  {mockCategories.map((category, index) => (
                    <div key={category.name} className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: COLORS[index % COLORS.length] }}
                      />
                      <span className="text-sm text-gray-600 dark:text-gray-300">
                        {category.name}: {category.value}%
                      </span>
                    </div>
                  ))}
                </div>
              </DashboardCard>
            </div>

            {/* Trend Analysis Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatsCard
                title="Total Predictions"
                value={mockTrendData.reduce((acc, curr) => acc + curr.predictions, 0).toString()}
                change="+12%"
                trend="up"
                icon={Target}
                color="blue"
              />
              <StatsCard
                title="Average Accuracy"
                value={`${(mockTrendData.reduce((acc, curr) => acc + curr.accuracy, 0) / mockTrendData.length).toFixed(1)}%`}
                change="+2.3%"
                trend="up"
                icon={Activity}
                color="green"
              />
              <StatsCard
                title="Detected Threats"
                value={mockTrendData.reduce((acc, curr) => acc + curr.threats, 0).toString()}
                change="-5%"
                trend="down"
                icon={AlertTriangle}
                color="red"
              />
              <StatsCard
                title="Analysis Period"
                value="30 Days"
                change=""
                trend="stable"
                icon={Clock}
                color="yellow"
              />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="models">
          <div className="space-y-6">
            {/* Models Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatsCard
                title="Active Models"
                value={mockModels.length.toString()}
                change="+1"
                trend="up"
                icon={Brain}
                color="blue"
              />
              <StatsCard
                title="Average Accuracy"
                value={`${(mockModels.reduce((acc, curr) => acc + curr.accuracy, 0) / mockModels.length).toFixed(1)}%`}
                change="+1.5%"
                trend="up"
                icon={Target}
                color="green"
              />
              <StatsCard
                title="Total Predictions"
                value={mockModels.reduce((acc, curr) => acc + curr.predictions, 0).toLocaleString()}
                change="+234"
                trend="up"
                icon={Activity}
                color="yellow"
              />
              <StatsCard
                title="Success Rate"
                value={`${(mockModels.reduce((acc, curr) => acc + curr.successRate, 0) / mockModels.length).toFixed(1)}%`}
                change="+0.8%"
                trend="up"
                icon={Check}
                color="green"
              />
            </div>

            {/* Models List */}
            <DashboardCard title="ML Models Performance" icon={Brain}>
              <div className="space-y-6">
                {mockModels.map((model) => (
                  <div
                    key={model.id}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                            {model.name}
                          </h3>
                          <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200">
                            v{model.version}
                          </span>
                          <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200">
                            {model.type}
                          </span>
                        </div>
                        <div className="mt-2 grid grid-cols-3 gap-4">
                          <div>
                            <span className="text-sm text-gray-500 dark:text-gray-400">Accuracy</span>
                            <p className="text-lg font-semibold text-gray-900 dark:text-white">
                              {model.accuracy}%
                            </p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500 dark:text-gray-400">Predictions</span>
                            <p className="text-lg font-semibold text-gray-900 dark:text-white">
                              {model.predictions.toLocaleString()}
                            </p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500 dark:text-gray-400">Success Rate</span>
                            <p className="text-lg font-semibold text-gray-900 dark:text-white">
                              {model.successRate}%
                            </p>
                          </div>
                        </div>
                      </div>
                      <div className="flex flex-col items-end">
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          Last Trained
                        </span>
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {new Date(model.lastTrained).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                    <div className="mt-4 flex justify-end gap-2">
                      <button className="px-3 py-1 text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300">
                        View Details
                      </button>
                      <button className="px-3 py-1 text-sm text-green-600 hover:text-green-700 dark:text-green-400 dark:hover:text-green-300">
                        Retrain Model
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </DashboardCard>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
} 