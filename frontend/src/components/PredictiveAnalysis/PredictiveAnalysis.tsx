'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { 
  LineChart as LineChartIcon, Brain, AlertTriangle, Target, 
  Clock, Shield, Activity, TrendingUp, Search, Filter, 
  RefreshCw, ChevronDown, ArrowUpRight, Check, X, Server, Users, Lock, Zap 
} from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell 
} from 'recharts'

// Interfaces dari PredictionDetail.tsx
interface PredictionDetail {
  id: string
  type: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  probability: number
  impact: {
    financial: number
    operational: number
    reputational: number
  }
  timeline: {
    detected: string
    estimated: string
    window: string
  }
  source: {
    ip?: string
    location?: string
    actor?: string
    technique?: string
  }
  affectedSystems: {
    id: string
    name: string
    type: string
    criticality: string
    status: string
  }[]
  indicators: {
    id: string
    type: string
    value: string
    confidence: number
    firstSeen: string
    lastSeen: string
  }[]
  mitigationSteps: {
    id: string
    action: string
    priority: 'high' | 'medium' | 'low'
    status: 'pending' | 'in-progress' | 'completed'
    assignedTo?: string
    eta?: string
  }[]
  analysis: {
    summary: string
    methodology: string
    confidence: number
    falsePositiveRisk: number
    dataPoints: number
    modelVersion: string
    lastUpdated: string
  }
  relatedEvents: {
    id: string
    type: string
    timestamp: string
    description: string
  }[]
  recommendations: {
    id: string
    type: 'immediate' | 'short-term' | 'long-term'
    description: string
    impact: string
    effort: string
    status: 'proposed' | 'approved' | 'in-progress' | 'completed'
  }[]
}

// Mock data dan komponen PredictionDetail
function PredictionDetailView({ prediction, onClose }: { prediction: PredictionDetail, onClose: () => void }) {
  return (
    <DashboardCard className="relative">
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-gray-400 hover:text-gray-500"
      >
        <X className="h-5 w-5" />
      </button>

      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="indicators">Indicators</TabsTrigger>
          <TabsTrigger value="systems">Affected Systems</TabsTrigger>
          <TabsTrigger value="mitigation">Mitigation</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          {/* Overview content dari PredictionDetail.tsx */}
          <div className="space-y-6">
            {/* ... konten overview ... */}
          </div>
        </TabsContent>

        <TabsContent value="indicators">
          <div className="space-y-6">
            {prediction.indicators.map(indicator => (
              <div key={indicator.id} className="p-4 border rounded-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium">{indicator.type}</h4>
                    <p className="text-sm text-gray-500">{indicator.value}</p>
                  </div>
                  <span className="text-sm font-medium text-blue-500">
                    {indicator.confidence}% confidence
                  </span>
                </div>
                <div className="mt-2 text-sm text-gray-500">
                  First seen: {new Date(indicator.firstSeen).toLocaleString()}
                  <br />
                  Last seen: {new Date(indicator.lastSeen).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </TabsContent>

        {/* Implementasi tab lainnya */}
      </Tabs>
    </DashboardCard>
  )
}

// Tambahkan mock data
const mockPredictionDetails: PredictionDetail[] = [
  {
    id: '1',
    type: 'Advanced Persistent Threat',
    severity: 'critical',
    probability: 89.5,
    impact: {
      financial: 85,
      operational: 75,
      reputational: 90
    },
    timeline: {
      detected: new Date(Date.now() - 3600000).toISOString(),
      estimated: new Date(Date.now() + 86400000).toISOString(),
      window: '24-48 hours'
    },
    source: {
      ip: '192.168.1.100',
      location: 'Eastern Europe',
      actor: 'APT-29',
      technique: 'Supply Chain Compromise'
    },
    affectedSystems: [
      {
        id: 'sys-1',
        name: 'Primary Database Server',
        type: 'Database',
        criticality: 'High',
        status: 'At Risk'
      },
      {
        id: 'sys-2',
        name: 'Authentication Server',
        type: 'Identity Management',
        criticality: 'Critical',
        status: 'Monitoring'
      }
    ],
    indicators: [
      {
        id: 'ind-1',
        type: 'Network Traffic',
        value: 'Unusual data exfiltration pattern',
        confidence: 92,
        firstSeen: new Date(Date.now() - 86400000).toISOString(),
        lastSeen: new Date().toISOString()
      },
      {
        id: 'ind-2',
        type: 'System Access',
        value: 'Privileged account creation',
        confidence: 88,
        firstSeen: new Date(Date.now() - 43200000).toISOString(),
        lastSeen: new Date().toISOString()
      }
    ],
    mitigationSteps: [
      {
        id: 'mit-1',
        action: 'Isolate affected systems',
        priority: 'high',
        status: 'pending',
        assignedTo: 'Security Team',
        eta: new Date(Date.now() + 3600000).toISOString()
      },
      {
        id: 'mit-2',
        action: 'Block suspicious IPs',
        priority: 'high',
        status: 'in-progress',
        assignedTo: 'Network Team',
        eta: new Date(Date.now() + 1800000).toISOString()
      }
    ],
    analysis: {
      summary: 'High-confidence detection of APT activity targeting critical infrastructure',
      methodology: 'ML-based pattern recognition with behavioral analysis',
      confidence: 92,
      falsePositiveRisk: 8,
      dataPoints: 15420,
      modelVersion: '2.1.0',
      lastUpdated: new Date().toISOString()
    },
    relatedEvents: [
      {
        id: 'evt-1',
        type: 'Failed Login',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        description: 'Multiple failed login attempts from suspicious IP'
      }
    ],
    recommendations: [
      {
        id: 'rec-1',
        type: 'immediate',
        description: 'Enable additional authentication factors',
        impact: 'High',
        effort: 'Medium',
        status: 'proposed'
      }
    ]
  }
  // Tambahkan lebih banyak mock data sesuai kebutuhan
]

// Update komponen PredictiveAnalysis
export function PredictiveAnalysis() {
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionDetail | null>(null)
  const [timeRange, setTimeRange] = useState('7d')
  const [searchQuery, setSearchQuery] = useState('')
  const [filterCategory, setFilterCategory] = useState('all')

  return (
    <div className="space-y-6">
      {/* Header */}
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

      {/* Main content */}
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
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatsCard
              title="Active Predictions"
              value={mockPredictionDetails.length.toString()}
              change="+2"
              trend="up"
              icon={Brain}
              color="blue"
            />
            {/* Add more StatsCards */}
          </div>
          
          {/* Recent Predictions */}
          <DashboardCard title="Recent Predictions" icon={AlertTriangle} className="mt-6">
            <div className="space-y-4">
              {mockPredictionDetails.map((prediction) => (
                <div
                  key={prediction.id}
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800"
                  onClick={() => setSelectedPrediction(prediction)}
                >
                  {/* Prediction card content */}
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">
                        {prediction.type}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        {prediction.analysis.summary}
                      </p>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      prediction.severity === 'critical'
                        ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                        : 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200'
                    }`}>
                      {prediction.probability}% probability
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </DashboardCard>
        </TabsContent>

        {/* Implement other tabs */}
      </Tabs>

      {/* Detail modal */}
      {selectedPrediction && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <PredictionDetailView 
              prediction={selectedPrediction} 
              onClose={() => setSelectedPrediction(null)} 
            />
          </div>
        </div>
      )}
    </div>
  )
} 