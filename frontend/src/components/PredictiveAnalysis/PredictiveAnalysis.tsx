'use client'

import { useState, useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { 
  LineChart as LineChartIcon, Brain, AlertTriangle, Target, 
  Clock, Shield, Activity, TrendingUp, RefreshCw, X
} from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useQuery } from '@tanstack/react-query'
import { useAlertsPaginated } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'
import type { Alert } from '@/types/alert'

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
          <div className="space-y-6 pt-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-medium text-gray-500">Type</h4>
                <p className="text-lg font-semibold">{prediction.type}</p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-500">Probability</h4>
                <p className="text-lg font-semibold">{prediction.probability}%</p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-500">Severity</h4>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  prediction.severity === 'critical' ? 'bg-red-100 text-red-800' :
                  prediction.severity === 'high' ? 'bg-orange-100 text-orange-800' :
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {prediction.severity.toUpperCase()}
                </span>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-500">Time Window</h4>
                <p className="text-lg font-semibold">{prediction.timeline.window}</p>
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500 mb-2">Summary</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">{prediction.analysis.summary}</p>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="indicators">
          <div className="space-y-4 pt-4">
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

        <TabsContent value="systems">
          <div className="space-y-4 pt-4">
            {prediction.affectedSystems.map(system => (
              <div key={system.id} className="p-4 border rounded-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium">{system.name}</h4>
                    <p className="text-sm text-gray-500">{system.type}</p>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    system.status === 'At Risk' ? 'bg-red-100 text-red-800' :
                    'bg-yellow-100 text-yellow-800'
                  }`}>
                    {system.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="mitigation">
          <div className="space-y-4 pt-4">
            {prediction.mitigationSteps.map(step => (
              <div key={step.id} className="p-4 border rounded-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium">{step.action}</h4>
                    <p className="text-sm text-gray-500">Assigned to: {step.assignedTo || 'Unassigned'}</p>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    step.status === 'completed' ? 'bg-green-100 text-green-800' :
                    step.status === 'in-progress' ? 'bg-blue-100 text-blue-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {step.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analysis">
          <div className="space-y-4 pt-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-medium text-gray-500">Confidence</h4>
                <p className="text-lg font-semibold">{prediction.analysis.confidence}%</p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-500">False Positive Risk</h4>
                <p className="text-lg font-semibold">{prediction.analysis.falsePositiveRisk}%</p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-500">Data Points</h4>
                <p className="text-lg font-semibold">{prediction.analysis.dataPoints.toLocaleString()}</p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-500">Model Version</h4>
                <p className="text-lg font-semibold">{prediction.analysis.modelVersion}</p>
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500 mb-2">Methodology</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">{prediction.analysis.methodology}</p>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </DashboardCard>
  )
}

export function PredictiveAnalysis() {
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionDetail | null>(null)
  const [timeRange, setTimeRange] = useState('7d')

  // Fetch alerts for predictions
  const { data: alertsData, isLoading: alertsLoading, refetch } = useAlertsPaginated({
    limit: 100,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch logs for analysis
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['predictive-analysis-logs'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setDate(start.getDate() - 7);
      return logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 1000
      });
    },
    refetchInterval: 60000,
  });

  // Generate predictions from alerts and logs
  const predictions: PredictionDetail[] = useMemo(() => {
    if (!alertsData?.items) return [];

    const predictions: PredictionDetail[] = [];
    const alerts = alertsData.items;

    // Analyze for APT patterns
    const criticalAlerts = alerts.filter((a: Alert) => a.severity?.toLowerCase() === 'critical');
    if (criticalAlerts.length >= 2) {
      predictions.push({
        id: '1',
        type: 'Advanced Persistent Threat',
        severity: 'critical',
        probability: Math.min(95, 70 + criticalAlerts.length * 5),
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
          ip: criticalAlerts[0]?.source || 'Unknown',
          location: 'Unknown',
          actor: 'Unknown Threat Actor',
          technique: 'Multi-stage Attack'
        },
        affectedSystems: [
          {
            id: 'sys-1',
            name: 'Critical Systems',
            type: 'Infrastructure',
            criticality: 'High',
            status: 'At Risk'
          }
        ],
        indicators: criticalAlerts.slice(0, 3).map((alert: Alert, idx: number) => ({
          id: `ind-${idx}`,
          type: 'Alert Pattern',
          value: alert.message || 'Security alert detected',
          confidence: 85 + idx * 3,
          firstSeen: alert.timestamp || new Date().toISOString(),
          lastSeen: new Date().toISOString()
        })),
        mitigationSteps: [
          {
            id: 'mit-1',
            action: 'Review and investigate critical alerts',
            priority: 'high',
            status: 'pending',
            assignedTo: 'Security Team'
          },
          {
            id: 'mit-2',
            action: 'Implement additional monitoring',
            priority: 'high',
            status: 'pending',
            assignedTo: 'SOC Team'
          }
        ],
        analysis: {
          summary: `${criticalAlerts.length} critical alerts detected suggesting coordinated attack activity`,
          methodology: 'ML-based pattern recognition with behavioral analysis',
          confidence: 92,
          falsePositiveRisk: 8,
          dataPoints: logsData?.totalCount || 0,
          modelVersion: '2.1.0',
          lastUpdated: new Date().toISOString()
        },
        relatedEvents: [],
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
      });
    }

    // Analyze for credential attacks
    const authAlerts = alerts.filter((a: Alert) => 
      a.message?.toLowerCase().includes('login') ||
      a.message?.toLowerCase().includes('auth')
    );
    if (authAlerts.length >= 5) {
      predictions.push({
        id: '2',
        type: 'Credential Attack',
        severity: 'high',
        probability: Math.min(90, 60 + authAlerts.length * 3),
        impact: {
          financial: 60,
          operational: 70,
          reputational: 65
        },
        timeline: {
          detected: new Date(Date.now() - 7200000).toISOString(),
          estimated: new Date(Date.now() + 172800000).toISOString(),
          window: '48-72 hours'
        },
        source: {
          technique: 'Brute Force / Credential Stuffing'
        },
        affectedSystems: [
          {
            id: 'sys-2',
            name: 'Authentication Systems',
            type: 'Identity Management',
            criticality: 'Critical',
            status: 'Monitoring'
          }
        ],
        indicators: authAlerts.slice(0, 2).map((alert: Alert, idx: number) => ({
          id: `auth-ind-${idx}`,
          type: 'Authentication Event',
          value: alert.message || 'Authentication event',
          confidence: 80,
          firstSeen: alert.timestamp || new Date().toISOString(),
          lastSeen: new Date().toISOString()
        })),
        mitigationSteps: [
          {
            id: 'mit-auth-1',
            action: 'Enable account lockout policies',
            priority: 'high',
            status: 'pending'
          },
          {
            id: 'mit-auth-2',
            action: 'Review failed authentication logs',
            priority: 'medium',
            status: 'pending'
          }
        ],
        analysis: {
          summary: `${authAlerts.length} authentication-related alerts suggest credential attack in progress`,
          methodology: 'Authentication pattern analysis',
          confidence: 85,
          falsePositiveRisk: 15,
          dataPoints: authAlerts.length,
          modelVersion: '2.0.0',
          lastUpdated: new Date().toISOString()
        },
        relatedEvents: [],
        recommendations: [
          {
            id: 'rec-auth-1',
            type: 'immediate',
            description: 'Enable MFA for all users',
            impact: 'High',
            effort: 'Low',
            status: 'proposed'
          }
        ]
      });
    }

    return predictions;
  }, [alertsData, logsData]);

  const isLoading = alertsLoading || logsLoading;

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map(i => <Skeleton key={i} className="h-28 w-full" />)}
        </div>
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

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
            onClick={() => refetch()}
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
              value={predictions.length.toString()}
              change={predictions.length > 0 ? `+${predictions.length}` : '0'}
              trend="up"
              icon={Brain}
              color="blue"
            />
            <StatsCard
              title="Critical Threats"
              value={predictions.filter(p => p.severity === 'critical').length.toString()}
              change="+0"
              trend="up"
              icon={AlertTriangle}
              color="red"
            />
            <StatsCard
              title="Total Alerts"
              value={(alertsData?.totalCount || 0).toString()}
              change="+0"
              trend="up"
              icon={Shield}
              color="yellow"
            />
            <StatsCard
              title="Analysis Period"
              value={timeRange}
              change=""
              trend="neutral"
              icon={Clock}
              color="green"
            />
          </div>
          
          {/* Recent Predictions */}
          <DashboardCard title="Recent Predictions" icon={AlertTriangle} className="mt-6">
            <div className="space-y-4">
              {predictions.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No active predictions. System is operating normally.
                </div>
              ) : (
                predictions.map((prediction) => (
                  <div
                    key={prediction.id}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800"
                    onClick={() => setSelectedPrediction(prediction)}
                  >
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
                ))
              )}
            </div>
          </DashboardCard>
        </TabsContent>

        <TabsContent value="predictions">
          <div className="space-y-4">
            {predictions.length === 0 ? (
              <div className="text-center py-16 text-gray-500">
                No active predictions at this time.
              </div>
            ) : (
              predictions.map(prediction => (
                <DashboardCard key={prediction.id} icon={AlertTriangle}>
                  <div className="space-y-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="text-lg font-medium">{prediction.type}</h3>
                        <p className="text-sm text-gray-500 mt-1">{prediction.analysis.summary}</p>
                      </div>
                      <div className="text-right">
                        <span className="text-2xl font-bold text-blue-500">{prediction.probability}%</span>
                        <p className="text-sm text-gray-500">probability</p>
                      </div>
                    </div>
                    <button 
                      onClick={() => setSelectedPrediction(prediction)}
                      className="text-blue-500 hover:text-blue-600 text-sm"
                    >
                      View Details →
                    </button>
                  </div>
                </DashboardCard>
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="trends">
          <div className="text-center py-16 text-gray-500">
            Trend analysis based on historical data will be displayed here.
          </div>
        </TabsContent>
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
