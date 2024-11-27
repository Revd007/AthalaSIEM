import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { StatCard } from '../../components/ui/stat-card'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { Cpu, HardDrive, Activity, MemoryStick } from 'lucide-react'
import { useAIAnalytics } from '../../hooks/use-ai-analytics'

// Define proper types
interface SystemMetrics {
  cpu: {
    usage: number;
  };
  memory: {
    used: number;
    total: number;
  };
  storage: {
    used: number;
    total: number;
    free: number;
  };
  network: {
    incoming: number;
    outgoing: number;
  };
}

interface HealthComponent {
  status: 'up' | 'down';
  latency: number;
}

interface SystemHealth {
  components: Record<string, HealthComponent>;
}

export function SystemMonitor() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [historicalData, setHistoricalData] = useState<any[]>([])
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [mounted, setMounted] = useState(true)
  const { data: aiData, isLoading: aiLoading } = useAIAnalytics()

  useEffect(() => {
    setMounted(true)
    
    const fetchData = async () => {
      if (!mounted) return
      
      try {
        setLoading(true)
        // Simulasi data untuk development
        const mockMetrics: SystemMetrics = {
          cpu: { usage: 45.5 },
          memory: { used: 8.2, total: 16 },
          storage: { used: 256, total: 512, free: 256 },
          network: { incoming: 1024 * 1024, outgoing: 512 * 1024 }
        }
        setMetrics(mockMetrics)

        // Mock historical data
        setHistoricalData([
          { timestamp: '00:00', cpu: { usage: 40 }, memory: { used: 7.8 } },
          { timestamp: '00:05', cpu: { usage: 42 }, memory: { used: 8.0 } },
          // ... more data points
        ])

        // Mock health data
        setHealth({
          components: {
            'Database': { status: 'up', latency: 45 },
            'API Server': { status: 'up', latency: 23 },
            'Cache': { status: 'up', latency: 12 }
          }
        })
      } catch (error) {
        console.error('Error fetching system metrics:', error)
      } finally {
        if (mounted) {
          setLoading(false)
        }
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000)

    return () => {
      setMounted(false)
      clearInterval(interval)
    }
  }, [])

  // Skip rendering if unmounted
  if (!mounted) return null

  if (loading || !metrics) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="CPU Usage"
          value={`${metrics.cpu.usage.toFixed(1)}%`}
          icon={<Cpu className="h-6 w-6 text-blue-500" />}
          variant={metrics.cpu.usage > 80 ? 'danger' : 'default'}
        />
        <StatCard
          title="Memory Usage"
          value={`${((metrics.memory.used / metrics.memory.total) * 100).toFixed(1)}%`}
          icon={<MemoryStick className="h-6 w-6 text-green-500" />}
          description={`${metrics.memory.used.toFixed(2)}GB / ${metrics.memory.total.toFixed(2)}GB`}
        />
        <StatCard
          title="Storage"
          value={`${((metrics.storage.used / metrics.storage.total) * 100).toFixed(1)}%`}
          icon={<HardDrive className="h-6 w-6 text-yellow-500" />}
          description={`${metrics.storage.free.toFixed(2)}GB free`}
        />
        <StatCard
          title="Network Traffic"
          value={`${(metrics.network.incoming / 1024 / 1024).toFixed(2)} MB/s`}
          icon={<Activity className="h-6 w-6 text-purple-500" />}
          description={`Out: ${(metrics.network.outgoing / 1024 / 1024).toFixed(2)} MB/s`}
        />
        {aiData && (
          <StatCard
            title="Anomaly Score"
            value={`${(aiData.anomalyDetection.score * 100).toFixed(1)}%`}
            icon={<Activity className="h-4 w-4" />}
            description="AI-detected system anomalies"
            trend={{
              direction: aiData.anomalyDetection.trend > 0 ? 'up' : 
                        aiData.anomalyDetection.trend < 0 ? 'down' : 'neutral',
              value: Math.abs(aiData.anomalyDetection.trend)
            }}
          />
        )}
      </div>

      {/* Performance Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-lg border bg-white p-6"
      >
        <h3 className="mb-4 text-lg font-medium">System Performance</h3>
        <div className="h-[300px]">
          {historicalData.length > 0 && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={historicalData}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <XAxis 
                  dataKey="timestamp"
                  stroke="#888888"
                  fontSize={12}
                />
                <YAxis
                  stroke="#888888"
                  fontSize={12}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '6px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="cpu.usage"
                  stroke="#3b82f6"
                  name="CPU"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="memory.used"
                  stroke="#10b981"
                  name="Memory"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </motion.div>

      {/* Health and Alerts Grid */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {health && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="rounded-lg border bg-white p-6"
          >
            <h3 className="mb-4 text-lg font-medium">System Health</h3>
            <div className="space-y-4">
              {Object.entries(health.components).map(([name, data]) => (
                <div key={name} className="flex items-center justify-between">
                  <span className="text-sm font-medium">{name}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-500">{data.latency}ms</span>
                    <span
                      className={`h-2 w-2 rounded-full ${
                        data.status === 'up' ? 'bg-green-500' : 'bg-red-500'
                      }`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export function getPatternSeverityColor(severity: number) {
  if (severity > 0.8) return 'bg-red-500'
  if (severity > 0.6) return 'bg-orange-500'
  if (severity > 0.4) return 'bg-yellow-500'
  return 'bg-green-500'
}

export function getThreatSeverityVariant(severity: number) {
  if (severity > 0.8) return 'destructive'
  if (severity > 0.6) return 'warning'
  return 'secondary'
}