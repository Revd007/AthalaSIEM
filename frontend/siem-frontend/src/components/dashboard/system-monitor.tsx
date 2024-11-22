import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MonitoringService, SystemMetrics } from '../../services/monitoring-service'
import { StatCard } from '../ui/stat-card'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { Cpu, Memory, HardDrive, Activity } from 'lucide-react'

export function SystemMonitor() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [historicalData, setHistoricalData] = useState<any[]>([])
  const [health, setHealth] = useState<any>(null)

  useEffect(() => {
    // Subscribe to real-time metrics
    const unsubscribe = MonitoringService.subscribeToMetrics((newMetrics) => {
      setMetrics(newMetrics)
    })

    // Fetch historical data
    const fetchHistorical = async () => {
      try {
        const data = await MonitoringService.getHistoricalMetrics('1h')
        setHistoricalData(data.metrics)
      } catch (error) {
        console.error('Failed to fetch historical metrics:', error)
      }
    }

    // Fetch system health
    const fetchHealth = async () => {
      try {
        const healthData = await MonitoringService.getSystemHealth()
        setHealth(healthData)
      } catch (error) {
        console.error('Failed to fetch system health:', error)
      }
    }

    fetchHistorical()
    fetchHealth()

    // Cleanup subscription
    return () => {
      unsubscribe()
    }
  }, [])

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="CPU Usage"
          value={`${metrics?.cpu.usage.toFixed(1)}%`}
          icon={<Cpu className="h-6 w-6 text-blue-500" />}
          variant={metrics?.cpu.usage > 80 ? 'danger' : 'default'}
        />
        <StatCard
          title="Memory Usage"
          value={`${((metrics?.memory.used || 0) / (metrics?.memory.total || 1) * 100).toFixed(1)}%`}
          icon={<Memory className="h-6 w-6 text-green-500" />}
          description={`${(metrics?.memory.used || 0).toFixed(2)}GB / ${(metrics?.memory.total || 0).toFixed(2)}GB`}
        />
        <StatCard
          title="Storage"
          value={`${((metrics?.storage.used || 0) / (metrics?.storage.total || 1) * 100).toFixed(1)}%`}
          icon={<HardDrive className="h-6 w-6 text-yellow-500" />}
          description={`${(metrics?.storage.free || 0).toFixed(2)}GB free`}
        />
        <StatCard
          title="Network Traffic"
          value={`${((metrics?.network.incoming || 0) / 1024 / 1024).toFixed(2)} MB/s`}
          icon={<Activity className="h-6 w-6 text-purple-500" />}
          description={`Out: ${((metrics?.network.outgoing || 0) / 1024 / 1024).toFixed(2)} MB/s`}
        />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-lg border bg-white p-6"
      >
        <h3 className="mb-4 text-lg font-medium">System Performance</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={historicalData}>
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="cpu.usage"
                stroke="#3b82f6"
                name="CPU"
              />
              <Line
                type="monotone"
                dataKey="memory.used"
                stroke="#10b981"
                name="Memory"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="rounded-lg border bg-white p-6"
        >
          <h3 className="mb-4 text-lg font-medium">System Health</h3>
          <div className="space-y-4">
            {health?.components && Object.entries(health.components).map(([name, data]: [string, any]) => (
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

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="rounded-lg border bg-white p-6"
        >
          <h3 className="mb-4 text-lg font-medium">Active Alerts</h3>
          {/* Add alerts component here */}
        </motion.div>
      </div>
    </div>
  )
}