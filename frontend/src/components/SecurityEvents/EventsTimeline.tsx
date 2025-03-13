import { DashboardCard } from '@/components/ui/DashboardCard'
import { Clock } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const mockData = Array.from({ length: 24 }, (_, i) => ({
  time: `${i}:00`,
  events: Math.floor(Math.random() * 100),
  anomalies: Math.floor(Math.random() * 20),
}))

export function EventsTimeline() {
  return (
    <DashboardCard title="Events Timeline" icon={Clock}>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={mockData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="time"
              tick={{ fontSize: 12 }}
              tickLine={false}
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: 'none',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
              }}
            />
            <Line
              type="monotone"
              dataKey="events"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Events"
            />
            <Line
              type="monotone"
              dataKey="anomalies"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              name="Anomalies"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Timeline Legend */}
      <div className="flex items-center justify-center space-x-6 mt-4">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-blue-500 mr-2" />
          <span className="text-sm text-gray-600 dark:text-gray-400">Normal Events</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-red-500 mr-2" />
          <span className="text-sm text-gray-600 dark:text-gray-400">Anomalies</span>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="text-sm text-gray-500 dark:text-gray-400">Peak Events</div>
          <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">
            {Math.max(...mockData.map(d => d.events))}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            at {mockData[mockData.findIndex(d => d.events === Math.max(...mockData.map(d => d.events)))].time}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="text-sm text-gray-500 dark:text-gray-400">Total Anomalies</div>
          <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">
            {mockData.reduce((acc, curr) => acc + curr.anomalies, 0)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            in last 24 hours
          </div>
        </div>
      </div>
    </DashboardCard>
  )
} 