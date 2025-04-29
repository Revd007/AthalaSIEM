import { DashboardCard } from '@/components/ui/DashboardCard'
import { Activity, AlertTriangle, Shield, Clock } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { StatsCard } from './StatsCard'

interface TimeRange {
  start: Date
  end: Date
}

interface SecurityFilters {
  severity?: string
  type?: string
  source?: string
}

interface SecurityEventsOverviewProps {
  timeRange: TimeRange
  filters: SecurityFilters
  onFilterChange: (filters: SecurityFilters) => void
}

const mockData = Array.from({ length: 24 }, (_, i) => ({
  time: `${i}:00`,
  total: Math.floor(Math.random() * 1000),
  critical: Math.floor(Math.random() * 100),
  high: Math.floor(Math.random() * 200),
  medium: Math.floor(Math.random() * 300),
}))

export function SecurityEventsOverview({ timeRange, filters, onFilterChange }: SecurityEventsOverviewProps) {
  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard 
          title="Total Events"
          value="157,893"
          change="+12.5%"
          icon={Activity}
          trend="up"
        />
        <StatsCard 
          title="Critical Events"
          value="23"
          change="-5.2%"
          icon={AlertTriangle}
          trend="down"
          color="red"
        />
        <StatsCard 
          title="Avg Response Time"
          value="1.2s"
          change="-1.8%"
          icon={Clock}
          trend="down"
          color="green"
        />
        <StatsCard 
          title="Threats Blocked"
          value="1,284"
          change="+8.1%"
          icon={Shield}
          trend="up"
          color="blue"
        />
      </div>

      {/* Events Timeline Chart */}
      <DashboardCard title="Events Over Time" icon={Activity}>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={mockData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="critical" 
                stackId="1"
                stroke="#ef4444" 
                fill="#fee2e2" 
                name="Critical"
              />
              <Area 
                type="monotone" 
                dataKey="high" 
                stackId="1"
                stroke="#f97316" 
                fill="#ffedd5" 
                name="High"
              />
              <Area 
                type="monotone" 
                dataKey="medium" 
                stackId="1"
                stroke="#eab308" 
                fill="#fef3c7" 
                name="Medium"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </DashboardCard>
    </div>
  )
} 