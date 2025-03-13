import { DashboardCard } from '@/components/ui/DashboardCard'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'
import { ChartPie } from 'lucide-react'

const data = [
  { name: 'Authentication', value: 45 },
  { name: 'Network', value: 30 },
  { name: 'System', value: 15 },
  { name: 'Application', value: 10 },
]

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

export function EventsDistribution() {
  return (
    <DashboardCard title="Events Distribution" icon={ChartPie}>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </DashboardCard>
  )
} 