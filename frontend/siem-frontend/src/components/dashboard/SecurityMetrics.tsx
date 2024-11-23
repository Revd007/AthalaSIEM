import { Card, CardHeader, CardTitle } from '../ui/card'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface MetricData {
  timestamp: string
  value: number
}

interface SecurityMetricsProps {
  data?: MetricData[]
  isLoading?: boolean
}

export function SecurityMetrics({ data = [], isLoading }: SecurityMetricsProps) {
  if (isLoading) return <div>Loading metrics...</div>

  return (
    <Card>
      <CardHeader>
        <CardTitle>Security Metrics</CardTitle>
      </CardHeader>
      <div className="p-6 h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#8884d8" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}