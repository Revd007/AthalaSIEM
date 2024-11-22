import { useQuery } from '@tanstack/react-query'
import { Card, CardHeader, CardTitle } from '../ui/card'
import { Alert } from '../../types/alert'

async function fetchAlertSummary(): Promise<Alert[]> {
  const response = await fetch('/api/alerts/summary')
  if (!response.ok) throw new Error('Failed to fetch alerts')
  return response.json()
}

export default function AlertSummary() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['alertSummary'],
    queryFn: fetchAlertSummary
  })

  if (isLoading) return <div>Loading...</div>
  if (error) return <div>Error loading alerts</div>

  return (
    <Card>
      <CardHeader>
        <CardTitle>Alert Summary</CardTitle>
      </CardHeader>
      <div className="p-6">
        {data?.map((alert) => (
          <div key={alert.id} className="flex items-center justify-between py-2 border-b last:border-0">
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${
                alert.severity === 'high' ? 'bg-red-500' :
                alert.severity === 'medium' ? 'bg-yellow-500' :
                'bg-blue-500'
              }`} />
              <span className="font-medium">{alert.title}</span>
            </div>
            <span className="text-sm text-gray-500">{alert.timestamp}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}