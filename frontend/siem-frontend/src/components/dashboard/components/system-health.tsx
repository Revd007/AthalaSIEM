import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle } from '../../ui/card'

interface SystemStatus {
  name: string
  status: 'healthy' | 'warning' | 'critical'
  uptime: string
}

export function SystemHealth() {
  const [systems, setSystems] = useState<SystemStatus[]>([])

  useEffect(() => {
    // Fetch system status
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/system/health')
        const data = await response.json()
        setSystems(data)
      } catch (error) {
        console.error('Failed to fetch system health:', error)
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 60000) // Update every minute
    return () => clearInterval(interval)
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Health</CardTitle>
      </CardHeader>
      <div className="p-6">
        {systems.map((system) => (
          <div key={system.name} className="flex items-center justify-between py-2">
            <span>{system.name}</span>
            <div className="flex items-center gap-2">
              <span className={`h-3 w-3 rounded-full ${
                system.status === 'healthy' ? 'bg-green-500' :
                system.status === 'warning' ? 'bg-yellow-500' :
                'bg-red-500'
              }`} />
              <span>{system.uptime}</span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}