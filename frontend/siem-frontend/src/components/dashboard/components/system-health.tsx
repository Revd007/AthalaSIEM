import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle } from '../../ui/card'

interface HealthData {
  status: string;
  components: {
    name: string;
    status: 'healthy' | 'warning' | 'critical';
    uptime: string;
  }[];
}

interface SystemHealthProps {
  healthData?: HealthData;
}

export function SystemHealth({ healthData }: SystemHealthProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">System Health</h3>
      <div className="space-y-2">
        {healthData?.components.map((component) => (
          <div key={component.name} className="flex items-center justify-between py-2">
            <span>{component.name}</span>
            <div className="flex items-center gap-2">
              <span className={`h-3 w-3 rounded-full ${
                component.status === 'healthy' ? 'bg-green-500' :
                component.status === 'warning' ? 'bg-yellow-500' :
                'bg-red-500'
              }`} />
              <span>{component.uptime}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}