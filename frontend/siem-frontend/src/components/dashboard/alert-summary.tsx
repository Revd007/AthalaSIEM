import { useQuery } from '@tanstack/react-query'
import { Card, CardHeader, CardTitle } from '../ui/card'
import { Alert } from '../../types/alert'

async function fetchAlertSummary(): Promise<Alert[]> {
  const response = await fetch('/api/alerts/summary')
  if (!response.ok) throw new Error('Failed to fetch alerts')
  return response.json()
}

interface AlertSummaryProps {
  alerts?: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

export default function AlertSummary({ alerts }: AlertSummaryProps) {
  if (!alerts) return null;

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="p-4 bg-red-100 rounded-lg">
        <p className="text-sm text-red-800">Critical</p>
        <p className="text-2xl font-bold text-red-900">{alerts.critical}</p>
      </div>
      <div className="p-4 bg-orange-100 rounded-lg">
        <p className="text-sm text-orange-800">High</p>
        <p className="text-2xl font-bold text-orange-900">{alerts.high}</p>
      </div>
      <div className="p-4 bg-yellow-100 rounded-lg">
        <p className="text-sm text-yellow-800">Medium</p>
        <p className="text-2xl font-bold text-yellow-900">{alerts.medium}</p>
      </div>
      <div className="p-4 bg-blue-100 rounded-lg">
        <p className="text-sm text-blue-800">Low</p>
        <p className="text-2xl font-bold text-blue-900">{alerts.low}</p>
      </div>
    </div>
  )
}