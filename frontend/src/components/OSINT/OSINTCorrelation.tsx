'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { Link2, AlertTriangle } from 'lucide-react'
import { Progress } from '@/components/ui/progress'

interface CorrelationData {
  osintFinding: string
  prediction: string
  confidence: number
  impact: string
  recommendations: string[]
}

const mockCorrelations: CorrelationData[] = [
  {
    osintFinding: 'Exposed Git Repository',
    prediction: 'Potential Data Breach',
    confidence: 85,
    impact: 'Critical',
    recommendations: [
      'Secure exposed repositories',
      'Review access logs',
      'Update security policies'
    ]
  },
  {
    osintFinding: 'Dark Web Credentials',
    prediction: 'Account Takeover Attempt',
    confidence: 92,
    impact: 'High',
    recommendations: [
      'Force password reset',
      'Enable MFA',
      'Monitor suspicious logins'
    ]
  }
]

export function OSINTCorrelation() {
  return (
    <DashboardCard title="OSINT-Prediction Correlation" icon={Link2}>
      <div className="space-y-6">
        {mockCorrelations.map((correlation, index) => (
          <div key={index} className="p-4 border rounded-lg">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-orange-500" />
                  <h3 className="font-medium">{correlation.prediction}</h3>
                </div>
                <p className="text-sm text-gray-500 mt-1">
                  Based on: {correlation.osintFinding}
                </p>
              </div>
              <div className="text-right">
                <span className="text-sm font-medium text-blue-500">
                  {correlation.confidence}% confidence
                </span>
                <p className="text-xs text-gray-500">
                  Impact: {correlation.impact}
                </p>
              </div>
            </div>
            
            <div className="mt-4">
              <h4 className="text-sm font-medium mb-2">Recommendations</h4>
              <ul className="space-y-2">
                {correlation.recommendations.map((rec, idx) => (
                  <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-4">
              <div className="flex justify-between text-sm mb-1">
                <span>Correlation Strength</span>
                <span>{correlation.confidence}%</span>
              </div>
              <Progress value={correlation.confidence} className="h-2" />
            </div>
          </div>
        ))}
      </div>
    </DashboardCard>
  )
} 