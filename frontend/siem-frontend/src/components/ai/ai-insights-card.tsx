import { Card } from '../ui/card'
import { LoadingSkeleton } from '../ui/loading-skeleton'
import { BrainIcon } from 'lucide-react'
import { useAIAnalytics } from '../../hooks/use-ai-analytics'

export function AIInsightsCard() {
  const { data, isLoading } = useAIAnalytics()

  if (isLoading) {
    return <Card className="p-6"><LoadingSkeleton rows={4} /></Card>
  }

  return (
    <Card className="p-6 bg-gradient-to-br from-purple-50 to-purple-100">
      <h3 className="text-lg font-semibold mb-4">AI Insights</h3>
      <div className="space-y-4">
        {data?.insights.map((insight, i) => (
          <div key={i} className="flex items-start space-x-3">
            <div className="p-2 rounded-full bg-purple-200">
              <BrainIcon className="w-4 h-4 text-purple-700" />
            </div>
            <div>
              <p className="font-medium">{insight.title}</p>
              <p className="text-sm text-gray-600">{insight.description}</p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}