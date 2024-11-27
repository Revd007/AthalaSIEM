import { useQuery } from '@tanstack/react-query'
import { dashboardService } from '../services/dashboard-service'

export function useAIAnalytics() {
  return useQuery({
    queryKey: ['ai-insights'],
    queryFn: () => dashboardService.getAIInsights(),
    refetchInterval: 30000 // Refresh every 30 seconds
  })
}