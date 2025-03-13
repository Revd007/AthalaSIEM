import { useQuery } from '@tanstack/react-query'
import type { UseQueryOptions } from '@tanstack/react-query'
import { api, endpoints } from '../lib/api'
import type { 
  AIStatusResponse, 
  SecurityEventResponse,
  SystemMetricsResponse,
  KnowledgeGraphResponse,
  AIStatus 
} from '../types/api'

export const useDonquixoteService = () => {
  const useAIAnalysis = () => {
    return useQuery<AIStatus, Error>({
      queryKey: ['ai-analysis'],
      queryFn: async () => {
        const response = await api.get<AIStatusResponse>(endpoints.donquixote.aiStatus);
        if (!response?.data?.data?.status) {
          throw new Error('Invalid AI status response');
        }
        return response.data.data.status;
      },
      retry: 3,
      retryDelay: (attemptIndex: number) => Math.min(1000 * 2 ** attemptIndex, 30000),
      refetchInterval: 30000
    })
  }

  const useKnowledgeGraph = () => {
    return useQuery<KnowledgeGraphResponse['data']>({
      queryKey: ['knowledge-graph'],
      queryFn: async () => {
        const response = await api.get<KnowledgeGraphResponse>(endpoints.donquixote.knowledgeGraph);
        if (!response?.data?.data) {
          throw new Error('Invalid knowledge graph response');
        }
        return response.data.data;
      }
    })
  }

  const useSystemMetrics = () => {
    return useQuery<SystemMetricsResponse['data']>({
      queryKey: ['system-metrics'],
      queryFn: async () => {
        const response = await api.get<SystemMetricsResponse>(endpoints.donquixote.systemMetrics);
        if (!response?.data?.data) {
          throw new Error('Invalid system metrics response');
        }
        return response.data.data;
      }
    })
  }

  const useRecentEvents = () => {
    return useQuery<SecurityEventResponse['data']>({
      queryKey: ['recent-events'],
      queryFn: async () => {
        const response = await api.get<SecurityEventResponse>(endpoints.donquixote.recentEvents);
        if (!response?.data?.data) {
          throw new Error('Invalid events response');
        }
        return response.data.data;
      }
    })
  }

  return {
    useAIAnalysis,
    useKnowledgeGraph,
    useSystemMetrics,
    useRecentEvents
  }
} 