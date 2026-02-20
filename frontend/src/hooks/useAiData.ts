'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { aiApi, aiKeys } from '@/lib/ai-api'

const REFETCH_MS = 15 * 1000 // 15s polling for live tabs

export function useAiOverview() {
  return useQuery({
    queryKey: aiKeys.overview,
    queryFn: () => aiApi.getOverview(),
    staleTime: REFETCH_MS,
    refetchOnWindowFocus: true,
  })
}

export function useAiAnomalies() {
  return useQuery({
    queryKey: aiKeys.anomalies,
    queryFn: () => aiApi.getAnomalies(),
    staleTime: REFETCH_MS,
    refetchOnWindowFocus: true,
  })
}

export function useAiBehavior() {
  return useQuery({
    queryKey: aiKeys.behavior,
    queryFn: () => aiApi.getBehavior(),
    staleTime: REFETCH_MS,
    refetchOnWindowFocus: true,
  })
}

export function useAiPredictive() {
  return useQuery({
    queryKey: aiKeys.predictive,
    queryFn: () => aiApi.getPredictive(),
    staleTime: REFETCH_MS,
    refetchOnWindowFocus: true,
  })
}

export function useAiAutomatedResponse() {
  return useQuery({
    queryKey: aiKeys.automatedResponse,
    queryFn: () => aiApi.getAutomatedResponse(),
    staleTime: REFETCH_MS,
    refetchOnWindowFocus: true,
  })
}

export function useAiOsint() {
  return useQuery({
    queryKey: aiKeys.osint,
    queryFn: () => aiApi.getOsint(),
    staleTime: REFETCH_MS,
    refetchOnWindowFocus: true,
  })
}

export function useHuntDashboard() {
  return useQuery({
    queryKey: aiKeys.huntDashboard,
    queryFn: () => aiApi.getHuntDashboard(),
    staleTime: REFETCH_MS,
    refetchOnWindowFocus: true,
  })
}

export function useHuntBehavior() {
  return useQuery({
    queryKey: aiKeys.huntBehavior,
    queryFn: () => aiApi.getHuntBehavior(),
    staleTime: REFETCH_MS,
    refetchOnWindowFocus: true,
  })
}

export function useIocScan() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: { value: string; types?: string[] }) => aiApi.scanIoc(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['threat-hunting', 'ioc-scan'] })
    },
  })
}

export function useLiveHuntStart() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: { query: string; timeRangeMinutes?: number }) => aiApi.liveHuntStart(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['threat-hunting', 'live'] })
    },
  })
}

export function useLiveHuntResults(sessionId: string | null) {
  return useQuery({
    queryKey: aiKeys.huntLive(sessionId),
    queryFn: () => (sessionId ? aiApi.liveHuntResults(sessionId) : Promise.resolve(null)),
    enabled: !!sessionId,
    refetchInterval: sessionId ? 3000 : false,
  })
}
