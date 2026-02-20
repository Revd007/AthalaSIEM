/**
 * File Integrity Monitoring (FIM) API service and React Query hooks.
 * Backend: .NET api/FileIntegrity (FileIntegrityController).
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type {
  FIMEvent,
  FIMRule,
  FIMStats,
  PagedFIMEvents,
  FIMEventsQuery,
  CreateFIMRuleRequest,
} from '@/types/fim'

const FIM_BASE = '/api/FileIntegrity'

export const fimKeys = {
  all: ['fim'] as const,
  events: (params?: FIMEventsQuery) => [...fimKeys.all, 'events', params] as const,
  event: (id: string) => [...fimKeys.all, 'event', id] as const,
  rules: () => [...fimKeys.all, 'rules'] as const,
  rule: (id: string) => [...fimKeys.all, 'rule', id] as const,
  statistics: (agentId?: string, days?: number) =>
    [...fimKeys.all, 'statistics', agentId, days] as const,
}

async function fetchFIMEvents(params: FIMEventsQuery = {}): Promise<PagedFIMEvents> {
  const search = new URLSearchParams()
  if (params.agentId) search.set('agentId', params.agentId)
  if (params.severity) search.set('severity', params.severity)
  if (params.changeType) search.set('changeType', params.changeType)
  if (params.acknowledged !== undefined) search.set('acknowledged', String(params.acknowledged))
  if (params.startDate) search.set('startDate', params.startDate)
  if (params.endDate) search.set('endDate', params.endDate)
  search.set('page', String(params.page ?? 1))
  search.set('pageSize', String(params.pageSize ?? 50))
  const { data } = await api.get<PagedFIMEvents>(`${FIM_BASE}/events?${search.toString()}`)
  if (!data) throw new Error('No data')
  return data
}

async function fetchFIMEvent(id: string): Promise<FIMEvent> {
  const { data } = await api.get<FIMEvent>(`${FIM_BASE}/events/${id}`)
  if (!data) throw new Error('No data')
  return data
}

async function fetchFIMRules(): Promise<FIMRule[]> {
  const { data } = await api.get<FIMRule[]>(`${FIM_BASE}/rules`)
  if (!data) throw new Error('No data')
  return data
}

async function fetchFIMRule(id: string): Promise<FIMRule> {
  const { data } = await api.get<FIMRule>(`${FIM_BASE}/rules/${id}`)
  if (!data) throw new Error('No data')
  return data
}

async function fetchFIMStatistics(
  agentId?: string,
  days: number = 7
): Promise<FIMStats> {
  const search = new URLSearchParams()
  if (agentId) search.set('agentId', agentId)
  search.set('days', String(days))
  const { data } = await api.get<FIMStats>(`${FIM_BASE}/statistics?${search.toString()}`)
  if (!data) throw new Error('No data')
  return data
}

async function acknowledgeFIMEvents(eventIds: string[]): Promise<void> {
  await api.post(`${FIM_BASE}/events/acknowledge`, { eventIds })
}

async function createFIMRule(body: CreateFIMRuleRequest): Promise<FIMRule> {
  const { data } = await api.post<FIMRule>(`${FIM_BASE}/rules`, body)
  if (!data) throw new Error('No data')
  return data
}

async function updateFIMRule(id: string, body: CreateFIMRuleRequest): Promise<FIMRule> {
  const { data } = await api.put<FIMRule>(`${FIM_BASE}/rules/${id}`, body)
  if (!data) throw new Error('No data')
  return data
}

async function deleteFIMRule(id: string): Promise<void> {
  await api.delete(`${FIM_BASE}/rules/${id}`)
}

/** Paginated FIM events with filters */
export function useFIMEvents(params: FIMEventsQuery = {}) {
  return useQuery({
    queryKey: fimKeys.events(params),
    queryFn: () => fetchFIMEvents(params),
    staleTime: 30_000,
    refetchOnWindowFocus: true,
  })
}

/** Single FIM event by ID */
export function useFIMEvent(id: string | null) {
  return useQuery({
    queryKey: fimKeys.event(id ?? ''),
    queryFn: () => fetchFIMEvent(id!),
    enabled: !!id,
  })
}

/** All FIM rules */
export function useFIMRules() {
  return useQuery({
    queryKey: fimKeys.rules(),
    queryFn: fetchFIMRules,
    staleTime: 60_000,
  })
}

/** Single FIM rule by ID */
export function useFIMRule(id: string | null) {
  return useQuery({
    queryKey: fimKeys.rule(id ?? ''),
    queryFn: () => fetchFIMRule(id!),
    enabled: !!id,
  })
}

/** FIM statistics (events by severity, change type, agent, over time) */
export function useFIMStats(agentId?: string, days: number = 7) {
  return useQuery({
    queryKey: fimKeys.statistics(agentId, days),
    queryFn: () => fetchFIMStatistics(agentId, days),
    staleTime: 30_000,
  })
}

/** Acknowledge FIM events */
export function useAcknowledgeFIMEvents() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: acknowledgeFIMEvents,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fimKeys.all })
    },
  })
}

/** Create FIM rule */
export function useCreateFIMRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: createFIMRule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fimKeys.rules() })
    },
  })
}

/** Update FIM rule */
export function useUpdateFIMRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: CreateFIMRuleRequest }) =>
      updateFIMRule(id, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fimKeys.all })
    },
  })
}

/** Delete FIM rule */
export function useDeleteFIMRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: deleteFIMRule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fimKeys.rules() })
    },
  })
}
