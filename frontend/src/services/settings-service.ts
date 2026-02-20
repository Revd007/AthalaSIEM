/**
 * System settings API: GET/PUT by category with backend persistence.
 * Backend: GET/PUT /api/settings/{category}
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export const settingsKeys = {
  all: ['settings'] as const,
  category: (category: string) => [...settingsKeys.all, category] as const,
}

const SETTINGS_BASE = '/api/settings'

export type SettingsCategory =
  | 'security'
  | 'agents'
  | 'monitoring'
  | 'notifications'
  | 'network'
  | 'compliance'
  | 'backup'
  | 'integrations'

/** Settings are returned as a flat object of key -> value (JSON). */
export type SettingsPayload = Record<string, unknown>

export async function getSettings(category: string): Promise<SettingsPayload> {
  const { data } = await api.get<SettingsPayload>(`${SETTINGS_BASE}/${category}`)
  return data ?? {}
}

export async function putSettings(
  category: string,
  payload: SettingsPayload
): Promise<void> {
  await api.put(`${SETTINGS_BASE}/${category}`, payload)
}

export function useSettings(category: SettingsCategory | null) {
  return useQuery({
    queryKey: settingsKeys.category(category ?? ''),
    queryFn: () => getSettings(category!),
    enabled: !!category,
    staleTime: 60_000,
  })
}

export function usePutSettings(category: SettingsCategory) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: SettingsPayload) => putSettings(category, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: settingsKeys.category(category) })
    },
  })
}
