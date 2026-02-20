import { api } from '@/lib/api'
import type { PlaybookDefinition, PlaybookExecution } from '@/types/api'

/** Playbook list item from Python backend */
export interface PlaybookListItem {
  id: string
  name: string
  description: string
  author: string
  category: string
  status: string
  steps: unknown[]
  lastModified: string | null
}

export const playbookService = {
  async list(): Promise<PlaybookListItem[]> {
    try {
      const { data } = await api.get<PlaybookListItem[]>('/api/playbooks')
      return Array.isArray(data) ? data : []
    } catch {
      return []
    }
  },

  async get(id: string): Promise<PlaybookListItem | null> {
    try {
      const { data } = await api.get<PlaybookListItem>(`/api/playbooks/${id}`)
      return data ?? null
    } catch {
      return null
    }
  },

  async create(body: { name: string; description?: string; category?: string; steps?: unknown[] }): Promise<PlaybookDefinition | null> {
    try {
      const { data } = await api.post<PlaybookDefinition>('/api/playbooks', body)
      return data ?? null
    } catch {
      return null
    }
  },

  async update(id: string, body: { name?: string; description?: string; category?: string; steps?: unknown[] }): Promise<PlaybookDefinition | null> {
    try {
      const { data } = await api.put<PlaybookDefinition>(`/api/playbooks/${id}`, body)
      return data ?? null
    } catch {
      return null
    }
  },

  async delete(id: string): Promise<void> {
    await api.delete(`/api/playbooks/${id}`)
  },

  async run(id: string): Promise<PlaybookExecution | null> {
    try {
      const { data } = await api.post<PlaybookExecution>(`/api/playbooks/${id}/run`)
      return data ?? null
    } catch {
      return null
    }
  },
}
