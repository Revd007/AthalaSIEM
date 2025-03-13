import { useQuery } from '@tanstack/react-query'
import type { SecurityEvent } from '../types/dashboard'

export function useEvents() {
  return useQuery<SecurityEvent[]>({
    queryKey: ['events'],
    queryFn: async () => {
      const response = await fetch('/api/events')
      if (!response.ok) {
        throw new Error('Failed to fetch events')
      }
      return response.json()
    },
    refetchInterval: 30000 // Refresh every 30 seconds
  })
}

export function useEventDetails(eventId: string) {
  return useQuery({
    queryKey: ['events', eventId],
    queryFn: async () => {
      const response = await fetch(`/api/events/${eventId}`)
      if (!response.ok) {
        throw new Error('Failed to fetch event details')
      }
      return response.json()
    }
  })
}
