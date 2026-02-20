import { QueryClient } from '@tanstack/react-query'
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister'

const MAX_AGE = 1000 * 60 * 60 * 24 // 24 hours

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: MAX_AGE,
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
})

function getStorage(): Storage | undefined {
  if (typeof window === 'undefined') return undefined
  return window.localStorage
}

export const persister = createSyncStoragePersister({
  storage: {
    getItem: (key) => getStorage()?.getItem(key) ?? null,
    setItem: (key, value) => getStorage()?.setItem(key, value),
    removeItem: (key) => getStorage()?.removeItem(key),
  },
  key: 'athalasiem-query-cache',
})

export const persistOptions = {
  persister,
  maxAge: MAX_AGE,
}
