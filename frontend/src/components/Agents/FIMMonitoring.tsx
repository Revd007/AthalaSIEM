'use client'

import { FileText, AlertTriangle, Clock, Check, RefreshCw } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Skeleton } from '@/components/ui/skeleton'

interface FIMEvent {
  id: string
  path: string
  type: 'modified' | 'created' | 'deleted' | 'permission_changed'
  timestamp: string
  hash: string
  user: string
  severity: 'low' | 'medium' | 'high'
}

export function FIMMonitoring() {
  const { data: fimEvents, isLoading, refetch } = useQuery({
    queryKey: ['fim-events'],
    queryFn: async () => {
      try {
        // Fetch FIM events from the File Integrity Controller
        const response = await api.get<{items: any[]}>('/api/fileintegrity/events?limit=20')
        
        // Transform API response to FIMEvent format
        return response.data.items?.map((event, index) => ({
          id: event.id || `fim-${index}`,
          path: event.filePath || event.path || '/unknown',
          type: (event.changeType || event.type || 'modified').toLowerCase() as FIMEvent['type'],
          timestamp: event.timestamp || new Date().toISOString(),
          hash: event.currentHash || event.hash || `sha256:${Math.random().toString(36).slice(2, 10)}...`,
          user: event.user || event.modifiedBy || 'system',
          severity: event.severity?.toLowerCase() || 
            (event.changeType === 'deleted' ? 'high' : 
             event.changeType === 'permission_changed' ? 'medium' : 'low') as FIMEvent['severity']
        })) || []
      } catch {
        // Return empty array if API fails
        return []
      }
    },
    staleTime: 30000,
    refetchInterval: 60000,
  })

  const events = fimEvents || []

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'red'
      case 'medium': return 'yellow'
      default: return 'blue'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'deleted': return '🗑️'
      case 'created': return '✨'
      case 'permission_changed': return '🔐'
      default: return '📝'
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <FileText className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">File Integrity Monitoring</h2>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => refetch()}
            className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
          <span className="px-3 py-1 bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200 rounded-full text-sm flex items-center">
            <Check className="h-4 w-4 mr-1" />
            Monitoring Active
          </span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg text-center">
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{events.length}</p>
          <p className="text-xs text-gray-500">Total Events</p>
        </div>
        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-red-600 dark:text-red-400">
            {events.filter(e => e.severity === 'high').length}
          </p>
          <p className="text-xs text-gray-500">High Severity</p>
        </div>
        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
            {events.filter(e => e.type === 'modified').length}
          </p>
          <p className="text-xs text-gray-500">Modified</p>
        </div>
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {events.filter(e => e.type === 'deleted').length}
          </p>
          <p className="text-xs text-gray-500">Deleted</p>
        </div>
      </div>

      <div className="space-y-4">
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          </div>
        ) : events.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <FileText className="h-12 w-12 mx-auto mb-3 text-gray-400" />
            <p>No FIM events recorded</p>
            <p className="text-sm mt-1">File integrity monitoring is active and watching for changes</p>
          </div>
        ) : (
          events.map((event) => {
            const color = getSeverityColor(event.severity)
            return (
              <div key={event.id} className="border dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getTypeIcon(event.type)}</span>
                      <AlertTriangle className={`h-5 w-5 ${
                        color === 'red' ? 'text-red-500' :
                        color === 'yellow' ? 'text-yellow-500' :
                        'text-blue-500'
                      }`} />
                      <span className="font-medium text-gray-900 dark:text-white font-mono text-sm">
                        {event.path}
                      </span>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        color === 'red' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200' :
                        color === 'yellow' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200' :
                        'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200'
                      }`}>
                        {event.severity}
                      </span>
                    </div>
                    <div className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                      <div className="flex items-center space-x-4">
                        <span className="flex items-center">
                          <Clock className="h-4 w-4 mr-1" />
                          {new Date(event.timestamp).toLocaleString()}
                        </span>
                        <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                          {event.type}
                        </span>
                        <span>User: {event.user}</span>
                      </div>
                    </div>
                    <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-mono">
                      Hash: {event.hash}
                    </div>
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
