'use client'

import { Users, MessageSquare, Clock, Activity } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Skeleton } from '@/components/ui/skeleton'

interface CollaborationItem {
  id: number
  user: string
  action: string
  target: string
  timestamp: string
  type: 'investigation' | 'alert' | 'comment' | 'update'
}

export function RealTimeCollaboration() {
  // Fetch real-time collaboration data from alerts/audit logs
  const { data, isLoading } = useQuery({
    queryKey: ['collaboration-activity'],
    queryFn: async () => {
      // Fetch recent alert updates as collaboration items
      const response = await api.get<{items: any[]}>('/api/alerts?limit=10&sortField=Timestamp&sortDirection=desc')
      
      // Transform alerts into collaboration items
      return response.data.items?.map((alert, index) => ({
        id: index + 1,
        user: alert.assignedTo || 'System',
        action: alert.status === 'new' ? 'created alert' : `updated to ${alert.status}`,
        target: alert.message?.substring(0, 50) || `Alert #${alert.id}`,
        timestamp: alert.timestamp || new Date().toISOString(),
        type: 'alert' as const
      })) || []
    },
    staleTime: 10000,
    refetchInterval: 30000,
  })

  const collaborationItems = data || []

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      </div>
    )
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'investigation': return Users
      case 'comment': return MessageSquare
      default: return Activity
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'investigation': return 'text-blue-500'
      case 'alert': return 'text-red-500'
      case 'comment': return 'text-green-500'
      default: return 'text-gray-500'
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Users className="h-6 w-6 text-blue-500" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Real-Time Collaboration
          </h3>
        </div>
        <div className="flex items-center space-x-2">
          <Activity className="h-4 w-4 text-green-500 animate-pulse" />
          <span className="text-sm text-gray-500">Live</span>
        </div>
      </div>

      {collaborationItems.length === 0 ? (
        <div className="text-center text-gray-500 py-8">
          <Users className="h-12 w-12 mx-auto mb-3 text-gray-400" />
          <p>No recent collaboration activity</p>
        </div>
      ) : (
        <div className="space-y-4">
          {collaborationItems.map((item) => {
            const TypeIcon = getTypeIcon(item.type)
            return (
              <div 
                key={item.id}
                className="flex items-start space-x-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
              >
                <div className={`p-2 rounded-full bg-white dark:bg-gray-600 ${getTypeColor(item.type)}`}>
                  <TypeIcon className="h-4 w-4" />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    <span className="font-medium">{item.user}</span>{' '}
                    <span className="text-gray-500 dark:text-gray-400">{item.action}</span>{' '}
                    <span className="font-medium">{item.target}</span>
                  </p>
                  <div className="flex items-center mt-1 text-xs text-gray-400">
                    <Clock className="h-3 w-3 mr-1" />
                    {new Date(item.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
