'use client'

import { ClipboardList, CheckCircle, XCircle, Clock, Plus } from 'lucide-react'
import { useChangeRequests, useChangeManagementStats, useUpdateChangeStatus } from '@/services/change-management-service'
import { Skeleton } from '@/components/ui/skeleton'

const statusIcons = {
  pending: Clock,
  approved: CheckCircle,
  rejected: XCircle,
  implemented: CheckCircle,
}

const statusColors = {
  pending: 'yellow',
  approved: 'green',
  rejected: 'red',
  implemented: 'blue',
}

export function ChangeManagement() {
  const { data: changes, isLoading } = useChangeRequests()
  const { data: stats } = useChangeManagementStats()
  const updateStatusMutation = useUpdateChangeStatus()

  const handleStatusUpdate = async (id: string, newStatus: string) => {
    await updateStatusMutation.mutateAsync({ id, status: newStatus })
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <ClipboardList className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Change Management</h2>
        </div>
        <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center">
          <Plus className="h-4 w-4 mr-2" />
          New Change Request
        </button>
      </div>

      {/* Stats Summary */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <p className="text-sm text-gray-500 dark:text-gray-400">Total</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.totalRequests}</p>
          </div>
          <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
            <p className="text-sm text-yellow-600 dark:text-yellow-400">Pending</p>
            <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{stats.pendingRequests}</p>
          </div>
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <p className="text-sm text-green-600 dark:text-green-400">Approved</p>
            <p className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.approvedRequests}</p>
          </div>
          <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <p className="text-sm text-red-600 dark:text-red-400">High Risk</p>
            <p className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.highRiskRequests}</p>
          </div>
        </div>
      )}

      <div className="space-y-6">
        {isLoading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-40 w-full" />
            ))}
          </div>
        ) : !changes || changes.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            No change requests found
          </div>
        ) : (
          changes.map((change) => {
            const StatusIcon = statusIcons[change.status as keyof typeof statusIcons] || Clock
            const color = statusColors[change.status as keyof typeof statusColors] || 'gray'

            return (
              <div key={change.id} className="border dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-900 dark:text-white">{change.id}</span>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        color === 'yellow' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200' :
                        color === 'green' ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200' :
                        color === 'red' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200' :
                        'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200'
                      }`}>
                        {change.status}
                      </span>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        change.type === 'emergency' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200' : 
                        'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200'
                      }`}>
                        {change.type}
                      </span>
                    </div>
                    <h3 className="mt-1 font-medium text-gray-900 dark:text-white">{change.title}</h3>
                    <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{change.description}</p>
                  </div>
                  <StatusIcon className={`h-5 w-5 ${
                    color === 'yellow' ? 'text-yellow-500' :
                    color === 'green' ? 'text-green-500' :
                    color === 'red' ? 'text-red-500' :
                    'text-blue-500'
                  }`} />
                </div>

                <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Requester</p>
                    <p className="font-medium text-gray-900 dark:text-white">{change.requester}</p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Implementation Date</p>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {change.implementation ? new Date(change.implementation).toLocaleDateString() : 'TBD'}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Risk Level</p>
                    <p className={`font-medium ${
                      change.risk === 'high' ? 'text-red-600 dark:text-red-400' : 
                      change.risk === 'medium' ? 'text-yellow-600 dark:text-yellow-400' : 
                      'text-green-600 dark:text-green-400'
                    }`}>
                      {change.risk.toUpperCase()}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Approvers</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {change.approvers.map((approver, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full"
                        >
                          {approver}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                {change.status === 'pending' && (
                  <div className="mt-4 flex space-x-2">
                    <button
                      onClick={() => handleStatusUpdate(change.id, 'approved')}
                      className="px-3 py-1 bg-green-500 text-white text-sm rounded hover:bg-green-600"
                      disabled={updateStatusMutation.isPending}
                    >
                      Approve
                    </button>
                    <button
                      onClick={() => handleStatusUpdate(change.id, 'rejected')}
                      className="px-3 py-1 bg-red-500 text-white text-sm rounded hover:bg-red-600"
                      disabled={updateStatusMutation.isPending}
                    >
                      Reject
                    </button>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
