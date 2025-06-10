'use client'

import React, { useState, useEffect } from 'react'
import { 
  FileText, 
  Shield, 
  AlertTriangle, 
  Settings, 
  Search, 
  Filter, 
  Download, 
  Eye, 
  CheckCircle, 
  XCircle,
  Plus,
  Edit,
  Trash2,
  Clock,
  Server,
  Hash,
  FileCheck,
  AlertCircle
} from 'lucide-react'
import { toast } from 'sonner'

interface FileIntegrityEvent {
  id: string
  agentId: string
  agentName: string
  filePath: string
  changeType: 'Created' | 'Modified' | 'Deleted' | 'Renamed'
  baselineHash?: string
  currentHash?: string
  baselineSize?: number
  currentSize?: number
  baselineModified?: string
  currentModified?: string
  fileAttributes?: string
  severity: 'Low' | 'Medium' | 'High' | 'Critical'
  detectedAt: string
  processedAt: string
  isAcknowledged: boolean
  acknowledgedBy?: string
  acknowledgedAt?: string
  details?: string
}

interface FileIntegrityRule {
  id: string
  name: string
  description?: string
  isEnabled: boolean
  monitoredPaths: string
  excludePatterns?: string
  realTimeMonitoring: boolean
  scanIntervalMinutes: number
  severity: string
  alertOnCreation: boolean
  alertOnModification: boolean
  alertOnDeletion: boolean
  alertOnRename: boolean
  createdAt: string
  updatedAt: string
  createdBy?: string
  targetAgents?: string
}

interface FimStatistics {
  totalEvents: number
  eventsBySeverity: Array<{ severity: string; count: number }>
  eventsByChangeType: Array<{ changeType: string; count: number }>
  eventsByAgent: Array<{ agentId: string; agentName: string; count: number }>
  acknowledgedEvents: number
  unacknowledgedEvents: number
  eventsOverTime: Array<{ date: string; count: number }>
}

const FileIntegrityMonitoring: React.FC = () => {
  const [activeTab, setActiveTab] = useState('events')
  const [events, setEvents] = useState<FileIntegrityEvent[]>([])
  const [rules, setRules] = useState<FileIntegrityRule[]>([])
  const [statistics, setStatistics] = useState<FimStatistics | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [selectedEvents, setSelectedEvents] = useState<string[]>([])
  
  // Filters
  const [filters, setFilters] = useState({
    agentId: '',
    severity: '',
    changeType: '',
    acknowledged: '',
    startDate: '',
    endDate: '',
    search: ''
  })

  // Pagination
  const [pagination, setPagination] = useState({
    page: 1,
    pageSize: 50,
    totalCount: 0,
    totalPages: 0
  })

  // Modal states
  const [showEventDetails, setShowEventDetails] = useState<FileIntegrityEvent | null>(null)
  const [showRuleModal, setShowRuleModal] = useState<FileIntegrityRule | null>(null)
  const [showCreateRule, setShowCreateRule] = useState(false)

  useEffect(() => {
    loadData()
  }, [activeTab, filters, pagination.page])

  const loadData = async () => {
    setIsLoading(true)
    try {
      if (activeTab === 'events') {
        await loadEvents()
      } else if (activeTab === 'rules') {
        await loadRules()
      } else if (activeTab === 'statistics') {
        await loadStatistics()
      }
    } catch (error) {
      console.error('Error loading data:', error)
      toast.error('Failed to load data')
    } finally {
      setIsLoading(false)
    }
  }

  const loadEvents = async () => {
    try {
      const queryParams = new URLSearchParams({
        page: pagination.page.toString(),
        pageSize: pagination.pageSize.toString(),
        ...Object.fromEntries(Object.entries(filters).filter(([_, value]) => value))
      })

      const response = await fetch(`/api/fileintegrity/events?${queryParams}`)
      if (!response.ok) throw new Error('Failed to load events')

      const data = await response.json()
      setEvents(data.items || [])
      setPagination(prev => ({
        ...prev,
        totalCount: data.totalCount || 0,
        totalPages: data.totalPages || 0
      }))
    } catch (error) {
      console.error('Error loading events:', error)
      toast.error('Failed to load FIM events')
    }
  }

  const loadRules = async () => {
    try {
      const response = await fetch('/api/fileintegrity/rules')
      if (!response.ok) throw new Error('Failed to load rules')

      const data = await response.json()
      setRules(data)
    } catch (error) {
      console.error('Error loading rules:', error)
      toast.error('Failed to load FIM rules')
    }
  }

  const loadStatistics = async () => {
    try {
      const response = await fetch('/api/fileintegrity/statistics?days=7')
      if (!response.ok) throw new Error('Failed to load statistics')

      const data = await response.json()
      setStatistics(data)
    } catch (error) {
      console.error('Error loading statistics:', error)
      toast.error('Failed to load FIM statistics')
    }
  }

  const acknowledgeEvents = async (eventIds: string[]) => {
    try {
      const response = await fetch('/api/fileintegrity/events/acknowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ eventIds })
      })

      if (!response.ok) throw new Error('Failed to acknowledge events')

      await loadEvents()
      setSelectedEvents([])
      toast.success(`Acknowledged ${eventIds.length} event(s)`)
    } catch (error) {
      console.error('Error acknowledging events:', error)
      toast.error('Failed to acknowledge events')
    }
  }

  const toggleEventSelection = (eventId: string) => {
    setSelectedEvents(prev =>
      prev.includes(eventId)
        ? prev.filter(id => id !== eventId)
        : [...prev, eventId]
    )
  }

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'text-red-600 bg-red-100'
      case 'high': return 'text-orange-600 bg-orange-100'
      case 'medium': return 'text-yellow-600 bg-yellow-100'
      case 'low': return 'text-green-600 bg-green-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getChangeTypeIcon = (changeType: string) => {
    switch (changeType.toLowerCase()) {
      case 'created': return <Plus className="h-4 w-4 text-green-600" />
      case 'modified': return <Edit className="h-4 w-4 text-blue-600" />
      case 'deleted': return <Trash2 className="h-4 w-4 text-red-600" />
      case 'renamed': return <FileText className="h-4 w-4 text-purple-600" />
      default: return <FileCheck className="h-4 w-4 text-gray-600" />
    }
  }

  const tabs = [
    { id: 'events', label: 'Events', icon: AlertTriangle },
    { id: 'rules', label: 'Rules', icon: Settings },
    { id: 'statistics', label: 'Statistics', icon: Shield }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Shield className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">File Integrity Monitoring</h1>
              <p className="text-gray-600">Monitor and detect file system changes</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {activeTab === 'events' && selectedEvents.length > 0 && (
              <button
                onClick={() => acknowledgeEvents(selectedEvents)}
                className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
              >
                <CheckCircle className="h-4 w-4 mr-2" />
                Acknowledge ({selectedEvents.length})
              </button>
            )}
            
            {activeTab === 'rules' && (
              <button
                onClick={() => setShowCreateRule(true)}
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                <Plus className="h-4 w-4 mr-2" />
                Create Rule
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {/* Events Tab */}
          {activeTab === 'events' && (
            <div className="space-y-6">
              {/* Filters */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Severity</label>
                    <select
                      value={filters.severity}
                      onChange={(e) => setFilters({ ...filters, severity: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    >
                      <option value="">All Severities</option>
                      <option value="Critical">Critical</option>
                      <option value="High">High</option>
                      <option value="Medium">Medium</option>
                      <option value="Low">Low</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Change Type</label>
                    <select
                      value={filters.changeType}
                      onChange={(e) => setFilters({ ...filters, changeType: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    >
                      <option value="">All Types</option>
                      <option value="Created">Created</option>
                      <option value="Modified">Modified</option>
                      <option value="Deleted">Deleted</option>
                      <option value="Renamed">Renamed</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                    <select
                      value={filters.acknowledged}
                      onChange={(e) => setFilters({ ...filters, acknowledged: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    >
                      <option value="">All Status</option>
                      <option value="false">Unacknowledged</option>
                      <option value="true">Acknowledged</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <input
                        type="text"
                        placeholder="Search file path..."
                        value={filters.search}
                        onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                        className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md text-sm"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Events Table */}
              <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          <input
                            type="checkbox"
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedEvents(events.map(event => event.id))
                              } else {
                                setSelectedEvents([])
                              }
                            }}
                            className="rounded border-gray-300"
                          />
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          File Path
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Change Type
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Severity
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Agent
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Detected
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {events.map((event) => (
                        <tr key={event.id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <input
                              type="checkbox"
                              checked={selectedEvents.includes(event.id)}
                              onChange={() => toggleEventSelection(event.id)}
                              className="rounded border-gray-300"
                            />
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <FileText className="h-4 w-4 text-gray-400 mr-2" />
                              <div className="text-sm text-gray-900 truncate max-w-xs" title={event.filePath}>
                                {event.filePath}
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              {getChangeTypeIcon(event.changeType)}
                              <span className="ml-2 text-sm text-gray-900">{event.changeType}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(event.severity)}`}>
                              {event.severity}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <Server className="h-4 w-4 text-gray-400 mr-2" />
                              <span className="text-sm text-gray-900">{event.agentName}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {new Date(event.detectedAt).toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            {event.isAcknowledged ? (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                <CheckCircle className="h-3 w-3 mr-1" />
                                Acknowledged
                              </span>
                            ) : (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                                <AlertCircle className="h-3 w-3 mr-1" />
                                Pending
                              </span>
                            )}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                            <button
                              onClick={() => setShowEventDetails(event)}
                              className="text-blue-600 hover:text-blue-900 mr-4"
                            >
                              <Eye className="h-4 w-4" />
                            </button>
                            {!event.isAcknowledged && (
                              <button
                                onClick={() => acknowledgeEvents([event.id])}
                                className="text-green-600 hover:text-green-900"
                              >
                                <CheckCircle className="h-4 w-4" />
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
                  <div className="flex-1 flex justify-between sm:hidden">
                    <button
                      onClick={() => setPagination(prev => ({ ...prev, page: Math.max(1, prev.page - 1) }))}
                      disabled={pagination.page === 1}
                      className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
                    >
                      Previous
                    </button>
                    <button
                      onClick={() => setPagination(prev => ({ ...prev, page: Math.min(prev.totalPages, prev.page + 1) }))}
                      disabled={pagination.page === pagination.totalPages}
                      className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
                    >
                      Next
                    </button>
                  </div>
                  <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm text-gray-700">
                        Showing <span className="font-medium">{((pagination.page - 1) * pagination.pageSize) + 1}</span> to{' '}
                        <span className="font-medium">{Math.min(pagination.page * pagination.pageSize, pagination.totalCount)}</span> of{' '}
                        <span className="font-medium">{pagination.totalCount}</span> results
                      </p>
                    </div>
                    <div>
                      <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
                        <button
                          onClick={() => setPagination(prev => ({ ...prev, page: Math.max(1, prev.page - 1) }))}
                          disabled={pagination.page === 1}
                          className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                        >
                          Previous
                        </button>
                        <span className="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium text-gray-700">
                          Page {pagination.page} of {pagination.totalPages}
                        </span>
                        <button
                          onClick={() => setPagination(prev => ({ ...prev, page: Math.min(prev.totalPages, prev.page + 1) }))}
                          disabled={pagination.page === pagination.totalPages}
                          className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                        >
                          Next
                        </button>
                      </nav>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Rules Tab */}
          {activeTab === 'rules' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {rules.map((rule) => (
                  <div key={rule.id} className="bg-white border border-gray-200 rounded-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-medium text-gray-900">{rule.name}</h3>
                      <div className="flex items-center space-x-2">
                        {rule.isEnabled ? (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            Enabled
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                            Disabled
                          </span>
                        )}
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-600 mb-4">{rule.description}</p>
                    
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500">Monitored Paths:</span>
                        <span className="text-gray-900 truncate max-w-32" title={rule.monitoredPaths}>
                          {rule.monitoredPaths.split(',').length} path(s)
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500">Scan Interval:</span>
                        <span className="text-gray-900">{rule.scanIntervalMinutes}m</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500">Real-time:</span>
                        <span className="text-gray-900">{rule.realTimeMonitoring ? 'Yes' : 'No'}</span>
                      </div>
                    </div>
                    
                    <div className="mt-4 flex justify-end space-x-2">
                      <button
                        onClick={() => setShowRuleModal(rule)}
                        className="text-blue-600 hover:text-blue-900"
                      >
                        <Edit className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Statistics Tab */}
          {activeTab === 'statistics' && statistics && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="bg-blue-50 rounded-lg p-6">
                  <div className="flex items-center">
                    <FileCheck className="h-8 w-8 text-blue-600" />
                    <div className="ml-4">
                      <p className="text-2xl font-bold text-blue-900">{statistics.totalEvents}</p>
                      <p className="text-blue-600">Total Events</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-green-50 rounded-lg p-6">
                  <div className="flex items-center">
                    <CheckCircle className="h-8 w-8 text-green-600" />
                    <div className="ml-4">
                      <p className="text-2xl font-bold text-green-900">{statistics.acknowledgedEvents}</p>
                      <p className="text-green-600">Acknowledged</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-red-50 rounded-lg p-6">
                  <div className="flex items-center">
                    <AlertCircle className="h-8 w-8 text-red-600" />
                    <div className="ml-4">
                      <p className="text-2xl font-bold text-red-900">{statistics.unacknowledgedEvents}</p>
                      <p className="text-red-600">Pending</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-purple-50 rounded-lg p-6">
                  <div className="flex items-center">
                    <Server className="h-8 w-8 text-purple-600" />
                    <div className="ml-4">
                      <p className="text-2xl font-bold text-purple-900">{statistics.eventsByAgent.length}</p>
                      <p className="text-purple-600">Active Agents</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Charts placeholder - you can integrate charting library here */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Events by Severity</h3>
                  <div className="space-y-3">
                    {statistics.eventsBySeverity.map((item) => (
                      <div key={item.severity} className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">{item.severity}</span>
                        <span className="text-sm font-medium text-gray-900">{item.count}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="bg-white border border-gray-200 rounded-lg p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Events by Change Type</h3>
                  <div className="space-y-3">
                    {statistics.eventsByChangeType.map((item) => (
                      <div key={item.changeType} className="flex items-center justify-between">
                        <div className="flex items-center">
                          {getChangeTypeIcon(item.changeType)}
                          <span className="text-sm text-gray-600 ml-2">{item.changeType}</span>
                        </div>
                        <span className="text-sm font-medium text-gray-900">{item.count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Event Details Modal */}
      {showEventDetails && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-96 overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">Event Details</h3>
                <button
                  onClick={() => setShowEventDetails(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <XCircle className="h-6 w-6" />
                </button>
              </div>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">File Path</label>
                    <p className="text-sm text-gray-900 break-all">{showEventDetails.filePath}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Change Type</label>
                    <p className="text-sm text-gray-900">{showEventDetails.changeType}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Severity</label>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(showEventDetails.severity)}`}>
                      {showEventDetails.severity}
                    </span>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Agent</label>
                    <p className="text-sm text-gray-900">{showEventDetails.agentName}</p>
                  </div>
                </div>
                
                {showEventDetails.baselineHash && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Baseline Hash</label>
                      <p className="text-xs text-gray-900 font-mono break-all">{showEventDetails.baselineHash}</p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Current Hash</label>
                      <p className="text-xs text-gray-900 font-mono break-all">{showEventDetails.currentHash}</p>
                    </div>
                  </div>
                )}
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Detected At</label>
                    <p className="text-sm text-gray-900">{new Date(showEventDetails.detectedAt).toLocaleString()}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Status</label>
                    <p className="text-sm text-gray-900">
                      {showEventDetails.isAcknowledged ? 'Acknowledged' : 'Pending'}
                    </p>
                  </div>
                </div>
                
                {showEventDetails.details && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Additional Details</label>
                    <pre className="text-xs text-gray-900 bg-gray-50 p-3 rounded-md overflow-x-auto">
                      {JSON.stringify(JSON.parse(showEventDetails.details), null, 2)}
                    </pre>
                  </div>
                )}
              </div>
              
              <div className="mt-6 flex justify-end space-x-3">
                {!showEventDetails.isAcknowledged && (
                  <button
                    onClick={() => {
                      acknowledgeEvents([showEventDetails.id])
                      setShowEventDetails(null)
                    }}
                    className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                  >
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Acknowledge
                  </button>
                )}
                <button
                  onClick={() => setShowEventDetails(null)}
                  className="inline-flex items-center px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default FileIntegrityMonitoring 