'use client'

import { useState, useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Search, Play, Save, Filter, Clock, AlertTriangle, Activity, Database } from 'lucide-react'
import { Editor } from '@monaco-editor/react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { logService } from '@/services/log-service'
import { useAlerts } from '@/services/alert-service'
import { Skeleton } from '@/components/ui/skeleton'

interface QueryResult {
  timestamp: string
  source: string
  event_type: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  message: string
  details: Record<string, any>
}

export function LiveHunting() {
  const [query, setQuery] = useState('severity=high OR severity=critical')
  const [isRunning, setIsRunning] = useState(false)
  const [selectedResult, setSelectedResult] = useState<QueryResult | null>(null)
  const [timeRange, setTimeRange] = useState('15m')

  // Calculate time range
  const getTimeRange = () => {
    const end = new Date();
    const start = new Date();
    switch (timeRange) {
      case '15m': start.setMinutes(start.getMinutes() - 15); break;
      case '1h': start.setHours(start.getHours() - 1); break;
      case '4h': start.setHours(start.getHours() - 4); break;
      case '24h': start.setHours(start.getHours() - 24); break;
    }
    return { start, end };
  };

  const { start, end } = getTimeRange();

  // Fetch logs based on query
  const { data: logsData, isLoading: logsLoading, refetch: refetchLogs } = useQuery({
    queryKey: ['live-hunting', query, timeRange],
    queryFn: async () => {
      // Simple query parsing - in production, use proper query parser
      let severity: string | undefined;
      if (query.toLowerCase().includes('severity=high') || query.toLowerCase().includes('severity=critical')) {
        severity = query.toLowerCase().includes('critical') ? 'critical' : 'high';
      }
      
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        severity,
        limit: 100,
        sortField: 'timestamp',
        sortDirection: 'desc'
      });
    },
    enabled: !isRunning,
    refetchInterval: 30000,
  });

  // Fetch alerts for high severity events
  const { data: alertsData } = useAlerts({
    limit: 50,
    severity: 'high',
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  const handleRunQuery = async () => {
    setIsRunning(true)
    await refetchLogs()
    setIsRunning(false)
  }

  // Convert logs to query results
  const results = useMemo(() => {
    if (!logsData?.items) return [];
    
    return logsData.items.map((log): QueryResult => ({
      timestamp: log.timestamp,
      source: log.source || 'Unknown',
      event_type: log.category || log.level || 'log',
      severity: (log.severity?.toLowerCase() || 'low') as 'critical' | 'high' | 'medium' | 'low',
      message: log.message || 'No message',
      details: {
        agentId: log.agentId,
        processName: log.processName,
        ipAddress: log.ipAddress,
        username: log.username,
        eventId: log.eventId,
        ...log
      }
    }));
  }, [logsData]);

  // Generate timeline data from logs
  const timelineData = useMemo(() => {
    if (!logsData?.items) {
      return Array.from({ length: 20 }, (_, i) => ({
        time: new Date(Date.now() - i * 60000).toISOString(),
        events: 0,
        matches: 0
      })).reverse();
    }

    const minuteData: Record<string, { events: number; matches: number }> = {};
    
    logsData.items.forEach(log => {
      if (log.timestamp) {
        const date = new Date(log.timestamp);
        const minuteKey = date.toISOString().substring(0, 16) + ':00';
        if (!minuteData[minuteKey]) {
          minuteData[minuteKey] = { events: 0, matches: 0 };
        }
        minuteData[minuteKey].events++;
        if (log.severity === 'High' || log.severity === 'Critical') {
          minuteData[minuteKey].matches++;
        }
      }
    });

    return Object.entries(minuteData)
      .map(([time, data]) => ({ time, ...data }))
      .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime())
      .slice(-20);
  }, [logsData]);

  const totalEvents = logsData?.totalCount || 0;
  const matches = results.filter(r => r.severity === 'high' || r.severity === 'critical').length;

  return (
    <div className="space-y-6">
      {/* Query Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Events Processed"
          value={totalEvents > 1000 ? `${(totalEvents / 1000).toFixed(1)}K` : totalEvents.toString()}
          change="+0"
          trend="up"
          icon={Activity}
        />
        <StatsCard
          title="Query Time"
          value={logsLoading ? '...' : '<1s'}
          change="+0"
          trend="down"
          icon={Clock}
          color="green"
        />
        <StatsCard
          title="Matches Found"
          value={matches.toString()}
          change="+0"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Data Sources"
          value={(new Set(results.map(r => r.source)).size || 1).toString()}
          change="+0"
          trend="up"
          icon={Database}
          color="blue"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Query Editor */}
        <div className="lg:col-span-2">
          <DashboardCard title="Live Query" icon={Search}>
            <div className="space-y-4">
              {/* Query Controls */}
              <div className="flex justify-between">
                <div className="space-x-2">
                  <button 
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center"
                    onClick={handleRunQuery}
                  >
                    {isRunning ? (
                      <Activity className="h-4 w-4 mr-2 animate-pulse" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    {isRunning ? 'Running...' : 'Run Query'}
                  </button>
                  <button className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 flex items-center">
                    <Save className="h-4 w-4 mr-2" />
                    Save Query
                  </button>
                </div>
                <select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                  className="px-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
                >
                  <option value="15m">Last 15 minutes</option>
                  <option value="1h">Last 1 hour</option>
                  <option value="4h">Last 4 hours</option>
                  <option value="24h">Last 24 hours</option>
                </select>
              </div>

              {/* Query Editor */}
              <div className="h-[200px] border rounded-lg dark:border-gray-700 overflow-hidden">
                <Editor
                  defaultLanguage="sql"
                  theme="vs-dark"
                  value={query}
                  onChange={(value) => setQuery(value || '')}
                  options={{
                    minimap: { enabled: false },
                    fontSize: 14,
                    lineNumbers: 'on',
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                  }}
                />
              </div>

              {/* Results Timeline */}
              <div className="h-[200px]">
                {logsLoading ? (
                  <Skeleton className="h-full w-full" />
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={timelineData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time"
                      tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(label) => new Date(label).toLocaleString()}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="events" 
                      stroke="#3b82f6" 
                      name="Total Events"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="matches" 
                      stroke="#ef4444" 
                      name="Matches"
                    />
                  </LineChart>
                </ResponsiveContainer>
                )}
              </div>
            </div>
          </DashboardCard>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-1">
          <DashboardCard title="Results" icon={AlertTriangle}>
            <div className="space-y-4">
              {/* Filters */}
              <div className="flex space-x-2">
                <div className="relative flex-1">
                  <input
                    type="text"
                    placeholder="Filter results..."
                    className="w-full pl-10 pr-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
                  />
                  <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                </div>
                <button className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                  <Filter className="h-5 w-5 text-gray-500" />
                </button>
              </div>

              {/* Results List */}
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {logsLoading ? (
                  <div className="space-y-2">
                    {[1, 2, 3].map((i) => (
                      <Skeleton key={i} className="h-20 w-full" />
                    ))}
                  </div>
                ) : results.length === 0 ? (
                  <div className="text-center text-gray-500 py-4">
                    No results found
                  </div>
                ) : (
                  results.map((result, index) => (
                  <div
                    key={index}
                    onClick={() => setSelectedResult(result)}
                    className={`p-4 rounded-lg cursor-pointer border ${
                      selectedResult === result
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            result.severity === 'critical' 
                              ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                          }`}>
                            {result.severity}
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {result.event_type}
                          </span>
                        </div>
                        <p className="text-sm text-gray-900 dark:text-white mt-1">
                          {result.message}
                        </p>
                      </div>
                    </div>
                    <div className="mt-2 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                      <span>{result.source}</span>
                      <span>{new Date(result.timestamp).toLocaleString()}</span>
                    </div>
                  </div>
                  ))
                )}
              </div>

              {/* Selected Result Details */}
              {selectedResult && (
                <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                    Event Details
                  </h3>
                  <pre className="text-xs text-gray-600 dark:text-gray-300 overflow-auto">
                    {JSON.stringify(selectedResult.details, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </DashboardCard>
        </div>
      </div>
    </div>
  )
}

interface StatsCardProps {
  title: string
  value: string
  change: string
  trend: 'up' | 'down'
  icon: any
  color?: 'blue' | 'red' | 'green'
}

function StatsCard({ title, value, change, trend, icon: Icon, color = 'blue' }: StatsCardProps) {
  const colors = {
    blue: 'bg-blue-50 dark:bg-blue-900/20',
    red: 'bg-red-50 dark:bg-red-900/20',
    green: 'bg-green-50 dark:bg-green-900/20',
  }

  return (
    <div className={`${colors[color]} rounded-lg p-6`}>
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</span>
        <Icon className="h-5 w-5 text-gray-400" />
      </div>
      <div className="mt-4">
        <span className="text-2xl font-bold text-gray-900 dark:text-white">{value}</span>
        <span className={`
          ml-2 text-sm font-medium
          ${trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
        `}>
          {change}
        </span>
      </div>
    </div>
  )
} 