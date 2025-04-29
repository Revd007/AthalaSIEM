'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/components/ui/use-toast'
import { aiAnalysisService } from '@/services/ai-analysis-service'
import type { AnalysisResult } from '@/types/ai-analysis'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Search, Play, Save, Filter, Clock, AlertTriangle, Activity, Database } from 'lucide-react'
import { Editor } from '@monaco-editor/react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface QueryResult {
  id: string
  query: string
  results: HuntingResult[]
  timestamp: string
  status: 'running' | 'completed' | 'failed'
}

interface HuntingResult extends AnalysisResult {
  event_type: string
  severity: string
  timestamp: string
  source: string
  details: Record<string, unknown>
}

interface LiveHuntingProps {
  initialResults?: HuntingResult[]
  onResultClick: (result: HuntingResult) => void
}

const mockTimelineData = Array.from({ length: 20 }, (_, i) => ({
  time: new Date(Date.now() - i * 60000).toISOString(),
  events: Math.floor(Math.random() * 100),
  matches: Math.floor(Math.random() * 20)
})).reverse()

const mockResults: QueryResult[] = [
  {
    id: '1',
    query: 'source=* | search severity=high',
    results: [
      {
        id: '1',
        type: 'threat',
        event_type: 'process_creation',
        severity: 'high',
        timestamp: new Date().toISOString(),
        source: 'windows-dc1',
        description: 'Suspicious PowerShell execution detected',
        recommendations: ['Investigate PowerShell execution', 'Check for malicious commands'],
        confidence: 0.85,
        details: {
          process_name: 'powershell.exe',
          command_line: 'powershell.exe -enc YWxlcnQoImhlbGxvIik=',
          user: 'SYSTEM',
          pid: 4528
        }
      },
      {
        id: '2',
        type: 'threat',
        event_type: 'authentication',
        severity: 'critical',
        timestamp: new Date(Date.now() - 5000).toISOString(),
        source: 'linux-web1',
        description: 'Multiple failed login attempts detected',
        recommendations: ['Investigate failed login attempts', 'Check for brute force attacks'],
        confidence: 0.95,
        details: {
          user: 'admin',
          source_ip: '192.168.1.100',
          attempts: 5
        }
      }
    ],
    timestamp: new Date().toISOString(),
    status: 'completed'
  }
]

export function LiveHunting({ initialResults, onResultClick }: LiveHuntingProps) {
  const [query, setQuery] = useState('')
  const [queryResults, setQueryResults] = useState<QueryResult | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [selectedResult, setSelectedResult] = useState<HuntingResult | null>(null)
  const { toast } = useToast()

  const handleSearch = async () => {
    try {
      setIsSearching(true)
      const response = await aiAnalysisService.analyzeThreat({ query })
      setQueryResults({
        id: Math.random().toString(36).substr(2, 9),
        query,
        results: [response as HuntingResult],
        timestamp: new Date().toISOString(),
        status: 'completed'
      })
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to analyze threat',
        variant: 'destructive',
      })
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Query Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Events Processed"
          value="1.2M"
          change="+24.5k"
          trend="up"
          icon={Activity}
        />
        <StatsCard
          title="Query Time"
          value="1.8s"
          change="-0.3s"
          trend="down"
          icon={Clock}
          color="green"
        />
        <StatsCard
          title="Matches Found"
          value="247"
          change="+18"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Data Sources"
          value="15"
          change="+2"
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
                  <Button
                    onClick={handleSearch}
                    disabled={isSearching}
                    className="w-full"
                  >
                    {isSearching ? 'Searching...' : 'Search'}
                  </Button>
                  <button className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 flex items-center">
                    <Save className="h-4 w-4 mr-2" />
                    Save Query
                  </button>
                </div>
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
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mockTimelineData}>
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
              <div className="space-y-2">
                {queryResults?.results.map((result: HuntingResult) => (
                  <div
                    key={result.id}
                    onClick={() => onResultClick(result)}
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
                            result.severity === 'high' ? 'bg-red-100 text-red-800' :
                            result.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {result.severity}
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {result.event_type}
                          </span>
                        </div>
                        <p className="text-sm text-gray-900 dark:text-white mt-1">
                          {result.description}
                        </p>
                      </div>
                    </div>
                    <div className="mt-2 flex justify-between text-xs text-gray-500 dark:text-gray-400">
                      <span>{result.source}</span>
                      <span>{new Date(result.timestamp).toLocaleString()}</span>
                    </div>
                  </div>
                ))}
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