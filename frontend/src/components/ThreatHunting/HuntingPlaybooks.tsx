'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Book, Play, Plus, Copy, Download, RefreshCw, Search, Clock, Users } from 'lucide-react'
import { Editor } from '@monaco-editor/react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Skeleton } from '@/components/ui/skeleton'

interface Playbook {
  id: string
  name: string
  description: string
  author: string
  category: string
  status: 'draft' | 'active' | 'archived'
  lastRun: string | null
  lastModified: string
  steps: PlaybookStep[]
}

interface PlaybookStep {
  id: string
  type: 'query' | 'analysis' | 'enrichment' | 'action'
  name: string
  description: string
  config: any
}

const defaultPlaybook = `{
  "name": "New Threat Hunt",
  "description": "Description of the hunt",
  "steps": [
    {
      "type": "query",
      "name": "Initial Search",
      "description": "Search for indicators",
      "config": {
        "query": "source=* | search suspicious_activity"
      }
    }
  ]
}`

// API functions
const playbookApi = {
  async getPlaybooks(): Promise<Playbook[]> {
    const response = await api.get<Playbook[]>('/api/playbooks')
    return response.data
  },

  async runPlaybook(id: string): Promise<any> {
    const response = await api.post(`/api/playbooks/${id}/run`)
    return response.data
  }
}

export function HuntingPlaybooks() {
  const [selectedPlaybook, setSelectedPlaybook] = useState<Playbook | null>(null)
  const [playbookContent, setPlaybookContent] = useState(defaultPlaybook)
  const [searchQuery, setSearchQuery] = useState('')
  const queryClient = useQueryClient()

  const { data: playbooks, isLoading } = useQuery({
    queryKey: ['playbooks'],
    queryFn: () => playbookApi.getPlaybooks(),
    staleTime: 30000,
  })

  const runPlaybookMutation = useMutation({
    mutationFn: (id: string) => playbookApi.runPlaybook(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['playbooks'] })
    }
  })

  const filteredPlaybooks = playbooks?.filter(playbook =>
    playbook.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    playbook.description.toLowerCase().includes(searchQuery.toLowerCase())
  ) || []

  const handleRunPlaybook = async () => {
    if (selectedPlaybook) {
      await runPlaybookMutation.mutateAsync(selectedPlaybook.id)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Playbooks List */}
      <div className="lg:col-span-1 space-y-6">
        <DashboardCard title="Hunting Playbooks" icon={Book}>
          <div className="space-y-4">
            {/* Search and Actions */}
            <div className="flex space-x-2">
              <div className="relative flex-1">
                <input
                  type="text"
                  placeholder="Search playbooks..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
                />
                <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
              </div>
              <button className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
                <Plus className="h-5 w-5" />
              </button>
            </div>

            {/* Playbooks List */}
            <div className="space-y-2 max-h-[500px] overflow-y-auto">
              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-32 w-full" />
                  ))}
                </div>
              ) : filteredPlaybooks.length === 0 ? (
                <div className="text-center text-gray-500 py-4">
                  No playbooks found
                </div>
              ) : (
                filteredPlaybooks.map(playbook => (
                <div
                  key={playbook.id}
                  onClick={() => {
                    setSelectedPlaybook(playbook)
                    setPlaybookContent(JSON.stringify({
                      name: playbook.name,
                      description: playbook.description,
                      steps: playbook.steps
                    }, null, 2))
                  }}
                  className={`p-4 rounded-lg cursor-pointer border ${
                    selectedPlaybook?.id === playbook.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">
                        {playbook.name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        {playbook.description}
                      </p>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      playbook.status === 'active'
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200'
                        : playbook.status === 'draft'
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                    }`}>
                      {playbook.status}
                    </span>
                  </div>
                  <div className="mt-3 flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center">
                      <Users className="h-4 w-4 mr-1" />
                      {playbook.author}
                    </div>
                    <div className="flex items-center">
                      <Clock className="h-4 w-4 mr-1" />
                      {new Date(playbook.lastModified).toLocaleDateString()}
                    </div>
                  </div>
                  {playbook.lastRun && (
                    <div className="mt-2 text-xs text-gray-400">
                      Last run: {new Date(playbook.lastRun).toLocaleString()}
                    </div>
                  )}
                </div>
              )))}
            </div>
          </div>
        </DashboardCard>
      </div>

      {/* Playbook Editor */}
      <div className="lg:col-span-2">
        <DashboardCard title="Playbook Editor" icon={Book}>
          <div className="space-y-4">
            {/* Editor Actions */}
            <div className="flex justify-between">
              <div className="space-x-2">
                <button 
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center"
                  onClick={handleRunPlaybook}
                  disabled={!selectedPlaybook || runPlaybookMutation.isPending}
                >
                  {runPlaybookMutation.isPending ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Run Playbook
                </button>
                <button className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600">
                  Save
                </button>
              </div>
              <div className="space-x-2">
                <button className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center">
                  <Copy className="h-4 w-4 mr-2" />
                  Clone
                </button>
                <button className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </button>
              </div>
            </div>

            {/* Run Status */}
            {runPlaybookMutation.isSuccess && (
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-green-800 dark:text-green-200">
                <p>Playbook execution started successfully</p>
                <p className="text-sm mt-1">
                  Run ID: {runPlaybookMutation.data?.id || 'N/A'}
                </p>
              </div>
            )}

            {/* Code Editor */}
            <div className="h-[600px] border rounded-lg dark:border-gray-700 overflow-hidden">
              <Editor
                defaultLanguage="json"
                theme="vs-dark"
                value={playbookContent}
                onChange={(value) => setPlaybookContent(value || '')}
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                }}
              />
            </div>
          </div>
        </DashboardCard>
      </div>
    </div>
  )
}
