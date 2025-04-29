'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Book, Play, Plus, Copy, Download, RefreshCw, Search, Clock, Users } from 'lucide-react'
import { Editor } from '@monaco-editor/react'

interface HuntingPlaybook {
  id: string
  name: string
  description: string
  tactics: string[]
  status: 'active' | 'inactive'
  lastRun: string
  results: {
    findings: number
    alerts: number
    falsePositives: number
  }
}

interface HuntingPlaybooksProps {
  playbooks: HuntingPlaybook[]
  onPlaybookClick: (playbook: HuntingPlaybook) => void
}

const mockPlaybooks: HuntingPlaybook[] = [
  {
    id: '1',
    name: 'Ransomware Hunt',
    description: 'Detect potential ransomware activity',
    tactics: ['Malware'],
    status: 'active',
    lastRun: new Date().toISOString(),
    results: {
      findings: 0,
      alerts: 0,
      falsePositives: 0
    }
  }
]

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

export function HuntingPlaybooks({ playbooks, onPlaybookClick }: HuntingPlaybooksProps) {
  const [selectedPlaybook, setSelectedPlaybook] = useState<HuntingPlaybook | null>(null)
  const [playbookContent, setPlaybookContent] = useState(defaultPlaybook)
  const [isRunning, setIsRunning] = useState(false)

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
                  className="w-full pl-10 pr-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
                />
                <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
              </div>
              <button className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
                <Plus className="h-5 w-5" />
              </button>
            </div>

            {/* Playbooks List */}
            <div className="space-y-2">
              {playbooks.map(playbook => (
                <div
                  key={playbook.id}
                  onClick={() => setSelectedPlaybook(playbook)}
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
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                    }`}>
                      {playbook.status}
                    </span>
                  </div>
                  <div className="mt-3 flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center">
                      <Users className="h-4 w-4 mr-1" />
                      {/* Assuming playbook.author is available */}
                    </div>
                    <div className="flex items-center">
                      <Clock className="h-4 w-4 mr-1" />
                      {/* Assuming playbook.lastModified is available */}
                    </div>
                  </div>
                </div>
              ))}
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
                  onClick={() => setIsRunning(true)}
                >
                  {isRunning ? (
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