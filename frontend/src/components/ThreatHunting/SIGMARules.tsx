'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { FileCode, Play, Plus, Upload, Download, RefreshCw, Search, Settings } from 'lucide-react'
import { Editor } from '@monaco-editor/react'

interface SigmaRule {
  id: string
  title: string
  description: string
  status: 'active' | 'disabled' | 'testing'
  level: 'critical' | 'high' | 'medium' | 'low'
  logsource: string
  tags: string[]
  lastModified: string
  matches: number
}

const mockRules: SigmaRule[] = [
  {
    id: '1',
    title: 'Suspicious Service Creation',
    description: 'Detects suspicious service creation from unusual processes',
    status: 'active',
    level: 'high',
    logsource: 'windows/security',
    tags: ['attack.persistence', 'attack.t1543.003'],
    lastModified: new Date().toISOString(),
    matches: 3
  },
  {
    id: '2',
    title: 'PowerShell Download Cradle',
    description: 'Detects PowerShell download cradles',
    status: 'active',
    level: 'medium',
    logsource: 'windows/powershell',
    tags: ['attack.execution', 'attack.t1059.001'],
    lastModified: new Date().toISOString(),
    matches: 7
  }
]

const defaultSigmaRule = `title: Suspicious Service Creation
id: 5268a407-391d-4ff6-8f78-87e5c7c6a8e5
status: experimental
description: Detects suspicious service creation
references:
    - https://attack.mitre.org/techniques/T1543/003/
author: Security Team
date: 2024/01/01
modified: 2024/01/01
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        CommandLine|contains: 
            - 'New-Service'
            - 'sc.exe create'
    condition: selection
falsepositives:
    - Legitimate service installations
level: high
tags:
    - attack.persistence
    - attack.t1543.003`

export function SIGMARules() {
  const [selectedRule, setSelectedRule] = useState<SigmaRule | null>(null)
  const [ruleContent, setRuleContent] = useState(defaultSigmaRule)
  const [isTestingRule, setIsTestingRule] = useState(false)
  const [selectedLogSource, setSelectedLogSource] = useState('windows/security')

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Rules List */}
      <div className="lg:col-span-1 space-y-6">
        <DashboardCard title="SIGMA Rules" icon={FileCode}>
          {/* Search and Actions */}
          <div className="space-y-4">
            <div className="flex space-x-2">
              <div className="relative flex-1">
                <input
                  type="text"
                  placeholder="Search rules..."
                  className="w-full pl-10 pr-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
                />
                <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
              </div>
              <button className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
                <Plus className="h-5 w-5" />
              </button>
              <button className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                <Upload className="h-5 w-5 text-gray-500 dark:text-gray-400" />
              </button>
            </div>

            {/* Rules List */}
            <div className="space-y-2">
              {mockRules.map(rule => (
                <div
                  key={rule.id}
                  onClick={() => setSelectedRule(rule)}
                  className={`p-4 rounded-lg cursor-pointer border ${
                    selectedRule?.id === rule.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">
                        {rule.title}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        {rule.description}
                      </p>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      rule.level === 'critical' 
                        ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                        : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                    }`}>
                      {rule.level}
                    </span>
                  </div>
                  <div className="mt-3">
                    <div className="flex flex-wrap gap-2">
                      {rule.tags.map((tag, index) => (
                        <span 
                          key={index}
                          className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="mt-3 flex justify-between text-sm text-gray-500 dark:text-gray-400">
                    <span>{rule.logsource}</span>
                    <span>{new Date(rule.lastModified).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </DashboardCard>

        {/* Log Sources */}
        <DashboardCard title="Log Sources" icon={Settings}>
          <div className="space-y-2">
            <select
              value={selectedLogSource}
              onChange={(e) => setSelectedLogSource(e.target.value)}
              className="w-full p-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
            >
              <option value="windows/security">Windows Security</option>
              <option value="windows/sysmon">Windows Sysmon</option>
              <option value="linux/auditd">Linux Auditd</option>
              <option value="apache/access">Apache Access</option>
            </select>
          </div>
        </DashboardCard>
      </div>

      {/* Rule Editor */}
      <div className="lg:col-span-2">
        <DashboardCard title="Rule Editor" icon={FileCode}>
          <div className="space-y-4">
            {/* Editor Actions */}
            <div className="flex justify-between">
              <div className="space-x-2">
                <button 
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center"
                  onClick={() => setIsTestingRule(true)}
                >
                  {isTestingRule ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Test Rule
                </button>
                <button className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600">
                  Save
                </button>
              </div>
              <div className="space-x-2">
                <button className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                  Convert
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
                defaultLanguage="yaml"
                theme="vs-dark"
                value={ruleContent}
                onChange={(value) => setRuleContent(value || '')}
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