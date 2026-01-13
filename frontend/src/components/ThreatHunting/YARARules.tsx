'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Code, Play, Plus, Upload, Download, RefreshCw, Search } from 'lucide-react'
import { Editor } from '@monaco-editor/react'
import { useYaraRules, useTestYaraRule } from '@/services/detection-rules-service'
import { Skeleton } from '@/components/ui/skeleton'

const defaultYaraRule = `rule suspicious_behavior
{
    meta:
        description = "Detects suspicious behavior"
        author = "Security Team"
        severity = "high"
        
    strings:
        $suspicious_cmd = "powershell.exe -enc"
        $sus_path = /\\temp\\.*\\.exe/
        
    condition:
        any of them
}`

export function YARARules() {
  const [selectedRuleId, setSelectedRuleId] = useState<string | null>(null)
  const [ruleContent, setRuleContent] = useState(defaultYaraRule)
  const [searchQuery, setSearchQuery] = useState('')

  const { data: rules, isLoading } = useYaraRules()
  const testRuleMutation = useTestYaraRule()

  const filteredRules = rules?.filter(rule => 
    rule.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    rule.description.toLowerCase().includes(searchQuery.toLowerCase())
  ) || []

  const handleTestRule = async () => {
    if (selectedRuleId) {
      await testRuleMutation.mutateAsync(selectedRuleId)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Rules List */}
      <div className="lg:col-span-1">
        <DashboardCard title="YARA Rules" icon={Code}>
          <div className="space-y-4">
            {/* Search and Actions */}
            <div className="flex space-x-2">
              <div className="relative flex-1">
                <input
                  type="text"
                  placeholder="Search rules..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
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
            <div className="space-y-2 max-h-[600px] overflow-y-auto">
              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-28 w-full" />
                  ))}
                </div>
              ) : filteredRules.length === 0 ? (
                <div className="text-center text-gray-500 py-4">
                  No rules found
                </div>
              ) : (
                filteredRules.map(rule => (
                <div
                  key={rule.id}
                  onClick={() => {
                    setSelectedRuleId(rule.id)
                    setRuleContent(rule.content || defaultYaraRule)
                  }}
                  className={`p-4 rounded-lg cursor-pointer border ${
                    selectedRuleId === rule.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">
                        {rule.name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        {rule.description}
                      </p>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      rule.severity === 'critical' 
                        ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                        : rule.severity === 'high'
                        ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200'
                        : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                    }`}>
                      {rule.severity}
                    </span>
                  </div>
                  <div className="mt-3 flex justify-between text-sm text-gray-500 dark:text-gray-400">
                    <span>{rule.category}</span>
                    <span>{rule.matches} matches</span>
                  </div>
                </div>
              )))}
            </div>
          </div>
        </DashboardCard>
      </div>

      {/* Rule Editor */}
      <div className="lg:col-span-2">
        <DashboardCard title="Rule Editor" icon={Code}>
          <div className="space-y-4">
            {/* Editor Actions */}
            <div className="flex justify-between">
              <div className="space-x-2">
                <button 
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center"
                  onClick={handleTestRule}
                  disabled={!selectedRuleId || testRuleMutation.isPending}
                >
                  {testRuleMutation.isPending ? (
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
              <button className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center">
                <Download className="h-4 w-4 mr-2" />
                Export
              </button>
            </div>

            {/* Test Results */}
            {testRuleMutation.data && (
              <div className={`p-4 rounded-lg ${
                testRuleMutation.data.success 
                  ? 'bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-200'
                  : 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-200'
              }`}>
                <p>Test {testRuleMutation.data.success ? 'passed' : 'failed'}</p>
                <p className="text-sm mt-1">
                  Found {testRuleMutation.data.matches} matches in {testRuleMutation.data.executionTime.toFixed(2)}s
                </p>
              </div>
            )}

            {/* Code Editor */}
            <div className="h-[600px] border rounded-lg dark:border-gray-700 overflow-hidden">
              <Editor
                defaultLanguage="javascript"
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
