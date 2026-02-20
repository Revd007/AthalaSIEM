'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Search, Upload, AlertTriangle, RefreshCw } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import { aiApi } from '@/lib/ai-api'
import { Skeleton } from '@/components/ui/skeleton'

interface IOC {
  type: 'ip' | 'domain' | 'hash' | 'url' | 'email'
  value: string
}

interface ScanResult {
  ioc: IOC
  matchesFound: number
  results: Array<{ type: string; value: string; sourceFeed: string; confidence: number }>
}

function detectIocType(value: string): IOC['type'] {
  if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(value)) return 'ip'
  if (/^[a-f0-9]{32,64}$/i.test(value)) return 'hash'
  if (/@/.test(value)) return 'email'
  if (/^https?:\/\//.test(value)) return 'url'
  return 'domain'
}

export function IOCScanner() {
  const [iocInput, setIocInput] = useState('')
  const [includeHistorical, setIncludeHistorical] = useState(false)
  const [realTimeMonitoring, setRealTimeMonitoring] = useState(false)
  const [crossReference, setCrossReference] = useState(true)

  const scanMutation = useMutation({
    mutationFn: async (iocs: string[]): Promise<ScanResult[]> => {
      const results: ScanResult[] = []
      for (const raw of iocs) {
        const trimmed = raw.trim()
        if (!trimmed) continue
        try {
          const res = await aiApi.scanIoc({ value: trimmed })
          results.push({
            ioc: { type: detectIocType(trimmed), value: trimmed },
            matchesFound: res.matchesFound ?? 0,
            results: res.results ?? [],
          })
        } catch {
          results.push({
            ioc: { type: detectIocType(trimmed), value: trimmed },
            matchesFound: 0,
            results: [],
          })
        }
      }
      return results
    },
  })

  const handleScan = () => {
    const iocs = iocInput.split('\n').filter(Boolean)
    if (iocs.length > 0) {
      scanMutation.mutate(iocs)
    }
  }

  const results = scanMutation.data || []
  const totalMatches = results.reduce((sum, r) => sum + r.matchesFound, 0)

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* IOC Input Panel */}
      <div className="lg:col-span-1">
        <DashboardCard title="IOC Scanner" icon={Search}>
          <div className="space-y-4">
            {/* IOC Input */}
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                Enter IOCs (one per line)
              </label>
              <textarea
                className="w-full h-48 p-3 border rounded-lg dark:bg-gray-800 dark:border-gray-700 text-sm"
                placeholder="Enter IPs, domains, hashes, URLs, or email addresses..."
                value={iocInput}
                onChange={(e) => setIocInput(e.target.value)}
              />
            </div>

            {/* Upload IOCs */}
            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 border-gray-300 dark:border-gray-600">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-8 h-8 mb-3 text-gray-400" />
                  <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    CSV, JSON, or TXT files
                  </p>
                </div>
                <input type="file" className="hidden" accept=".csv,.json,.txt" />
              </label>
            </div>

            {/* Scan Options */}
            <div className="space-y-3">
              <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                Scan Options
              </label>
              <div className="flex items-center">
                <input 
                  type="checkbox" 
                  className="rounded border-gray-300 dark:border-gray-600"
                  checked={includeHistorical}
                  onChange={(e) => setIncludeHistorical(e.target.checked)}
                />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Include historical data</span>
              </div>
              <div className="flex items-center">
                <input 
                  type="checkbox" 
                  className="rounded border-gray-300 dark:border-gray-600"
                  checked={realTimeMonitoring}
                  onChange={(e) => setRealTimeMonitoring(e.target.checked)}
                />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Real-time monitoring</span>
              </div>
              <div className="flex items-center">
                <input 
                  type="checkbox" 
                  className="rounded border-gray-300 dark:border-gray-600"
                  checked={crossReference}
                  onChange={(e) => setCrossReference(e.target.checked)}
                />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Cross-reference with threat intel</span>
              </div>
            </div>

            {/* Scan Button */}
            <button
              onClick={handleScan}
              disabled={scanMutation.isPending || !iocInput.trim()}
              className="w-full py-2 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {scanMutation.isPending ? (
                <RefreshCw className="w-5 h-5 animate-spin mr-2" />
              ) : (
                <Search className="w-5 h-5 mr-2" />
              )}
              {scanMutation.isPending ? 'Scanning...' : 'Start Scan'}
            </button>
          </div>
        </DashboardCard>
      </div>

      {/* Results Panel */}
      <div className="lg:col-span-2">
        <DashboardCard title="Scan Results" icon={AlertTriangle}>
          <div className="space-y-4">
            {/* Results Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Total IOCs</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {iocInput.split('\n').filter(Boolean).length}
                </div>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
                <div className="text-sm text-red-600 dark:text-red-400">Matches Found</div>
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {totalMatches}
                </div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <div className="text-sm text-green-600 dark:text-green-400">Scan Status</div>
                <div className="text-lg font-bold text-green-600 dark:text-green-400">
                  {scanMutation.isPending ? 'Running' : scanMutation.isSuccess ? 'Complete' : 'Ready'}
                </div>
              </div>
            </div>

            {/* Results Table */}
            {scanMutation.isPending ? (
              <div className="space-y-2">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : results.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <Search className="h-12 w-12 mx-auto mb-3 text-gray-400" />
                <p>Enter IOCs and click scan to search</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead>
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        IOC
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Matches
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Severity
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {results.map((result, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                          {result.ioc.value}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {result.ioc.type}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                          {result.matchesFound}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {result.matchesFound > 0 ? (
                            <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200">
                              match
                            </span>
                          ) : (
                            <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200">
                              clean
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <button className="text-blue-500 hover:text-blue-600">
                            View Details
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </DashboardCard>
      </div>
    </div>
  )
}
