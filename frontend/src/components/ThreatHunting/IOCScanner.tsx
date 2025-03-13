'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Search, Upload, AlertTriangle, Shield, Database, RefreshCw } from 'lucide-react'

interface IOC {
  type: 'ip' | 'domain' | 'hash' | 'url' | 'email'
  value: string
  description?: string
}

interface ScanResult {
  ioc: IOC
  matches: {
    source: string
    timestamp: string
    details: string
    severity: 'critical' | 'high' | 'medium' | 'low'
  }[]
}

export function IOCScanner() {
  const [scanning, setScanning] = useState(false)
  const [results, setResults] = useState<ScanResult[]>([])
  const [selectedIOCs, setSelectedIOCs] = useState<IOC[]>([])

  const handleScan = async () => {
    setScanning(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))
    setScanning(false)
    // Add mock results
    setResults([
      {
        ioc: { type: 'ip', value: '192.168.1.100' },
        matches: [
          {
            source: 'Network Logs',
            timestamp: new Date().toISOString(),
            details: 'Suspicious outbound connection detected',
            severity: 'high'
          }
        ]
      }
    ])
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* IOC Input Panel */}
      <div className="lg:col-span-1">
        <DashboardCard title="IOC Scanner" icon={Search}>
          <div className="space-y-4">
            {/* IOC Input */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Enter IOCs (one per line)
              </label>
              <textarea
                className="w-full h-48 p-3 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
                placeholder="Enter IPs, domains, hashes, URLs, or email addresses..."
              />
            </div>

            {/* Upload IOCs */}
            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-8 h-8 mb-3 text-gray-400" />
                  <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    CSV, JSON, or TXT files
                  </p>
                </div>
                <input type="file" className="hidden" />
              </label>
            </div>

            {/* Scan Options */}
            <div className="space-y-3">
              <label className="block text-sm font-medium mb-2">
                Scan Options
              </label>
              <div className="flex items-center">
                <input type="checkbox" className="rounded border-gray-300" />
                <span className="ml-2 text-sm">Include historical data</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" className="rounded border-gray-300" />
                <span className="ml-2 text-sm">Real-time monitoring</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" className="rounded border-gray-300" />
                <span className="ml-2 text-sm">Cross-reference with threat intel</span>
              </div>
            </div>

            {/* Scan Button */}
            <button
              onClick={handleScan}
              disabled={scanning}
              className="w-full py-2 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center justify-center"
            >
              {scanning ? (
                <RefreshCw className="w-5 h-5 animate-spin mr-2" />
              ) : (
                <Search className="w-5 h-5 mr-2" />
              )}
              {scanning ? 'Scanning...' : 'Start Scan'}
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
                <div className="text-2xl font-bold">{selectedIOCs.length}</div>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
                <div className="text-sm text-red-600 dark:text-red-400">Matches Found</div>
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {results.length}
                </div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <div className="text-sm text-green-600 dark:text-green-400">Scan Time</div>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">2.3s</div>
              </div>
            </div>

            {/* Results Table */}
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
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {result.ioc.value}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {result.ioc.type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {result.matches.length}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          result.matches[0].severity === 'critical' 
                            ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                            : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                        }`}>
                          {result.matches[0].severity}
                        </span>
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
          </div>
        </DashboardCard>
      </div>
    </div>
  )
} 