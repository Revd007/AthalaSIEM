'use client'

import { useState } from 'react'
import { Table } from '@/components/ui/table'

interface Log {
  id: string
  timestamp: string
  source: string
  severity: string
  message: string
}

interface LogTableProps {
  filters: {
    severity: string
    timeRange: string
    source: string
  }
}

export function LogTable({ filters }: LogTableProps) {
  const [logs] = useState<Log[]>([
    {
      id: '1',
      timestamp: '2024-03-20 10:30:00',
      source: 'Firewall',
      severity: 'High',
      message: 'Unauthorized access attempt detected',
    },
    // Add more sample logs
  ])

  return (
    <div className="overflow-x-auto">
      <Table>
        <thead>
          <tr className="bg-gray-50">
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Timestamp
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Source
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Severity
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Message
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {logs.map((log) => (
            <tr key={log.id} className="hover:bg-gray-50">
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                {log.timestamp}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {log.source}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                  log.severity === 'High' ? 'bg-red-100 text-red-800' :
                  log.severity === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {log.severity}
                </span>
              </td>
              <td className="px-6 py-4 text-sm text-gray-500">
                {log.message}
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  )
}