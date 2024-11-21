'use client'

import { useState } from 'react'
import { Table } from '../../ui/table'
import { Button } from '../../ui/button'
import { ArrowDownTrayIcon, EyeIcon, TrashIcon } from '@heroicons/react/24/outline'
import React from 'react'

interface Report {
  id: string
  name: string
  type: string
  status: string
  createdAt: string
  size: string
}

interface ReportTableProps {
  filters: {
    type: string
    status: string
    dateRange: string
  }
}

export function ReportTable({ filters }: ReportTableProps) {
  const [reports] = useState<Report[]>([
    {
      id: '1',
      name: 'Monthly Security Analysis',
      type: 'Security',
      status: 'Completed',
      createdAt: '2024-03-01',
      size: '2.5 MB',
    },
    // Add more sample reports
  ])

  return (
    <div className="overflow-x-auto">
      <Table>
        <thead>
          <tr className="bg-gray-50">
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Report Name
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Type
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Status
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Created At
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Size
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {reports.map((report) => (
            <tr key={report.id} className="hover:bg-gray-50">
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                {report.name}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {report.type}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                  report.status === 'Completed' ? 'bg-green-100 text-green-800' :
                  report.status === 'Processing' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {report.status}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {report.createdAt}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {report.size}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <div className="flex space-x-2">
                  <Button variant="outline" className="h-8 px-2 text-sm">
                    <EyeIcon className="h-4 w-4 mr-1" />
                    View
                  </Button>
                  <Button variant="outline" className="h-8 px-2 text-sm">
                    <ArrowDownTrayIcon className="h-4 w-4 mr-1" />
                    Download
                  </Button>
                  <Button variant="outline" className="h-8 px-2 text-sm text-red-600 hover:text-red-700 hover:bg-red-50">
                    <TrashIcon className="h-4 w-4" />
                  </Button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  )
}