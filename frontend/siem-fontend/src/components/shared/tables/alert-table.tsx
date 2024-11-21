'use client'

import { useState } from 'react'
import { Table } from '../../ui/table'
import { Button } from '../../ui/button'
import React from 'react'

interface Alert {
  id: string
  name: string
  priority: 'High' | 'Medium' | 'Low'
  status: 'Active' | 'Resolved' | 'Investigating'
  timestamp: string
  description: string
}

interface AlertTableProps {
  filters: {
    status: string;
    priority: string;
    timeRange: string;
  };
}

export const AlertTable: React.FC<AlertTableProps> = ({ filters }) => {
  return (
    <div className="overflow-x-auto">
      <Table>
        <thead>
          <tr className="bg-gray-50">
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Alert Name
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Priority
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Status
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Timestamp
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {alerts.map((alert) => (
            <tr key={alert.id} className="hover:bg-gray-50">
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                {alert.name}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                  alert.priority === 'High' ? 'bg-red-100 text-red-800' :
                  alert.priority === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {alert.priority}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                  alert.status === 'Active' ? 'bg-red-100 text-red-800' :
                  alert.status === 'Investigating' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {alert.status}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {alert.timestamp}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <Button variant="outline" className="mr-2">
                  View Details
                </Button>
                <Button variant="primary">
                  Resolve
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  )
}