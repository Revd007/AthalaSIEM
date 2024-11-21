'use client'

import React, { useState } from 'react'
import { LogTable } from '../../../components/shared/tables/log-table'
import { LogFilter } from '../../../components/shared/filters/log-filter'
import { Card } from '../../../components/ui/card'

export default function Logs() {
  const [filters, setFilters] = useState({
    severity: 'all',
    timeRange: '24h',
    source: 'all',
  })

  return (
    <div className="py-6">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-semibold text-gray-900">Log Management</h1>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
            Export Logs
          </button>
        </div>

        <div className="mt-6">
          <Card className="p-6">
            <LogFilter filters={filters} setFilters={setFilters} />
            <div className="mt-6">
              <LogTable filters={filters} />
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}