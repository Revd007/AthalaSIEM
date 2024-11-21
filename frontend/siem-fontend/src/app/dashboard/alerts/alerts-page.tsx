'use client'

import { useState } from 'react'
import { Card } from '../../../components/ui/card'
import { Button } from '../../../components/ui/button'
import { AlertTable } from '../../../components/shared/tables/alert-table'
import { AlertFilter } from '../../../components/shared/filters/alert-filter'
import React from 'react'

export default function Alerts() {
  const [filters, setFilters] = useState({
    status: 'all',
    priority: 'all',
    timeRange: '24h',
  })

  return (
    <div className="py-6">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-semibold text-gray-900">Alert Management</h1>
          <Button variant="primary">
            Create Alert Rule
          </Button>
        </div>

        <div className="mt-6">
          <Card className="p-6">
            <AlertFilter filters={filters} setFilters={setFilters} />
            <div className="mt-6">
              <AlertTable filters={filters} />
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}