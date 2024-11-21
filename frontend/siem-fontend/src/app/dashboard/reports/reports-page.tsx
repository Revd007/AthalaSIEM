'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ReportTable } from '../../components/shared/tables/report-table'
import { ReportFilter } from '../../components/shared/filters/report-filter'
import { CreateReportModal } from '../../components/shared/modals/create-report-modal'
import React from 'react'

export default function Reports() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [filters, setFilters] = useState({
    type: 'all',
    status: 'all',
    dateRange: '30d',
  })

  return (
    <div className="py-6">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-semibold text-gray-900">Report Management</h1>
          <Button variant="primary" onClick={() => setIsModalOpen(true)}>
            Generate New Report
          </Button>
        </div>

        <div className="mt-6">
          <Card className="p-6">
            <ReportFilter filters={filters} setFilters={setFilters} />
            <div className="mt-6">
              <ReportTable filters={filters} />
            </div>
          </Card>
        </div>

        <CreateReportModal 
          isOpen={isModalOpen} 
          onClose={() => setIsModalOpen(false)} 
        />
      </div>
    </div>
  )
}