'use client'

import { SystemHealthOverview } from '@/components/SystemHealth/SystemHealthOverview'
import { DevicesList } from '@/components/SystemHealth/DevicesList'
import { DeviceMetrics } from '@/components/SystemHealth/DeviceMetrics'
import { DeviceDetails } from '@/components/SystemHealth/DeviceDetails'
import { SystemHealthFilters } from '@/components/SystemHealth/SystemHealthFilters'
import type { DeviceType } from '@/types/system-health'
import { useState } from 'react'

export default function HealthPage() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [selectedTypes, setSelectedTypes] = useState<DeviceType[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string[]>([])

  return (
    <div className="p-4 sm:p-6 lg:p-8 space-y-4 sm:space-y-6 min-h-screen">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
        <h1 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
          System Health Monitoring
        </h1>
      </div>

      <SystemHealthOverview />

      <SystemHealthFilters
        selectedTypes={selectedTypes}
        onTypeChange={setSelectedTypes}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        statusFilter={statusFilter}
        onStatusChange={setStatusFilter}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6">
        <div className="lg:col-span-1 order-2 lg:order-1">
          <DevicesList
            selectedDevice={selectedDevice}
            onDeviceSelect={setSelectedDevice}
            typeFilter={selectedTypes}
            searchQuery={searchQuery}
            statusFilter={statusFilter}
          />
        </div>
        
        <div className="lg:col-span-2 order-1 lg:order-2">
          {selectedDevice ? (
            <div className="space-y-4 lg:space-y-6">
              <DeviceMetrics deviceId={selectedDevice} />
              <DeviceDetails deviceId={selectedDevice} />
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 lg:h-full text-gray-500 bg-gray-50 dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
              <p className="text-sm sm:text-base">Select a device to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 