'use client'

import { SystemHealthOverview } from '@/components/SystemHealth/SystemHealthOverview'
import { DevicesList } from '@/components/SystemHealth/DevicesList'
import { DeviceMetrics } from '@/components/SystemHealth/DeviceMetrics'
import { DeviceDetails } from '@/components/SystemHealth/DeviceDetails'
import { SystemHealthFilters } from '@/components/SystemHealth/SystemHealthFilters'
import type { DeviceType } from '@/types/system-health'
import { useState } from 'react'

export default function SystemHealthPage() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [selectedTypes, setSelectedTypes] = useState<DeviceType[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string[]>([])

  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <DevicesList
            selectedDevice={selectedDevice}
            onDeviceSelect={setSelectedDevice}
            typeFilter={selectedTypes}
            searchQuery={searchQuery}
            statusFilter={statusFilter}
          />
        </div>
        
        <div className="lg:col-span-2">
          {selectedDevice ? (
            <div className="space-y-6">
              <DeviceMetrics deviceId={selectedDevice} />
              <DeviceDetails deviceId={selectedDevice} />
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              Select a device to view details
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 