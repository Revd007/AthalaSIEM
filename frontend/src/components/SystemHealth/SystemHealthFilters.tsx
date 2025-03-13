'use client'

import { Search } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import type { DeviceType } from '@/types/system-health'

interface SystemHealthFiltersProps {
  selectedTypes: DeviceType[]
  onTypeChange: (types: DeviceType[]) => void
  searchQuery: string
  onSearchChange: (query: string) => void
  statusFilter: string[]
  onStatusChange: (statuses: string[]) => void
}

const deviceTypes: { value: DeviceType; label: string }[] = [
  { value: 'server', label: 'Servers' },
  { value: 'firewall', label: 'Firewalls' },
  { value: 'network', label: 'Network' },
  { value: 'endpoint', label: 'Endpoints' },
  { value: 'storage', label: 'Storage' },
  { value: 'security-appliance', label: 'Security' }
]

const statusOptions = [
  { value: 'healthy', label: 'Healthy', color: 'text-green-500' },
  { value: 'warning', label: 'Warning', color: 'text-yellow-500' },
  { value: 'critical', label: 'Critical', color: 'text-red-500' },
  { value: 'offline', label: 'Offline', color: 'text-gray-500' }
]

export function SystemHealthFilters({
  selectedTypes,
  onTypeChange,
  searchQuery,
  onSearchChange,
  statusFilter,
  onStatusChange
}: SystemHealthFiltersProps) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4">
        <div className="flex-1 min-w-[300px]">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <Input
              placeholder="Search devices..."
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {deviceTypes.map(({ value, label }) => (
            <Button
              key={value}
              variant={selectedTypes.includes(value) ? 'default' : 'outline'}
              onClick={() => {
                if (selectedTypes.includes(value)) {
                  onTypeChange(selectedTypes.filter(t => t !== value))
                } else {
                  onTypeChange([...selectedTypes, value])
                }
              }}
              size="sm"
              className="min-w-[100px]"
            >
              {label}
            </Button>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        {statusOptions.map(({ value, label, color }) => (
          <Button
            key={value}
            variant={statusFilter.includes(value) ? 'default' : 'outline'}
            onClick={() => {
              if (statusFilter.includes(value)) {
                onStatusChange(statusFilter.filter(s => s !== value))
              } else {
                onStatusChange([...statusFilter, value])
              }
            }}
            size="sm"
            className={`min-w-[100px] ${statusFilter.includes(value) ? '' : color}`}
          >
            {label}
          </Button>
        ))}
      </div>
    </div>
  )
} 