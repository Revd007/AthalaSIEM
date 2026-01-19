'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { type NormalizationFilters } from '@/hooks/useNormalization'
import { Filter, X } from 'lucide-react'

interface NormalizationFiltersProps {
  filters: NormalizationFilters
  dateRange: { start?: string; end?: string }
  onFiltersChange: (filters: NormalizationFilters) => void
  onDateRangeChange: (range: { start?: string; end?: string }) => void
}

const EVENT_TYPES = [
  'authentication',
  'process',
  'network',
  'file',
  'security',
  'general',
  'unknown',
]

export function NormalizationFilters({
  filters,
  dateRange,
  onFiltersChange,
  onDateRangeChange,
}: NormalizationFiltersProps) {
  const [localFilters, setLocalFilters] = useState<NormalizationFilters>(filters)
  const [localDateRange, setLocalDateRange] = useState(dateRange)

  const handleApply = () => {
    onFiltersChange(localFilters)
    onDateRangeChange(localDateRange)
  }

  const handleReset = () => {
    const resetFilters: NormalizationFilters = {
      page: 1,
      pageSize: 50,
    }
    const resetDateRange = {
      start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
      end: new Date().toISOString(),
    }
    setLocalFilters(resetFilters)
    setLocalDateRange(resetDateRange)
    onFiltersChange(resetFilters)
    onDateRangeChange(resetDateRange)
  }

  const hasActiveFilters = Boolean(
    localFilters.eventType ||
    localFilters.sourceIp ||
    localFilters.minSeverity ||
    localDateRange.start ||
    localDateRange.end
  )

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label htmlFor="eventType">Event Type</Label>
          <Select
            value={localFilters.eventType || 'all'}
            onValueChange={(value) =>
              setLocalFilters((prev) => ({ ...prev, eventType: value === 'all' ? undefined : value }))
            }
          >
            <SelectTrigger>
              <SelectValue placeholder="All event types" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All event types</SelectItem>
              {EVENT_TYPES.map((type) => (
                <SelectItem key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="sourceIp">Source IP</Label>
          <Input
            id="sourceIp"
            placeholder="e.g., 192.168.1.1"
            value={localFilters.sourceIp || ''}
            onChange={(e) =>
              setLocalFilters((prev) => ({ ...prev, sourceIp: e.target.value || undefined }))
            }
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="minSeverity">Min Severity</Label>
          <Select
            value={localFilters.minSeverity?.toString() || 'any'}
            onValueChange={(value) =>
              setLocalFilters((prev) => ({
                ...prev,
                minSeverity: value === 'any' ? undefined : parseInt(value),
              }))
            }
          >
            <SelectTrigger>
              <SelectValue placeholder="Any severity" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="any">Any severity</SelectItem>
              <SelectItem value="1">Debug (1)</SelectItem>
              <SelectItem value="2">Info (2)</SelectItem>
              <SelectItem value="4">Warning (4)</SelectItem>
              <SelectItem value="7">Error (7)</SelectItem>
              <SelectItem value="10">Critical (10)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="startDate">Start Date</Label>
          <Input
            id="startDate"
            type="datetime-local"
            value={localDateRange.start ? new Date(localDateRange.start).toISOString().slice(0, 16) : ''}
            onChange={(e) =>
              setLocalDateRange((prev) => ({
                ...prev,
                start: e.target.value ? new Date(e.target.value).toISOString() : undefined,
              }))
            }
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="endDate">End Date</Label>
          <Input
            id="endDate"
            type="datetime-local"
            value={localDateRange.end ? new Date(localDateRange.end).toISOString().slice(0, 16) : ''}
            onChange={(e) =>
              setLocalDateRange((prev) => ({
                ...prev,
                end: e.target.value ? new Date(e.target.value).toISOString() : undefined,
              }))
            }
          />
        </div>
      </div>

      <div className="flex gap-2">
        <Button onClick={handleApply}>
          <Filter className="w-4 h-4 mr-2" />
          Apply Filters
        </Button>
        {hasActiveFilters && (
          <Button variant="outline" onClick={handleReset}>
            <X className="w-4 h-4 mr-2" />
            Reset
          </Button>
        )}
      </div>
    </div>
  )
}
