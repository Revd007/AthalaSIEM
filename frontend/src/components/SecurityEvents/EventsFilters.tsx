import { Filter, Calendar, X } from 'lucide-react'
import { useState } from 'react'

const timeRanges = ['1h', '6h', '12h', '24h', '7d', '30d']
const severityLevels = ['Critical', 'High', 'Medium', 'Low']
const eventTypes = ['Authentication', 'Network', 'System', 'Application']

interface FilterChangeEvent {
  target: {
    name: string
    value: string | number | boolean
  }
}

interface FilterProps {
  onFilterChange: (filters: Record<string, unknown>) => void
}

interface EventsFiltersProps {
  filters: any
  onFiltersChange: (filters: any) => void
  timeRange: string
  onTimeRangeChange: (range: string) => void
}

export function EventsFilters({ 
  filters, 
  onFiltersChange,
  timeRange,
  onTimeRangeChange 
}: EventsFiltersProps) {
  const [isOpen, setIsOpen] = useState(false)

  const handleFilterChange = (event: FilterChangeEvent) => {
    const { name, value } = event.target
    onFiltersChange({ [name]: value })
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-4 py-2 bg-white dark:bg-gray-800 rounded-lg shadow-sm"
      >
        <Filter className="h-4 w-4" />
        <span>Filters</span>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-medium">Filters</h3>
            <button onClick={() => setIsOpen(false)}>
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* Time Range */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Time Range</label>
            <div className="flex flex-wrap gap-2">
              {timeRanges.map(range => (
                <button
                  key={range}
                  className={`px-3 py-1 text-sm rounded-full ${
                    timeRange === range 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                  onClick={() => onTimeRangeChange(range)}
                >
                  {range}
                </button>
              ))}
            </div>
          </div>

          {/* Severity */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Severity</label>
            <div className="flex flex-wrap gap-2">
              {severityLevels.map(level => (
                <button
                  key={level}
                  className={`px-3 py-1 text-sm rounded-full ${
                    filters.severity.includes(level)
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                  onClick={() => {
                    const newSeverity = filters.severity.includes(level)
                      ? filters.severity.filter((s: string) => s !== level)
                      : [...filters.severity, level]
                    onFiltersChange({ ...filters, severity: newSeverity })
                  }}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>

          {/* Event Types */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Event Type</label>
            <div className="flex flex-wrap gap-2">
              {eventTypes.map(type => (
                <button
                  key={type}
                  className={`px-3 py-1 text-sm rounded-full ${
                    filters.eventType.includes(type)
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                  onClick={() => {
                    const newTypes = filters.eventType.includes(type)
                      ? filters.eventType.filter((t: string) => t !== type)
                      : [...filters.eventType, type]
                    onFiltersChange({ ...filters, eventType: newTypes })
                  }}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 