'use client'

import { Filter, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

export interface Filters {
  status: string[];
  priority: string[];
  category: string[];
  assignee: string[];
  [key: string]: string[];
}

interface IncidentFiltersProps {
  filters: Filters;
  onFiltersChange: (filters: Filters) => void;
}

export function IncidentFilters({ filters, onFiltersChange }: IncidentFiltersProps) {
  const statuses = ['investigating', 'containment', 'eradication', 'recovery', 'resolved']
  const priorities = ['critical', 'high', 'medium', 'low']
  const categories = ['security', 'network', 'system', 'application']

  return (
    <div className="flex items-center space-x-4">
      <div className="flex items-center space-x-2">
        <Filter className="h-5 w-5 text-gray-500" />
        <span className="text-sm font-medium">Filters:</span>
      </div>

      <div className="flex flex-wrap gap-2">
        {Object.entries(filters).map(([key, values]) => 
          values.map((value: string) => (
            <div 
              key={`${key}-${value}`}
              className="flex items-center px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
            >
              {value}
              <X 
                className="h-4 w-4 ml-2 cursor-pointer"
                onClick={() => {
                  const newValues = filters[key].filter((v: string) => v !== value)
                  onFiltersChange({ ...filters, [key]: newValues })
                }}
              />
            </div>
          ))
        )}
      </div>

      <Button variant="outline" size="sm">
        Add Filter
      </Button>
    </div>
  )
} 