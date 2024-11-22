import React from 'react'
import { format } from 'date-fns'
import { Calendar as CalendarIcon } from 'lucide-react'
import { Popover, PopoverTrigger, PopoverContent } from '../popover'
import { Button } from '../button'
import { Calendar } from '../calendar'
import { cn } from '../../../lib/utils'

interface DatePickerProps {
  date?: Date
  onSelect: (date: Date | undefined) => void
  label?: string
  error?: string
}

export function DatePicker({ date, onSelect, label, error }: DatePickerProps) {
  return (
    <div className="space-y-1">
      {label && (
        <label className="text-sm font-medium text-gray-700">{label}</label>
      )}
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            className={cn(
              'w-full justify-start text-left font-normal',
              !date && 'text-gray-500',
              error && 'border-red-300 focus:border-red-500 focus:ring-red-500'
            )}
          >
            <CalendarIcon className="mr-2 h-4 w-4" />
            {date ? format(date, 'PPP') : 'Pick a date'}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0">
          <Calendar
            mode="single"
            selected={date}
            onSelect={onSelect}
            initialFocus
          />
        </PopoverContent>
      </Popover>
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  )
}