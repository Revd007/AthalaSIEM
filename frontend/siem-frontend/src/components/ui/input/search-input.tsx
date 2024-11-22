import React, { useState } from 'react'
import { Search, X } from 'lucide-react'
import { TextInput } from './text-input'

interface SearchInputProps {
  onSearch: (value: string) => void
  placeholder?: string
  className?: string
  debounceMs?: number
}

export function SearchInput({
  onSearch,
  placeholder = 'Search...',
  className,
  debounceMs = 300,
}: SearchInputProps) {
  const [value, setValue] = useState('')
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value
    setValue(newValue)
    
    // Debounce search
    const timeoutId = setTimeout(() => {
      onSearch(newValue)
    }, debounceMs)
    
    return () => clearTimeout(timeoutId)
  }
  
  const handleClear = () => {
    setValue('')
    onSearch('')
  }

  return (
    <TextInput
      value={value}
      onChange={handleChange}
      placeholder={placeholder}
      className={className}
      icon={Search}
      rightElement={
        value ? (
          <button
            onClick={handleClear}
            className="text-gray-400 hover:text-gray-500"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null
      }
    />
  )
}