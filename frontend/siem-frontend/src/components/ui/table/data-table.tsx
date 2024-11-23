import React, { useState } from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './table'
import { Button } from '../button'
import { Input } from '../input'
import { 
  ChevronLeft, 
  ChevronRight, 
  ChevronsLeft, 
  ChevronsRight,
  ArrowUpDown,
  ArrowUp,
  ArrowDown
} from 'lucide-react'

interface DataTableProps<T> {
  data: T[]
  columns: {
    key: string
    title: string
    sortable?: boolean
    render?: (item: T) => React.ReactNode
  }[]
  pagination?: {
    pageSize: number
    totalItems: number
    currentPage: number
    onPageChange: (page: number) => void
  }
  onSort?: (key: string, direction: 'asc' | 'desc') => void
  loading?: boolean
}

export function DataTable<T>({
  data,
  columns,
  pagination,
  onSort,
  loading
}: DataTableProps<T>) {
  const [sortConfig, setSortConfig] = useState<{
    key: string
    direction: 'asc' | 'desc'
  } | null>(null)

  const handleSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'asc'
    if (sortConfig?.key === key && sortConfig.direction === 'asc') {
      direction = 'desc'
    }
    setSortConfig({ key, direction })
    onSort?.(key, direction)
  }

  const getSortIcon = (key: string) => {
    if (sortConfig?.key !== key) return <ArrowUpDown className="h-4 w-4" />
    return sortConfig.direction === 'asc' ? (
      <ArrowUp className="h-4 w-4" />
    ) : (
      <ArrowDown className="h-4 w-4" />
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              {columns.map((column) => (
                <TableHead key={column.key}>
                  {column.sortable ? (
                    <button
                      className="flex items-center space-x-1"
                      onClick={() => handleSort(column.key)}
                    >
                      <span>{column.title}</span>
                      {getSortIcon(column.key)}
                    </button>
                  ) : (
                    column.title
                  )}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  Loading...
                </TableCell>
              </TableRow>
            ) : data.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  No data available
                </TableCell>
              </TableRow>
            ) : (
              data.map((item, index) => (
                <TableRow key={index}>
                  {columns.map((column) => (
                    <TableCell key={column.key}>
                      {column.render
                        ? column.render(item)
                        : (item as any)[column.key]}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {pagination && (
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <p className="text-sm text-gray-700">
              Showing{' '}
              <span className="font-medium">
                {(pagination.currentPage - 1) * pagination.pageSize + 1}
              </span>{' '}
              to{' '}
              <span className="font-medium">
                {Math.min(
                  pagination.currentPage * pagination.pageSize,
                  pagination.totalItems
                )}
              </span>{' '}
              of{' '}
              <span className="font-medium">{pagination.totalItems}</span>{' '}
              results
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              onClick={() => pagination.onPageChange(1)}
              disabled={pagination.currentPage === 1}
            >
              <ChevronsLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              onClick={() => pagination.onPageChange(pagination.currentPage - 1)}
              disabled={pagination.currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Input
              className="w-16 text-center"
              value={pagination.currentPage}
              onChange={(e) => {
                const page = parseInt(e.target.value)
                if (page > 0 && page <= Math.ceil(pagination.totalItems / pagination.pageSize)) {
                  pagination.onPageChange(page)
                }
              }}
            />
            <Button
              variant="outline"
              onClick={() => pagination.onPageChange(pagination.currentPage + 1)}
              disabled={
                pagination.currentPage ===
                Math.ceil(pagination.totalItems / pagination.pageSize)
              }
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              onClick={() =>
                pagination.onPageChange(
                  Math.ceil(pagination.totalItems / pagination.pageSize)
                )
              }
              disabled={
                pagination.currentPage ===
                Math.ceil(pagination.totalItems / pagination.pageSize)
              }
            >
              <ChevronsRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}