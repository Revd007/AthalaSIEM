import React from 'react';
import { Table } from './table';
import { LoadingSkeleton } from './loading-skeleton';

interface Column {
  key: string;
  title: string;
  render?: (item: any) => React.ReactNode;
}

interface DataTableProps<T> {
  data: T[];
  columns: { key: string; title: string }[];
  loading?: boolean;
}

export function DataTable<T>({ data, columns, loading = false }: DataTableProps<T>) {
  if (loading) {
    return <LoadingSkeleton rows={5} />;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {column.title}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((row: any, i) => (
            <tr key={i}>
              {columns.map((column) => (
                <td
                  key={column.key}
                  className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                >
                  {row[column.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}