import React from 'react';
import { Table } from './table';

interface Column {
  key: string;
  title: string;
  render?: (item: any) => React.ReactNode;
}

interface DataTableProps {
  data: any[];
  columns: Column[];
}

export function DataTable({ data, columns }: DataTableProps): JSX.Element {
  return (
    <div className="rounded-md border">
      <Table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} className="px-4 py-2">
                {column.title}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((item, index) => (
            <tr key={item.id || index}>
              {columns.map((column) => (
                <td key={`${item.id}-${column.key}`} className="px-4 py-2">
                  {column.render ? column.render(item) : item[column.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
}