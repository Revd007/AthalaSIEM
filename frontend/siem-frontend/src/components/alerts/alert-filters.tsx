import React from 'react';
import { Select } from '../ui/select';
import { Button } from '../ui/button';

interface AlertFiltersProps {
  filters: {
    severity: string;
    status: string;
    timeRange: string;
  };
  onChange: (filters: any) => void;
}

export function AlertFilters({ filters, onChange }: AlertFiltersProps) {
  return (
    <div className="flex space-x-4 mb-4">
      <Select
        value={filters.severity}
        onValueChange={(value) => onChange({ ...filters, severity: value })}
      >
        <option value="all">All Severities</option>
        <option value="low">Low</option>
        <option value="medium">Medium</option>
        <option value="high">High</option>
        <option value="critical">Critical</option>
      </Select>

      <Select
        value={filters.status}
        onValueChange={(value) => onChange({ ...filters, status: value })}
      >
        <option value="all">All Status</option>
        <option value="new">New</option>
        <option value="investigating">Investigating</option>
        <option value="resolved">Resolved</option>
      </Select>

      <Select
        value={filters.timeRange}
        onValueChange={(value) => onChange({ ...filters, timeRange: value })}
      >
        <option value="24h">Last 24 Hours</option>
        <option value="7d">Last 7 Days</option>
        <option value="30d">Last 30 Days</option>
        <option value="custom">Custom Range</option>
      </Select>

      <Button
        variant="outline"
        onClick={() => onChange({
          severity: 'all',
          status: 'all',
          timeRange: '24h'
        })}
      >
        Reset Filters
      </Button>
    </div>
  );
}