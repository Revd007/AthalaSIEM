import React, { useState } from 'react';
import { Search, Filter, Clock, Download } from 'lucide-react';

const timeRanges = ['Last 24 hours', 'Last 7 days', 'Last 30 days', 'Custom'];
const eventTypes = ['All Events', 'Security Alerts', 'System Logs', 'Network Traffic', 'User Activity'];
const severityLevels = ['All', 'Critical', 'High', 'Medium', 'Low'];

export function AdvancedSearch() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTimeRange, setSelectedTimeRange] = useState(timeRanges[0]);
  const [selectedEventType, setSelectedEventType] = useState(eventTypes[0]);
  const [selectedSeverity, setSelectedSeverity] = useState(severityLevels[0]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="space-y-4">
        <div className="relative">
          <input
            type="text"
            placeholder="Search logs, alerts, and events..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="border rounded-lg p-2 dark:bg-gray-700 dark:border-gray-600"
          >
            {timeRanges.map(range => (
              <option key={range} value={range}>{range}</option>
            ))}
          </select>

          <select
            value={selectedEventType}
            onChange={(e) => setSelectedEventType(e.target.value)}
            className="border rounded-lg p-2 dark:bg-gray-700 dark:border-gray-600"
          >
            {eventTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>

          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="border rounded-lg p-2 dark:bg-gray-700 dark:border-gray-600"
          >
            {severityLevels.map(level => (
              <option key={level} value={level}>{level}</option>
            ))}
          </select>
        </div>

        <div className="flex justify-between">
          <button className="flex items-center px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
            <Search className="w-4 h-4 mr-2" />
            Search
          </button>
          <div className="flex space-x-2">
            <button className="flex items-center px-4 py-2 border rounded-lg dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700">
              <Filter className="w-4 h-4 mr-2" />
              Save Filter
            </button>
            <button className="flex items-center px-4 py-2 border rounded-lg dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700">
              <Download className="w-4 h-4 mr-2" />
              Export
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}