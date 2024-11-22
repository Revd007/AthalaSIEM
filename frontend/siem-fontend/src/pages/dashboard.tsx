import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Calendar } from '../components/ui/calendar';
import { Button } from '../components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Popover, PopoverContent, PopoverTrigger } from '../components/ui/popover';
import { Input } from '../components/ui/input';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, Cell } from 'recharts';
import { Search, Calendar as CalendarIcon, Filter, Download, Settings, RefreshCcw } from 'lucide-react';

const Dashboard = () => {
  const [dateRange, setDateRange] = useState({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date()
  });
  const [selectedView, setSelectedView] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Enhanced sample data
  const logTrendData = Array.from({ length: 30 }, (_, i) => ({
    date: `Day ${i + 1}`,
    events: Math.floor(Math.random() * 5000) + 8000,
    errors: Math.floor(Math.random() * 1000),
    warnings: Math.floor(Math.random() * 2000)
  }));

  const securityEventsData = [
    { name: 'Login', success: 24241, failed: 2000, total: 26241, color: '#4299e1' },
    { name: 'Account Login', success: 2281, failed: 300, total: 2581, color: '#48bb78' },
    { name: 'Account Management', success: 1247, failed: 100, total: 1347, color: '#ed8936' },
    { name: 'Object Access', success: 44263, failed: 1000, total: 45263, color: '#9f7aea' },
    { name: 'System Events', success: 57, failed: 10, total: 67, color: '#f56565' }
  ];

  const severityData = [
    { name: 'Critical', value: 156, color: '#ef4444' },
    { name: 'High', value: 342, color: '#f97316' },
    { name: 'Medium', value: 897, color: '#eab308' },
    { name: 'Low', value: 2345, color: '#22c55e' }
  ];

  const timeDistributionData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    events: Math.floor(Math.random() * 1000) + 100
  }));

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      {/* Header with Enhanced Controls */}
      <div className="mb-6 space-y-4">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-800">EventLog Analyzer</h1>
          <div className="flex gap-4">
            <Button variant="outline">
              <RefreshCcw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
            <Button variant="outline">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
            <Button variant="outline">
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </Button>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="flex gap-4 flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <Input
              placeholder="Search logs..."
              value={searchQuery}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
              className="w-full"
            />
          </div>
          
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" className="w-[240px] justify-start text-left font-normal">
                <CalendarIcon className="w-4 h-4 mr-2" />
                {dateRange.from ? (
                  <span>
                    {dateRange.from.toLocaleDateString()} - {dateRange.to.toLocaleDateString()}
                  </span>
                ) : (
                  <span>Pick a date range</span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="range"
                selected={{ from: dateRange.from, to: dateRange.to }}
                onSelect={(range) => {
                  if (range?.from && range?.to) {
                    setDateRange({ from: range.from, to: range.to });
                  }
                }}
                initialFocus
              />
            </PopoverContent>
          </Popover>

          <Select value={selectedView} onValueChange={setSelectedView}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select view" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Events</SelectItem>
              <SelectItem value="security">Security Events</SelectItem>
              <SelectItem value="system">System Events</SelectItem>
              <SelectItem value="application">Application Events</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Enhanced Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Event Overview Cards */}
        <Card className="col-span-full">
          <CardHeader>
            <CardTitle>Event Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[
                { label: 'Total Events', value: '54,168', change: '+12%' },
                { label: 'Critical Events', value: '156', change: '-5%' },
                { label: 'Failed Logins', value: '2,341', change: '+8%' },
                { label: 'System Alerts', value: '892', change: '+15%' }
              ].map((stat, i) => (
                <div key={i} className="bg-white p-4 rounded-lg shadow-sm">
                  <p className="text-sm text-gray-500">{stat.label}</p>
                  <p className="text-2xl font-bold">{stat.value}</p>
                  <p className={`text-sm ${stat.change.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>
                    {stat.change} from last period
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Enhanced Log Trend Chart */}
        <Card className="col-span-full">
          <CardHeader>
            <CardTitle>Log Trend Analysis</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={logTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="events" stroke="#3b82f6" name="Total Events" />
                <Line type="monotone" dataKey="errors" stroke="#ef4444" name="Errors" />
                <Line type="monotone" dataKey="warnings" stroke="#f59e0b" name="Warnings" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Security Events Distribution */}
        <Card className="col-span-2">
          <CardHeader>
            <CardTitle>Security Events Distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={securityEventsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="success" stackId="a" fill="#22c55e" name="Success" />
                <Bar dataKey="failed" stackId="a" fill="#ef4444" name="Failed" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Event Severity Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Event Severity</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={severityData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label
                >
                  {severityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* 24-Hour Event Distribution */}
        <Card className="col-span-full">
          <CardHeader>
            <CardTitle>24-Hour Event Distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={timeDistributionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="events" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;