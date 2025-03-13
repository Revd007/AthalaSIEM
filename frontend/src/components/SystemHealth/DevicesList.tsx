'use client'

import { useState } from 'react'
import { Server, Shield, Network, Monitor, Database, AlertTriangle } from 'lucide-react'
import { Card } from '@/components/ui/card'
import type { SystemDevice, DeviceType } from '@/types/system-health'

interface DevicesListProps {
  selectedDevice: string | null
  onDeviceSelect: (deviceId: string) => void
  typeFilter: DeviceType[]
  searchQuery: string
  statusFilter: string[]
}

const mockDevices: SystemDevice[] = [
  {
    id: '1',
    name: 'Main Firewall',
    type: 'firewall',
    status: 'healthy',
    ipAddress: '192.168.1.1',
    location: 'Main DC',
    lastSeen: '2024-03-19T10:00:00Z',
    agentVersion: '2.0.0',
    manufacturer: 'Palo Alto',
    model: 'PA-3260'
  },
  {
    id: '2',
    name: 'Web Server 01',
    type: 'server',
    status: 'healthy',
    ipAddress: '192.168.1.10',
    location: 'Main DC',
    lastSeen: '2024-03-19T10:05:00Z',
    agentVersion: '2.0.0',
    manufacturer: 'Dell',
    model: 'PowerEdge R740'
  },
  {
    id: '3', 
    name: 'Syslog Server',
    type: 'syslog',
    status: 'warning',
    ipAddress: '192.168.1.15',
    location: 'Main DC',
    lastSeen: '2024-03-19T09:55:00Z',
    agentVersion: '2.0.0',
    manufacturer: 'HP',
    model: 'ProLiant DL380'
  },
  {
    id: '4',
    name: 'AWS EC2 Instance',
    type: 'cloud',
    status: 'healthy',
    ipAddress: '10.0.1.100',
    location: 'AWS us-east-1',
    lastSeen: '2024-03-19T10:02:00Z',
    agentVersion: '2.0.0',
    manufacturer: 'Amazon',
    model: 't3.large'
  },
  {
    id: '5',
    name: 'Linux App Server',
    type: 'server',
    status: 'critical',
    ipAddress: '192.168.1.20',
    location: 'Main DC',
    lastSeen: '2024-03-19T09:30:00Z', 
    agentVersion: '2.0.0',
    manufacturer: 'NetApp',
    model: 'FAS8700'
  },
  {
    id: '6',
    name: 'IDS Sensor',
    type: 'security-appliance',
    status: 'warning',
    ipAddress: '192.168.1.40',
    location: 'Main DC',
    lastSeen: '2024-03-19T09:58:00Z',
    agentVersion: '2.0.0',
    manufacturer: 'Cisco',
    model: 'ASA 5500-X'
  }
]

const statusConfig = {
  healthy: { color: 'text-green-500', bgColor: 'bg-green-50' },
  warning: { color: 'text-yellow-500', bgColor: 'bg-yellow-50' },
  critical: { color: 'text-red-500', bgColor: 'bg-red-50' },
  offline: { color: 'text-gray-500', bgColor: 'bg-gray-50' }
}

export function DevicesList({
  selectedDevice,
  onDeviceSelect,
  typeFilter,
  searchQuery,
  statusFilter
}: DevicesListProps) {
  const filteredDevices = mockDevices.filter(device => {
    if (typeFilter.length && !typeFilter.includes(device.type)) return false
    if (statusFilter.length && !statusFilter.includes(device.status)) return false
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      return (
        device.name.toLowerCase().includes(query) ||
        device.ipAddress.includes(query) ||
        device.location.toLowerCase().includes(query)
      )
    }
    return true
  })

  return (
    <Card>
      <div className="p-4 space-y-4">
        <h2 className="font-semibold">Devices ({filteredDevices.length})</h2>
        <div className="space-y-2">
          {filteredDevices.map(device => (
            <div
              key={device.id}
              onClick={() => onDeviceSelect(device.id)}
              className={`w-full p-4 rounded-lg border transition-colors cursor-pointer ${
                selectedDevice === device.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 hover:border-blue-500 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <div className="font-medium">{device.name}</div>
                  <div className="text-sm text-gray-500">{device.ipAddress}</div>
                  <div className="flex items-center space-x-2 text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      statusConfig[device.status].bgColor
                    } ${statusConfig[device.status].color}`}>
                      {device.status}
                    </span>
                    <span className="text-gray-500">{device.location}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
} 