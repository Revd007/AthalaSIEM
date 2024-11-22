import { Input } from '../../ui/input'
import { Button } from '../../ui/button'
import React from 'react'


export function GeneralSettings() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium leading-6 text-gray-900">System Information</h3>
        <div className="mt-4 grid grid-cols-1 gap-y-6 sm:grid-cols-2 sm:gap-x-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">System Name</label>
            <Input
              defaultValue="SIEM System"
              placeholder="Enter system name"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Time Zone</label>
            <Input
              defaultValue="UTC+00:00"
              type="select"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Data Retention Period</label>
            <Input
              defaultValue="30"
              type="number"
              min="1"
              max="365"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Log Format</label>
            <Input
              defaultValue="JSON"
              type="select"
            />
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-medium leading-6 text-gray-900">Contact Information</h3>
        <div className="mt-4 grid grid-cols-1 gap-y-6 sm:grid-cols-2 sm:gap-x-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Admin Email</label>
            <Input
              type="email"
              placeholder="admin@example.com"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">Support Phone</label>
            <Input
              type="tel"
              placeholder="+1 (555) 000-0000"
            />
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <Button variant="outline" className="mr-3">
          Cancel
        </Button>
        <Button variant="primary">
          Save Changes
        </Button>
      </div>
    </div>
  )
}