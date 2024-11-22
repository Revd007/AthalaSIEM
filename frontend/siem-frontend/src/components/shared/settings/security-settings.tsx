import { Input } from '../../ui/input'
import { Button } from '../../ui/button'
import { Switch } from '../../ui/switch'
import React from 'react'

export function SecuritySettings() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium leading-6 text-gray-900">Authentication</h3>
        <div className="mt-4 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900">Two-Factor Authentication</h4>
              <p className="text-sm text-gray-500">Add an extra layer of security to your account</p>
            </div>
            <Switch />
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-gray-900">Single Sign-On (SSO)</h4>
              <p className="text-sm text-gray-500">Enable SSO integration</p>
            </div>
            <Switch />
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-medium leading-6 text-gray-900">Password Policy</h3>
        <div className="mt-4 grid grid-cols-1 gap-y-6 sm:grid-cols-2 sm:gap-x-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Minimum Password Length</label>
            <Input
              type="number"
              defaultValue="8"
              min="8"
              max="32"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Password Expiry (days)</label>
            <Input
              type="number"
              defaultValue="90"
              min="1"
              max="365"
            />
          </div>
        </div>
        <div className="mt-4 space-y-4">
          <div className="flex items-center">
            <input type="checkbox" className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded" />
            <label className="ml-2 text-sm text-gray-700">Require uppercase letters</label>
          </div>
          <div className="flex items-center">
            <input type="checkbox" className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded" />
            <label className="ml-2 text-sm text-gray-700">Require numbers</label>
          </div>
          <div className="flex items-center">
            <input type="checkbox" className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded" />
            <label className="ml-2 text-sm text-gray-700">Require special characters</label>
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