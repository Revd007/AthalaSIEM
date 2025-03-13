'use client'

import { Card } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue, 
} from '@/components/ui/select'
import { Shield, Lock, Key } from 'lucide-react'

export function SecuritySettings() {
  return (
    <div className="space-y-6">
      {/* SSL/TLS Configuration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Shield className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">SSL/TLS Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">SSL Certificate</label>
              <Input type="file" accept=".crt,.pem" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Private Key</label>
              <Input type="file" accept=".key,.pem" />
            </div>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Minimum TLS Version</label>
            <Select defaultValue="1.2">
              <SelectTrigger>
                <SelectValue placeholder="Select TLS version" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1.1">TLS 1.1</SelectItem>
                <SelectItem value="1.2">TLS 1.2</SelectItem>
                <SelectItem value="1.3">TLS 1.3</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Strict Transport Security (HSTS)</label>
              <p className="text-sm text-gray-500">Enforce HTTPS connections</p>
            </div>
            <Switch />
          </div>
        </div>
      </Card>

      {/* Access Control */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Lock className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Access Control</h2>
        </div>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Two-Factor Authentication</label>
              <p className="text-sm text-gray-500">Require 2FA for all users</p>
            </div>
            <Switch />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">IP Whitelisting</label>
              <p className="text-sm text-gray-500">Restrict access to specific IPs</p>
            </div>
            <Switch />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Session Timeout (minutes)</label>
            <Input type="number" defaultValue={30} min={5} max={120} />
          </div>
        </div>
      </Card>

      {/* Password Policy */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Key className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Password Policy</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Minimum Length</label>
              <Input type="number" defaultValue={12} min={8} max={32} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Maximum Age (days)</label>
              <Input type="number" defaultValue={90} min={30} max={180} />
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Require Special Characters</label>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Require Numbers</label>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Require Mixed Case</label>
              <Switch defaultChecked />
            </div>
          </div>
        </div>
      </Card>

      <div className="flex justify-end space-x-4">
        <Button variant="outline">Reset to Defaults</Button>
        <Button>Save Changes</Button>
      </div>
    </div>
  )
} 