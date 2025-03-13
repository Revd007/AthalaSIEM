'use client'

import { Card } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue, 
} from '@/components/ui/select'
import { 
  History,
  Key,
  Database,
  Zap,
  Plug,
  Brain,
  Shield,
  Search
} from 'lucide-react'

export function MonitoringSettings() {
  return (
    <div className="space-y-6">
      {/* Audit Log Settings */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <History className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Audit Log Settings</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Retention Period (days)</label>
              <Input type="number" defaultValue={90} min={30} max={365} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Log Level</label>
              <Select defaultValue="info">
                <SelectTrigger>
                  <SelectValue placeholder="Select level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="debug">Debug</SelectItem>
                  <SelectItem value="info">Info</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Archive Old Logs</label>
              <p className="text-sm text-gray-500">Automatically archive logs older than retention period</p>
            </div>
            <Switch defaultChecked />
          </div>
        </div>
      </Card>

      {/* API Key Management */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Key className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">API Key Management</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Active API Keys</label>
            <div className="border rounded-lg divide-y">
              <div className="p-4 flex items-center justify-between">
                <div>
                  <div className="font-medium">Production Key</div>
                  <div className="text-sm text-gray-500">Created: 2024-03-01</div>
                </div>
                <Button variant="destructive" size="sm">Revoke</Button>
              </div>
              <div className="p-4 flex items-center justify-between">
                <div>
                  <div className="font-medium">Development Key</div>
                  <div className="text-sm text-gray-500">Created: 2024-03-15</div>
                </div>
                <Button variant="destructive" size="sm">Revoke</Button>
              </div>
            </div>
          </div>
          <Button>Generate New API Key</Button>
        </div>
      </Card>

      {/* Data Retention */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Database className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Data Retention Policies</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Event Data (days)</label>
              <Input type="number" defaultValue={30} min={7} max={365} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Metrics Data (days)</label>
              <Input type="number" defaultValue={90} min={30} max={365} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Alert History (days)</label>
              <Input type="number" defaultValue={180} min={30} max={730} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Report Data (days)</label>
              <Input type="number" defaultValue={365} min={90} max={1095} />
            </div>
          </div>
        </div>
      </Card>

      {/* Automated Response Rules */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Zap className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Automated Response Rules</h2>
        </div>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Enable Automated Response</label>
              <p className="text-sm text-gray-500">Automatically respond to security events</p>
            </div>
            <Switch defaultChecked />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Default Actions</label>
            <div className="space-y-2">
              {['Block IP', 'Disable User', 'Isolate Host'].map((action) => (
                <div key={action} className="flex items-center justify-between">
                  <span className="text-sm">{action}</span>
                  <Switch />
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Machine Learning Configuration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Brain className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Machine Learning Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Enable ML Analysis</label>
              <p className="text-sm text-gray-500">Use ML for anomaly detection</p>
            </div>
            <Switch defaultChecked />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Training Interval (hours)</label>
              <Input type="number" defaultValue={24} min={1} max={168} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Sensitivity</label>
              <Select defaultValue="medium">
                <SelectTrigger>
                  <SelectValue placeholder="Select sensitivity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </Card>

      {/* Threat Intelligence */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Shield className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Threat Intelligence Feeds</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Active Feeds</label>
            <div className="space-y-2">
              {[
                'AlienVault OTX',
                'Abuse.ch',
                'EmergingThreats',
                'Custom Feed 1'
              ].map((feed) => (
                <div key={feed} className="flex items-center justify-between">
                  <span className="text-sm">{feed}</span>
                  <Switch defaultChecked />
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Custom Feed URL</label>
            <Input placeholder="https://..." />
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