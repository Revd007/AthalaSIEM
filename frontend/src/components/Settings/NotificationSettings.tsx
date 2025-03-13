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
  Mail,
  Webhook,
  MessageSquare,
  Bell,
  Smartphone,
  AlertTriangle,
  Clock,
  Settings
} from 'lucide-react'

export function NotificationSettings() {
  return (
    <div className="space-y-6">
      {/* Email Configuration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Mail className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Email Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">SMTP Server</label>
              <Input placeholder="smtp.example.com" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">SMTP Port</label>
              <Input type="number" defaultValue={587} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Username</label>
              <Input placeholder="notifications@company.com" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Password</label>
              <Input type="password" />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Use TLS</label>
              <p className="text-sm text-gray-500">Secure email transmission</p>
            </div>
            <Switch defaultChecked />
          </div>
        </div>
      </Card>

      {/* Webhook Integration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Webhook className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Webhook Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Active Webhooks</label>
            <div className="space-y-2">
              {[
                { name: 'Slack Alerts', url: 'https://hooks.slack.com/...' },
                { name: 'Teams Notifications', url: 'https://outlook.office.com/...' },
                { name: 'Custom Endpoint', url: 'https://api.company.com/...' }
              ].map((webhook) => (
                <div key={webhook.name} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{webhook.name}</div>
                    <div className="text-sm text-gray-500">{webhook.url}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">Test</Button>
                    <Switch defaultChecked />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <Button>Add New Webhook</Button>
        </div>
      </Card>

      {/* Alert Rules */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <AlertTriangle className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Alert Rules</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Severity Levels</label>
              <div className="space-y-2">
                {[
                  { level: 'Critical', channels: ['email', 'slack', 'sms'] },
                  { level: 'High', channels: ['email', 'slack'] },
                  { level: 'Medium', channels: ['slack'] },
                  { level: 'Low', channels: ['dashboard'] }
                ].map((rule) => (
                  <div key={rule.level} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <div className="font-medium">{rule.level}</div>
                      <div className="text-sm text-gray-500">
                        Channels: {rule.channels.join(', ')}
                      </div>
                    </div>
                    <Button variant="outline" size="sm">Edit</Button>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Notification Schedule</label>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Working Hours (9:00-17:00)</span>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">After Hours</span>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Weekends</span>
                  <Switch />
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Mobile Notifications */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Smartphone className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Mobile Notifications</h2>
        </div>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Push Notifications</label>
              <p className="text-sm text-gray-500">Send alerts to mobile devices</p>
            </div>
            <Switch defaultChecked />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Registered Devices</label>
            <div className="space-y-2">
              {[
                { name: 'iPhone 13 Pro', id: 'device1', owner: 'John Doe' },
                { name: 'Samsung S21', id: 'device2', owner: 'Jane Smith' }
              ].map((device) => (
                <div key={device.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{device.name}</div>
                    <div className="text-sm text-gray-500">Owner: {device.owner}</div>
                  </div>
                  <Button variant="destructive" size="sm">Remove</Button>
                </div>
              ))}
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