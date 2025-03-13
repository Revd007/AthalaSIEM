'use client'

import { Card } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue, 
} from '@/components/ui/select'
import { 
  Database,
  Cloud,
  Calendar,
  HardDrive,
  History,
  RefreshCw,
  Lock,
  Download
} from 'lucide-react'

export function BackupSettings() {
  return (
    <div className="space-y-6">
      {/* Automated Backup */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <RefreshCw className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Automated Backup</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Backup Frequency</label>
              <Select defaultValue="daily">
                <SelectTrigger>
                  <SelectValue placeholder="Select frequency" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hourly">Every Hour</SelectItem>
                  <SelectItem value="daily">Daily</SelectItem>
                  <SelectItem value="weekly">Weekly</SelectItem>
                  <SelectItem value="monthly">Monthly</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Retention Period (days)</label>
              <Input type="number" defaultValue={30} min={7} max={365} />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Backup Components</label>
            <div className="space-y-2">
              {[
                { name: 'Configuration Data', size: '250MB' },
                { name: 'Event Logs', size: '2.5GB' },
                { name: 'Alert History', size: '500MB' },
                { name: 'Reports', size: '750MB' },
                { name: 'Custom Rules', size: '100MB' }
              ].map((component) => (
                <div key={component.name} className="flex items-center justify-between">
                  <div>
                    <span className="text-sm">{component.name}</span>
                    <span className="text-xs text-gray-500 ml-2">({component.size})</span>
                  </div>
                  <Switch defaultChecked />
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Storage Configuration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Cloud className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Storage Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Storage Provider</label>
            <Select defaultValue="s3">
              <SelectTrigger>
                <SelectValue placeholder="Select provider" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="s3">Amazon S3</SelectItem>
                <SelectItem value="azure">Azure Blob Storage</SelectItem>
                <SelectItem value="gcs">Google Cloud Storage</SelectItem>
                <SelectItem value="local">Local Storage</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Bucket/Container</label>
              <Input placeholder="backup-bucket" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Path Prefix</label>
              <Input placeholder="siem/backups/" />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Encryption</label>
              <p className="text-sm text-gray-500">Enable server-side encryption</p>
            </div>
            <Switch defaultChecked />
          </div>
        </div>
      </Card>

      {/* Restore Points */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <History className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Restore Points</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Available Backups</label>
            <div className="space-y-2">
              {[
                { date: '2024-03-19 00:00', size: '4.1GB', type: 'Full' },
                { date: '2024-03-18 00:00', size: '4.0GB', type: 'Full' },
                { date: '2024-03-17 00:00', size: '3.9GB', type: 'Full' }
              ].map((backup) => (
                <div key={backup.date} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{backup.date}</div>
                    <div className="text-sm text-gray-500">
                      {backup.type} Backup • {backup.size}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-1" />
                      Download
                    </Button>
                    <Button size="sm">Restore</Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Backup Status */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <HardDrive className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Backup Status</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Storage Usage</span>
              <span>4.1GB / 10GB</span>
            </div>
            <Progress value={41} />
          </div>

          <div className="text-sm text-gray-500">
            Last successful backup: 2024-03-19 00:00 UTC
          </div>
          <div className="text-sm text-gray-500">
            Next scheduled backup: 2024-03-20 00:00 UTC
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