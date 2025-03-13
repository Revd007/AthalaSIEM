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
  ShieldCheck,
  FileCheck,
  ClipboardList,
  Calendar,
  Settings2,
  AlertCircle,
  FileText,
  Upload
} from 'lucide-react'

export function ComplianceSettings() {
  return (
    <div className="space-y-6">
      {/* Compliance Frameworks */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <ShieldCheck className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Compliance Frameworks</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Active Frameworks</label>
            <div className="space-y-2">
              {[
                { name: 'ISO 27001', description: 'Information Security Management' },
                { name: 'NIST CSF', description: 'Cybersecurity Framework' },
                { name: 'PCI DSS', description: 'Payment Card Industry Data Security Standard' },
                { name: 'HIPAA', description: 'Health Insurance Portability and Accountability Act' }
              ].map((framework) => (
                <div key={framework.name} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{framework.name}</div>
                    <div className="text-sm text-gray-500">{framework.description}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">Configure</Button>
                    <Switch defaultChecked />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Audit Configuration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <ClipboardList className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Audit Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Audit Frequency</label>
              <Select defaultValue="monthly">
                <SelectTrigger>
                  <SelectValue placeholder="Select frequency" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="weekly">Weekly</SelectItem>
                  <SelectItem value="monthly">Monthly</SelectItem>
                  <SelectItem value="quarterly">Quarterly</SelectItem>
                  <SelectItem value="annually">Annually</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Retention Period (months)</label>
              <Input type="number" defaultValue={36} min={12} max={84} />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Automated Audits</label>
              <p className="text-sm text-gray-500">Schedule and run automated compliance checks</p>
            </div>
            <Switch defaultChecked />
          </div>
        </div>
      </Card>

      {/* Evidence Collection */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <FileCheck className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Evidence Collection</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Evidence Types</label>
            <div className="space-y-2">
              {[
                'System Logs',
                'Configuration Snapshots',
                'Access Reports',
                'Security Scans',
                'Policy Documents'
              ].map((type) => (
                <div key={type} className="flex items-center justify-between">
                  <span className="text-sm">{type}</span>
                  <Switch defaultChecked />
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Storage Location</label>
            <Input placeholder="s3://compliance-evidence/" />
          </div>
        </div>
      </Card>

      {/* Reporting Configuration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <FileText className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Reporting Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Report Format</label>
              <Select defaultValue="pdf">
                <SelectTrigger>
                  <SelectValue placeholder="Select format" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pdf">PDF</SelectItem>
                  <SelectItem value="html">HTML</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Schedule Reports</label>
              <Select defaultValue="monthly">
                <SelectTrigger>
                  <SelectValue placeholder="Select schedule" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="weekly">Weekly</SelectItem>
                  <SelectItem value="monthly">Monthly</SelectItem>
                  <SelectItem value="quarterly">Quarterly</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Report Recipients</label>
            <Textarea placeholder="Enter email addresses (one per line)" />
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