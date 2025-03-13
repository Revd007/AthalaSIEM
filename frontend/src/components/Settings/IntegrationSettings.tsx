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
  Link,
  Shield,
  Zap,
  MessageSquare,
  Ticket,
  AlertTriangle,
  Database,
  Workflow
} from 'lucide-react'

export function IntegrationSettings() {
  return (
    <div className="space-y-6">
      {/* SIEM Integration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Database className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">SIEM Integration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Active Integrations</label>
            <div className="space-y-2">
              {[
                { name: 'Splunk Enterprise', url: 'https://splunk.company.com', status: 'connected' },
                { name: 'Elastic Security', url: 'https://elastic.company.com', status: 'connected' },
                { name: 'IBM QRadar', url: 'https://qradar.company.com', status: 'disconnected' }
              ].map((integration) => (
                <div key={integration.name} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{integration.name}</div>
                    <div className="text-sm text-gray-500">{integration.url}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      integration.status === 'connected' 
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {integration.status}
                    </span>
                    <Button variant="outline" size="sm">Configure</Button>
                    <Switch defaultChecked={integration.status === 'connected'} />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <Button>Add New Integration</Button>
        </div>
      </Card>

      {/* SOAR Integration */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Workflow className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">SOAR Integration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Platform</label>
              <Select defaultValue="phantom">
                <SelectTrigger>
                  <SelectValue placeholder="Select platform" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="phantom">Splunk Phantom</SelectItem>
                  <SelectItem value="demisto">Palo Alto Cortex XSOAR</SelectItem>
                  <SelectItem value="swimlane">Swimlane</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
              <Input type="password" placeholder="Enter API key" />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Playbooks</label>
            <div className="space-y-2">
              {[
                'Malware Response',
                'Phishing Investigation',
                'Data Exfiltration',
                'Ransomware Mitigation'
              ].map((playbook) => (
                <div key={playbook} className="flex items-center justify-between">
                  <span className="text-sm">{playbook}</span>
                  <Switch defaultChecked />
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Threat Intelligence */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Shield className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Threat Intelligence Platforms</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Connected Platforms</label>
            <div className="space-y-2">
              {[
                { name: 'VirusTotal', type: 'API', key: '********' },
                { name: 'AlienVault OTX', type: 'API', key: '********' },
                { name: 'IBM X-Force', type: 'API', key: '********' }
              ].map((platform) => (
                <div key={platform.name} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{platform.name}</div>
                    <div className="text-sm text-gray-500">Integration Type: {platform.type}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">Update Key</Button>
                    <Switch defaultChecked />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Ticketing Systems */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Ticket className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Ticketing Integration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">System</label>
              <Select defaultValue="jira">
                <SelectTrigger>
                  <SelectValue placeholder="Select system" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="jira">Jira Service Management</SelectItem>
                  <SelectItem value="servicenow">ServiceNow</SelectItem>
                  <SelectItem value="zendesk">Zendesk</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Project/Queue</label>
              <Input placeholder="Enter project key" />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Default Template</label>
            <Textarea 
              placeholder="Enter ticket template"
              defaultValue={`Title: [Severity] Alert Description\nPriority: {severity}\nDescription: {description}\nSource: {source}\nTimestamp: {timestamp}`}
            />
          </div>
        </div>
      </Card>

      {/* Communication Platforms */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <MessageSquare className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Communication Platforms</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Connected Platforms</label>
            <div className="space-y-2">
              {[
                { name: 'Slack', channel: '#security-alerts' },
                { name: 'Microsoft Teams', channel: 'Security Team' },
                { name: 'Discord', channel: 'incident-response' }
              ].map((platform) => (
                <div key={platform.name} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{platform.name}</div>
                    <div className="text-sm text-gray-500">Channel: {platform.channel}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">Test</Button>
                    <Switch defaultChecked />
                  </div>
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