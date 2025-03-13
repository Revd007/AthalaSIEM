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
  Network,
  Shield,
  Lock,
  Eye,
  Filter,
  AlertTriangle,
  Radio,
  Wifi
} from 'lucide-react'

export function NetworkSettings() {
  return (
    <div className="space-y-6">
      {/* Firewall Rules */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Shield className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Firewall Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Whitelisted IPs</label>
              <Textarea 
                placeholder="Enter IPs (one per line)"
                defaultValue="10.0.0.0/8&#10;192.168.1.0/24"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Blacklisted IPs</label>
              <Textarea 
                placeholder="Enter IPs (one per line)"
                defaultValue="185.143.223.0/24&#10;45.95.147.0/24"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Allowed Ports</label>
              <Input defaultValue="80,443,22,3389" placeholder="Comma-separated ports" />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Blocked Ports</label>
              <Input defaultValue="23,25,445" placeholder="Comma-separated ports" />
            </div>
          </div>
        </div>
      </Card>

      {/* IDS/IPS Settings */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Eye className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">IDS/IPS Configuration</h2>
        </div>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Enable IPS Mode</label>
              <p className="text-sm text-gray-500">Automatically block detected threats</p>
            </div>
            <Switch defaultChecked />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Detection Sensitivity</label>
            <Select defaultValue="balanced">
              <SelectTrigger>
                <SelectValue placeholder="Select sensitivity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="low">Low (Fewer False Positives)</SelectItem>
                <SelectItem value="balanced">Balanced</SelectItem>
                <SelectItem value="high">High (Aggressive Detection)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Custom Rules</label>
            <Textarea 
              placeholder="Enter Snort/Suricata rules"
              className="font-mono text-sm"
            />
          </div>
        </div>
      </Card>

      {/* Network Segmentation */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Filter className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Network Segmentation</h2>
        </div>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">VLAN Configuration</label>
            <div className="space-y-2">
              {[
                { name: 'Management Network', id: '10', subnet: '10.0.10.0/24' },
                { name: 'User Network', id: '20', subnet: '10.0.20.0/24' },
                { name: 'Server Network', id: '30', subnet: '10.0.30.0/24' },
                { name: 'IoT Network', id: '40', subnet: '10.0.40.0/24' }
              ].map((vlan) => (
                <div key={vlan.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{vlan.name}</div>
                    <div className="text-sm text-gray-500">VLAN {vlan.id} - {vlan.subnet}</div>
                  </div>
                  <Switch defaultChecked />
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Network Monitoring */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Radio className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Network Monitoring</h2>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Packet Capture Size (MB)</label>
              <Input type="number" defaultValue={100} min={10} max={1000} />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Retention Period (days)</label>
              <Input type="number" defaultValue={30} min={1} max={90} />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Deep Packet Inspection</label>
              <p className="text-sm text-gray-500">Analyze packet contents for threats</p>
            </div>
            <Switch defaultChecked />
          </div>
        </div>
      </Card>

      {/* Wireless Security */}
      <Card className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <Wifi className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold">Wireless Security</h2>
        </div>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <label className="text-sm font-medium">Rogue AP Detection</label>
              <p className="text-sm text-gray-500">Detect unauthorized access points</p>
            </div>
            <Switch defaultChecked />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Allowed Wireless Protocols</label>
            <div className="space-y-2">
              {[
                'WPA3',
                'WPA2-Enterprise',
                'WPA2-Personal',
                'WPA-Personal'
              ].map((protocol) => (
                <div key={protocol} className="flex items-center justify-between">
                  <span className="text-sm">{protocol}</span>
                  <Switch defaultChecked={protocol !== 'WPA-Personal'} />
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