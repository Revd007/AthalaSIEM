'use client'

import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { SecuritySettings } from '@/components/Settings/SecuritySettings'
import { AgentSettings } from '@/components/Settings/AgentSettings'
import { MonitoringSettings } from '@/components/Settings/MonitoringSettings'
import { NotificationSettings } from '@/components/Settings/NotificationSettings'
import { NetworkSettings } from '@/components/Settings/NetworkSettings'
import { ComplianceSettings } from '@/components/Settings/ComplianceSettings'
import { BackupSettings } from '@/components/Settings/BackupSettings'
import { IntegrationSettings } from '@/components/Settings/IntegrationSettings'

export default function SettingsPage() {
  const router = useRouter()

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    if (!token) {
      router.push('/login')
    }
  }, [router])
  return (
    <div className="p-8 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>
      
      <Tabs defaultValue="security" className="space-y-6">
        <TabsList className="grid grid-cols-4 lg:grid-cols-8 gap-2">
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
          <TabsTrigger value="backup">Backup</TabsTrigger>
          <TabsTrigger value="integrations">Integrations</TabsTrigger>
        </TabsList>

        <TabsContent value="security">
          <SecuritySettings />
        </TabsContent>
        
        <TabsContent value="agents">
          <AgentSettings />
        </TabsContent>
        
        <TabsContent value="monitoring">
          <MonitoringSettings />
        </TabsContent>
        
        <TabsContent value="notifications">
          <NotificationSettings />
        </TabsContent>
        
        <TabsContent value="network">
          <NetworkSettings />
        </TabsContent>
        
        <TabsContent value="compliance">
          <ComplianceSettings />
        </TabsContent>
        
        <TabsContent value="backup">
          <BackupSettings />
        </TabsContent>
        
        <TabsContent value="integrations">
          <IntegrationSettings />
        </TabsContent>
      </Tabs>
    </div>
  )
} 