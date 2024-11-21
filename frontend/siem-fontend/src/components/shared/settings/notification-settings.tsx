'use client'

import React, { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Tabs } from '@/components/ui/tabs'
import { GeneralSettings } from './general-settings'
import { SecuritySettings } from './security-settings'
import { NotificationSettings } from './notification-settings'
import { IntegrationSettings } from './integration-settings'

const tabs = [
  { id: 'general', label: 'General' },
  { id: 'security', label: 'Security' },
  { id: 'notifications', label: 'Notifications' },
  { id: 'integrations', label: 'Integrations' }
]

export default function Settings() {
  const [activeTab, setActiveTab] = useState('general')

  return (
    <div className="py-6">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <h1 className="text-2xl font-semibold text-gray-900">Settings</h1>
        
        <div className="mt-6">
          <Card className="p-6">
            <Tabs 
              tabs={tabs} 
              activeTab={activeTab} 
              onChange={setActiveTab} 
            />
            
            <div className="mt-6">
              {activeTab === 'general' && <GeneralSettings />}
              {activeTab === 'security' && <SecuritySettings />}
              {activeTab === 'notifications' && <NotificationSettings />}
              {activeTab === 'integrations' && <IntegrationSettings />}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}