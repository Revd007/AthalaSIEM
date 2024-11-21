'use client'

import React, { useState } from 'react'
import { Card } from '../../../components/ui/card'
import { Button } from '../../../components/ui/button'
import { Input } from '../../../components/ui/input'

interface Integration {
  id: string
  name: string
  description: string
  isConnected: boolean
  icon: string
}

const availableIntegrations: Integration[] = [
  {
    id: 'aws',
    name: 'Amazon Web Services',
    description: 'Connect to AWS CloudWatch logs and metrics',
    isConnected: false,
    icon: '/icons/aws.svg'
  },
  {
    id: 'azure',
    name: 'Microsoft Azure',
    description: 'Integrate with Azure Monitor and Log Analytics',
    isConnected: true,
    icon: '/icons/azure.svg'
  }
  // Add more integrations
]

export function IntegrationSettings() {
  const [integrations, setIntegrations] = useState(availableIntegrations)

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Integrations</h3>
        <p className="mt-1 text-sm text-gray-500">
          Connect your SIEM with other security tools and services
        </p>
      </div>

      <div className="grid gap-6">
        {integrations.map((integration) => (
          <div key={integration.id}>
            <Card>
              <div className="p-6 flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <img 
                    src={integration.icon} 
                    alt={integration.name} 
                    className="h-12 w-12"
                  />
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">{integration.name}</h4>
                    <p className="text-sm text-gray-500">{integration.description}</p>
                  </div>
                </div>
                <Button
                  variant={integration.isConnected ? 'outline' : 'primary'}
                  onPress={() => {
                    // Handle connection/disconnection
                  }}
                  type="button"
                >
                  {integration.isConnected ? 'Disconnect' : 'Connect'}
                </Button>
              </div>
            </Card>
          </div>
        ))}
      </div>
    </div>
  )
}