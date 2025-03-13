'use client'

import { useState } from 'react'
import { AlertTriangle, CheckCircle, Clock } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { ComplianceFramework } from '@/types/compliance'

interface RemediationItem {
  id: string
  controlId: string
  issue: string
  status: 'open' | 'in-progress' | 'resolved'
  priority: 'high' | 'medium' | 'low'
  assignee: string
  dueDate: string
  progress: number
}

interface ComplianceRemediationProps {
  framework: ComplianceFramework
}

export function ComplianceRemediation({ framework }: ComplianceRemediationProps) {
  const [items] = useState<RemediationItem[]>([
    {
      id: '1',
      controlId: 'A.5.1.1',
      issue: 'Missing documentation for information security policy review process',
      status: 'open',
      priority: 'high',
      assignee: 'John Doe',
      dueDate: '2024-04-01',
      progress: 0
    },
    // Add more items...
  ])

  const statusConfig = {
    open: { icon: AlertTriangle, color: 'text-red-500', bg: 'bg-red-50' },
    'in-progress': { icon: Clock, color: 'text-yellow-500', bg: 'bg-yellow-50' },
    resolved: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50' }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-semibold tracking-tight">
            Remediation Tasks
          </h2>
          <p className="text-sm text-muted-foreground">
            Track and manage compliance remediation activities
          </p>
        </div>
        <Button>
          Create Task
        </Button>
      </div>

      <div className="space-y-4">
        {items.map((item) => {
          const StatusIcon = statusConfig[item.status].icon
          
          return (
            <div
              key={item.id}
              className="p-4 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
            >
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium">Control {item.controlId}</span>
                    <span className={`
                      inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                      ${statusConfig[item.status].bg}
                      ${statusConfig[item.status].color}
                    `}>
                      <StatusIcon className="w-4 h-4 mr-1" />
                      {item.status}
                    </span>
                    <span className={`
                      inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                      ${item.priority === 'high' ? 'bg-red-50 text-red-700' :
                        item.priority === 'medium' ? 'bg-yellow-50 text-yellow-700' :
                        'bg-blue-50 text-blue-700'}
                    `}>
                      {item.priority} priority
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    {item.issue}
                  </p>
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <span>Assignee: {item.assignee}</span>
                    <span>Due: {new Date(item.dueDate).toLocaleDateString()}</span>
                  </div>
                </div>
                
                <Button variant="outline" size="sm">
                  Update Status
                </Button>
              </div>
              
              <div className="mt-4">
                <div className="flex items-center justify-between text-sm">
                  <span>Progress</span>
                  <span>{item.progress}%</span>
                </div>
                <div className="mt-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500"
                    style={{ width: `${item.progress}%` }}
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
} 