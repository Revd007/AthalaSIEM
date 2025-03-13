'use client'

import { Card } from '@/components/ui/card'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible'
import { ChevronRight, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'
import type { ComplianceFramework } from '@/types/compliance'

interface ComplianceControlsListProps {
  framework: ComplianceFramework
}

type ControlStatus = 'compliant' | 'non-compliant' | 'in-progress'

interface Control {
  id: string
  title: string
  status: ControlStatus
  lastAssessed: string
  nextAssessment: string
  evidence: string[]
  assignee: string
}

const mockControls = [
  {
    section: 'A.5 Information Security Policies',
    controls: [
      {
        id: 'A.5.1.1',
        title: 'Policies for information security',
        status: 'compliant',
        lastAssessed: '2024-02-15',
        nextAssessment: '2024-05-15',
        evidence: ['policy.pdf', 'review.doc'],
        assignee: 'John Doe'
      },
      {
        id: 'A.5.1.2',
        title: 'Review of the policies for information security',
        status: 'non-compliant',
        lastAssessed: '2024-02-15',
        nextAssessment: '2024-05-15',
        evidence: [],
        assignee: 'Jane Smith'
      }
    ]
  },
  // Add more sections...
]

const statusConfig: Record<ControlStatus, { icon: any; color: string; bg: string }> = {
  compliant: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50' },
  'non-compliant': { icon: XCircle, color: 'text-red-500', bg: 'bg-red-50' },
  'in-progress': { icon: AlertTriangle, color: 'text-yellow-500', bg: 'bg-yellow-50' }
}

export function ComplianceControlsList({ framework }: ComplianceControlsListProps) {
  return (
    <div className="space-y-4">
      {mockControls.map((section) => (
        <Collapsible key={section.section}>
          <Card>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between p-4">
                <h3 className="font-medium">{section.section}</h3>
                <ChevronRight className="h-4 w-4" />
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="p-4 pt-0 space-y-4">
                {section.controls.map((control) => {
                  const StatusIcon = statusConfig[control.status].icon
                  return (
                    <div
                      key={control.id}
                      className="p-4 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center space-x-2">
                            <span className="font-medium">{control.id}</span>
                            <span className={`
                              inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                              ${statusConfig[control.status].bg}
                              ${statusConfig[control.status].color}
                            `}>
                              <StatusIcon className="w-4 h-4 mr-1" />
                              {control.status}
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-300">
                            {control.title}
                          </p>
                          <div className="flex items-center space-x-4 text-sm text-gray-500">
                            <span>Assignee: {control.assignee}</span>
                            <span>Last Assessed: {new Date(control.lastAssessed).toLocaleDateString()}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      ))}
    </div>
  )
} 