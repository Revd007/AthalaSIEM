'use client'

import { Card } from '@/components/ui/card'
import { CheckCircle, AlertTriangle, Calendar } from 'lucide-react'
import type { ComplianceFramework } from '@/types/compliance'

interface ComplianceAuditsListProps {
  framework: ComplianceFramework
}

type AuditStatus = 'completed' | 'in-progress' | 'scheduled'

interface Audit {
  id: string
  title: string
  status: AuditStatus
  startDate: string
  endDate: string
  auditor: string
  score?: number
  findings: number
}

const mockAudits: Audit[] = [
  {
    id: '1',
    title: 'Annual ISO 27001 Certification',
    status: 'completed',
    startDate: '2024-01-15',
    endDate: '2024-02-15',
    auditor: 'External Auditor Inc.',
    score: 92,
    findings: 3
  },
  {
    id: '2',
    title: 'Q1 Internal Audit',
    status: 'in-progress',
    startDate: '2024-03-01',
    endDate: '2024-03-31',
    auditor: 'Internal Audit Team',
    findings: 5
  }
]

const statusConfig: Record<AuditStatus, { icon: any; color: string; bg: string }> = {
  completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50' },
  'in-progress': { icon: AlertTriangle, color: 'text-yellow-500', bg: 'bg-yellow-50' },
  scheduled: { icon: Calendar, color: 'text-blue-500', bg: 'bg-blue-50' }
}

export function ComplianceAuditsList({ framework }: ComplianceAuditsListProps) {
  return (
    <div className="space-y-4">
      {mockAudits.map((audit) => {
        const StatusIcon = statusConfig[audit.status].icon
        return (
          <Card key={audit.id} className="p-6">
            <div className="flex items-start justify-between">
              <div className="space-y-3">
                <div className="space-y-1">
                  <h3 className="text-lg font-medium">{audit.title}</h3>
                  <div className="flex items-center space-x-2">
                    <span className={`
                      inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                      ${statusConfig[audit.status].bg}
                      ${statusConfig[audit.status].color}
                    `}>
                      <StatusIcon className="w-4 h-4 mr-1" />
                      {audit.status}
                    </span>
                    {audit.score && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700">
                        Score: {audit.score}%
                      </span>
                    )}
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700">
                      Findings: {audit.findings}
                    </span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm text-gray-500">
                  <div>
                    <span className="block font-medium text-gray-900">Start Date</span>
                    {new Date(audit.startDate).toLocaleDateString()}
                  </div>
                  <div>
                    <span className="block font-medium text-gray-900">End Date</span>
                    {new Date(audit.endDate).toLocaleDateString()}
                  </div>
                  <div>
                    <span className="block font-medium text-gray-900">Auditor</span>
                    {audit.auditor}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        )
      })}
    </div>
  )
} 