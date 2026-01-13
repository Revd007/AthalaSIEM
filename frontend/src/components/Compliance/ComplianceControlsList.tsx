'use client'

import { Card } from '@/components/ui/card'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible'
import { ChevronRight, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { complianceService, type ComplianceControl } from '@/services/compliance-service'
import { Skeleton } from '@/components/ui/skeleton'
import type { ComplianceFramework } from '@/types/compliance'

interface ComplianceControlsListProps {
  framework: ComplianceFramework
}

type ControlStatus = 'compliant' | 'non-compliant' | 'in-progress'

const statusConfig: Record<ControlStatus, { icon: any; color: string; bg: string }> = {
  compliant: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50' },
  'non-compliant': { icon: XCircle, color: 'text-red-500', bg: 'bg-red-50' },
  'in-progress': { icon: AlertTriangle, color: 'text-yellow-500', bg: 'bg-yellow-50' }
}

export function ComplianceControlsList({ framework }: ComplianceControlsListProps) {
  const { data: controls, isLoading } = useQuery({
    queryKey: ['compliance-controls', framework],
    queryFn: () => complianceService.getControls(framework),
    refetchInterval: 300000, // 5 minutes
  });

  // Group controls by section
  const controlsBySection = controls?.reduce((acc, control) => {
    const section = control.section || 'Other';
    if (!acc[section]) {
      acc[section] = [];
    }
    acc[section].push(control);
    return acc;
  }, {} as Record<string, ComplianceControl[]>) ?? {};

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-32 w-full" />
        ))}
      </div>
    );
  }

  if (Object.keys(controlsBySection).length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        No compliance controls configured for {framework}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {Object.entries(controlsBySection).map(([section, sectionControls]) => (
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
                {sectionControls.map((control) => {
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
                            <span>Assignee: {control.assignee || 'Unassigned'}</span>
                            <span>Last Assessed: {control.lastAssessed ? new Date(control.lastAssessed).toLocaleDateString() : 'Never'}</span>
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