'use client'

import { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ComplianceControlsList } from '@/components/Compliance/ComplianceControlsList'
import { ComplianceAuditsList } from '@/components/Compliance/ComplianceAuditsList'
import { ComplianceEvidence } from '@/components/Compliance/ComplianceEvidence'
import { ComplianceRemediation } from '@/components/Compliance/ComplianceRemediation'
import type { ComplianceFramework } from '@/types/compliance'

interface ComplianceDashboardProps {
  framework: ComplianceFramework
}

export function ComplianceDashboard({ framework }: ComplianceDashboardProps) {
  return (
    <Tabs defaultValue="controls" className="space-y-4">
      <TabsList>
        <TabsTrigger value="controls">Controls</TabsTrigger>
        <TabsTrigger value="audits">Audits</TabsTrigger>
        <TabsTrigger value="evidence">Evidence</TabsTrigger>
        <TabsTrigger value="remediation">Remediation</TabsTrigger>
      </TabsList>

      <TabsContent value="controls">
        <ComplianceControlsList framework={framework} />
      </TabsContent>

      <TabsContent value="audits">
        <ComplianceAuditsList framework={framework} />
      </TabsContent>

      <TabsContent value="evidence">
        <ComplianceEvidence framework={framework} />
      </TabsContent>

      <TabsContent value="remediation">
        <ComplianceRemediation framework={framework} />
      </TabsContent>
    </Tabs>
  )
} 