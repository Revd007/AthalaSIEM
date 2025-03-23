'use client'

import { ComplianceFrameworkSelector } from '@/components/Compliance/ComplianceFrameworkSelector'
import { ComplianceMetrics } from '@/components/Compliance/ComplianceMetrics'
import { ComplianceDashboard } from '@/components/Compliance/ComplianceDashboard'
import type { ComplianceFramework } from '@/types/compliance'
import { useState } from 'react'

export default function CompliancePage() {
  const [selectedFramework, setSelectedFramework] = useState<ComplianceFramework>('ISO27001')

  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Compliance Management
        </h1>
      </div>

      <ComplianceMetrics framework={selectedFramework} />
      
      <ComplianceFrameworkSelector
        selected={selectedFramework}
        onSelect={setSelectedFramework}
      />

      <ComplianceDashboard framework={selectedFramework} />
    </div>
  )
} 