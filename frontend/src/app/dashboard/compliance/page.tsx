'use client'

import { ComplianceFrameworkSelector } from '@/components/Compliance/ComplianceFrameworkSelector'
import { ComplianceMetrics } from '@/components/Compliance/ComplianceMetrics'
import { ComplianceDashboard } from '@/components/Compliance/ComplianceDashboard'
import type { ComplianceFramework } from '@/types/compliance'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function CompliancePage() {
  const [selectedFramework, setSelectedFramework] = useState<ComplianceFramework>('ISO27001')
  const router = useRouter()

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    if (!token) {
      router.push('/login')
    }
  }, [router])

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