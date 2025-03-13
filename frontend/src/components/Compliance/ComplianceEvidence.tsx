'use client'

import { useState } from 'react'
import { FileText, Upload, Search, Filter } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { ComplianceEvidence as Evidence, ComplianceFramework } from '@/types/compliance'

interface ComplianceEvidenceProps {
  framework: ComplianceFramework
}

export function ComplianceEvidence({ framework }: ComplianceEvidenceProps) {
  const [searchTerm, setSearchTerm] = useState('')

  // Mock data - replace with API call
  const evidence: Evidence[] = [
    {
      id: '1',
      controlId: 'A.5.1.1',
      type: 'document',
      name: 'Information Security Policy.pdf',
      url: '/documents/policy.pdf',
      uploadedAt: '2024-02-15T10:30:00Z',
      uploadedBy: 'John Doe',
      description: 'Current version of the Information Security Policy'
    },
    // Add more evidence...
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-semibold tracking-tight">
            Evidence Repository
          </h2>
          <p className="text-sm text-muted-foreground">
            Manage compliance evidence and documentation
          </p>
        </div>
        <Button>
          <Upload className="h-4 w-4 mr-2" />
          Upload Evidence
        </Button>
      </div>

      <div className="flex space-x-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search evidence..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <Button variant="outline">
          <Filter className="h-4 w-4 mr-2" />
          Filter
        </Button>
      </div>

      <div className="space-y-4">
        {evidence.map((item) => (
          <div
            key={item.id}
            className="flex items-start justify-between p-4 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            <div className="flex items-start space-x-4">
              <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <FileText className="h-6 w-6 text-blue-500" />
              </div>
              <div>
                <h3 className="font-medium">{item.name}</h3>
                <p className="text-sm text-gray-500">{item.description}</p>
                <div className="mt-1 flex items-center text-xs text-gray-400 space-x-4">
                  <span>Control: {item.controlId}</span>
                  <span>Uploaded by: {item.uploadedBy}</span>
                  <span>
                    {new Date(item.uploadedAt).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>
            <Button variant="ghost" size="sm">
              View
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
} 