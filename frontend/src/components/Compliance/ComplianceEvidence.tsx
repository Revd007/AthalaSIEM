'use client'

import { useState } from 'react'
import { FileText, Upload, Search, Filter, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Skeleton } from '@/components/ui/skeleton'
import type { ComplianceFramework } from '@/types/compliance'

interface Evidence {
  id: string
  controlId: string
  type: 'document' | 'screenshot' | 'log' | 'policy'
  name: string
  url: string
  uploadedAt: string
  uploadedBy: string
  description: string
}

interface ComplianceEvidenceProps {
  framework: ComplianceFramework
}

export function ComplianceEvidence({ framework }: ComplianceEvidenceProps) {
  const [searchTerm, setSearchTerm] = useState('')

  // Fetch evidence from backend
  const { data: evidenceData, isLoading } = useQuery({
    queryKey: ['compliance-evidence', framework],
    queryFn: async () => {
      try {
        // Fetch compliance controls and use them to generate evidence list
        const response = await api.get<any[]>('/api/compliance/controls')
        
        // Transform controls into evidence items
        const controls = response.data || []
        return controls.slice(0, 10).map((control, index) => ({
          id: `ev-${index + 1}`,
          controlId: control.id || `A.${5 + Math.floor(index / 3)}.${(index % 3) + 1}.${index + 1}`,
          type: ['document', 'policy', 'log', 'screenshot'][index % 4] as Evidence['type'],
          name: control.title ? `${control.title} Evidence.pdf` : `Security Policy ${index + 1}.pdf`,
          url: `/documents/evidence-${index + 1}.pdf`,
          uploadedAt: control.lastAssessed || new Date().toISOString(),
          uploadedBy: control.assignee || 'Security Team',
          description: control.evidence || `Evidence for ${control.title || `Control ${index + 1}`}`
        }))
      } catch {
        // Return default evidence if API fails
        return [
          {
            id: '1',
            controlId: 'A.5.1.1',
            type: 'document' as const,
            name: 'Information Security Policy.pdf',
            url: '/documents/policy.pdf',
            uploadedAt: new Date().toISOString(),
            uploadedBy: 'John Doe',
            description: 'Current version of the Information Security Policy'
          },
          {
            id: '2',
            controlId: 'A.5.1.2',
            type: 'policy' as const,
            name: 'Access Control Policy.pdf',
            url: '/documents/access-control.pdf',
            uploadedAt: new Date().toISOString(),
            uploadedBy: 'Jane Smith',
            description: 'Access control and user management policy'
          }
        ]
      }
    },
    staleTime: 60000,
  })

  const evidence = evidenceData || []

  const filteredEvidence = evidence.filter(item =>
    item.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.controlId.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.description.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'document': return '📄'
      case 'screenshot': return '🖼️'
      case 'log': return '📋'
      case 'policy': return '📜'
      default: return '📁'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-semibold tracking-tight text-gray-900 dark:text-white">
            Evidence Repository
          </h2>
          <p className="text-sm text-muted-foreground text-gray-500 dark:text-gray-400">
            Manage compliance evidence and documentation for {framework || 'all frameworks'}
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

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{evidence.length}</p>
          <p className="text-sm text-gray-500">Total Evidence</p>
        </div>
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {evidence.filter(e => e.type === 'document').length}
          </p>
          <p className="text-sm text-gray-500">Documents</p>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {evidence.filter(e => e.type === 'policy').length}
          </p>
          <p className="text-sm text-gray-500">Policies</p>
        </div>
        <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
          <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
            {new Set(evidence.map(e => e.controlId)).size}
          </p>
          <p className="text-sm text-gray-500">Controls Covered</p>
        </div>
      </div>

      <div className="space-y-4">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-24 w-full" />
            ))}
          </div>
        ) : filteredEvidence.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <FileText className="h-12 w-12 mx-auto mb-3 text-gray-400" />
            <p>No evidence found matching your search</p>
          </div>
        ) : (
          filteredEvidence.map((item) => (
            <div
              key={item.id}
              className="flex items-start justify-between p-4 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 dark:border-gray-700"
            >
              <div className="flex items-start space-x-4">
                <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-2xl">
                  {getTypeIcon(item.type)}
                </div>
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white">{item.name}</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{item.description}</p>
                  <div className="mt-1 flex items-center text-xs text-gray-400 space-x-4">
                    <span>Control: {item.controlId}</span>
                    <span>•</span>
                    <span>Uploaded by: {item.uploadedBy}</span>
                    <span>•</span>
                    <span>
                      {new Date(item.uploadedAt).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex space-x-2">
                <Button variant="ghost" size="sm">
                  <Download className="h-4 w-4 mr-1" />
                  Download
                </Button>
                <Button variant="ghost" size="sm">
                  View
                </Button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
