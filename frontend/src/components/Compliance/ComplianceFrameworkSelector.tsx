'use client'

import { Shield, FileCheck, Activity, Lock, CheckCircle } from 'lucide-react'
import type { ComplianceFramework } from '@/types/compliance'

interface ComplianceFrameworkSelectorProps {
  selected: ComplianceFramework
  onSelect: (framework: ComplianceFramework) => void
}

const frameworks = [
  {
    id: 'ISO27001',
    name: 'ISO 27001',
    description: 'Information Security Management',
    icon: Shield,
  },
  {
    id: 'PCIDSS',
    name: 'PCI DSS',
    description: 'Payment Card Industry Data Security Standard',
    icon: Lock,
  },
  {
    id: 'HIPAA',
    name: 'HIPAA',
    description: 'Health Insurance Portability and Accountability Act',
    icon: Activity,
  },
  {
    id: 'GDPR',
    name: 'GDPR',
    description: 'General Data Protection Regulation',
    icon: FileCheck,
  },
  {
    id: 'SOC2',
    name: 'SOC 2',
    description: 'Service Organization Control 2',
    icon: CheckCircle,
  },
] as const

export function ComplianceFrameworkSelector({ selected, onSelect }: ComplianceFrameworkSelectorProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
      {frameworks.map(({ id, name, description, icon: Icon }) => (
        <button
          key={id}
          onClick={() => onSelect(id as ComplianceFramework)}
          className={`
            p-4 rounded-lg border transition-colors
            ${selected === id 
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
              : 'border-gray-200 dark:border-gray-700 hover:border-blue-500 hover:bg-gray-50 dark:hover:bg-gray-800'
            }
          `}
        >
          <div className="flex flex-col items-center text-center space-y-2">
            <Icon className={`h-8 w-8 ${selected === id ? 'text-blue-500' : 'text-gray-500'}`} />
            <div>
              <h3 className="font-medium">{name}</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">{description}</p>
            </div>
          </div>
        </button>
      ))}
    </div>
  )
} 