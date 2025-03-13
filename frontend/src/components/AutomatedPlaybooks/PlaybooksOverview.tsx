'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { 
  Play, 
  Pause, 
  AlertTriangle, 
  Check, 
  Clock, 
  Activity,
  Settings, 
  Plus, 
  Edit, 
  Trash2, 
  Copy,
  FileCode,
  Workflow,
  GitBranch,
  Zap,
  Shield,
  Database,
  Mail,
  MessageSquare,
  Server,
  Lock,
  RefreshCw
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { PlaybooksOverviewTab } from './PlaybooksOverviewTab'
import { ActivePlaybooksTab } from './ActivePlaybooksTab'
import { PlaybookBuilder } from './PlaybookBuilder'
import { PlaybookExecutions } from './PlaybookExecutions'
import { PlaybookTemplates } from './PlaybookTemplates'
import { PlaybookSettings } from './PlaybookSettings'
import { PlaybookCreationModal } from './PlaybookCreationModal'

interface Playbook {
  id: string
  name: string
  description: string
  trigger: {
    type: 'alert' | 'schedule' | 'manual' | 'api'
    condition: string
  }
  status: 'active' | 'inactive' | 'draft'
  steps: PlaybookStep[]
  lastRun?: string
  successRate: number
  totalRuns: number
  averageRuntime: number
  category: 'incident' | 'threat' | 'compliance' | 'remediation'
  priority: 'critical' | 'high' | 'medium' | 'low'
  tags: string[]
  owner: string
  created: string
  modified: string
}

interface PlaybookStep {
  id: string
  type: 'action' | 'condition' | 'delay' | 'notification'
  name: string
  description: string
  config: any
  status: 'pending' | 'running' | 'completed' | 'failed'
  duration?: number
  result?: any
}

interface PlaybookExecution {
  id: string
  playbookId: string
  startTime: string
  endTime?: string
  status: 'running' | 'completed' | 'failed'
  steps: {
    stepId: string
    status: 'pending' | 'running' | 'completed' | 'failed'
    startTime: string
    endTime?: string
    output?: any
  }[]
  trigger: {
    type: string
    details: any
  }
}

export function PlaybooksOverview() {
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedPlaybook, setSelectedPlaybook] = useState<Playbook | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Automated Playbooks</h1>
        <Button onClick={() => setIsCreating(true)}>
          <Plus className="w-4 h-4 mr-2" />
          Create Playbook
        </Button>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList>
          <TabsTrigger value="overview">
            <Activity className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="active">
            <Play className="w-4 h-4 mr-2" />
            Active Playbooks
          </TabsTrigger>
          <TabsTrigger value="builder">
            <FileCode className="w-4 h-4 mr-2" />
            Playbook Builder
          </TabsTrigger>
          <TabsTrigger value="executions">
            <Clock className="w-4 h-4 mr-2" />
            Executions
          </TabsTrigger>
          <TabsTrigger value="templates">
            <Copy className="w-4 h-4 mr-2" />
            Templates
          </TabsTrigger>
          <TabsTrigger value="settings">
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </TabsTrigger>
        </TabsList>

        <div className="mt-6">
          <TabsContent value="overview">
            <PlaybooksOverviewTab />
          </TabsContent>

          <TabsContent value="active">
            <ActivePlaybooksTab />
          </TabsContent>

          <TabsContent value="builder">
            <PlaybookBuilder />
          </TabsContent>

          <TabsContent value="executions">
            <PlaybookExecutions />
          </TabsContent>

          <TabsContent value="templates">
            <PlaybookTemplates />
          </TabsContent>

          <TabsContent value="settings">
            <PlaybookSettings />
          </TabsContent>
        </div>
      </Tabs>

      {isCreating && (
        <PlaybookCreationModal onClose={() => setIsCreating(false)} />
      )}
    </div>
  )
} 