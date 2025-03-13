'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { Play, Pause, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function ActivePlaybooksTab() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4">
        <DashboardCard>
          <div className="flex justify-between items-center">
            <div>
              <h3 className="font-medium">Malware Detection Response</h3>
              <p className="text-sm text-gray-500">Automatically responds to malware alerts</p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm">
                <Pause className="w-4 h-4 mr-2" />
                Pause
              </Button>
              <Button variant="outline" size="sm">
                <Settings className="w-4 h-4 mr-2" />
                Configure
              </Button>
            </div>
          </div>
        </DashboardCard>
        {/* Add more active playbooks */}
      </div>
    </div>
  )
} 