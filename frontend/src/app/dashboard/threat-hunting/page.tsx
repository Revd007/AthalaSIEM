'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { ThreatHuntingDashboard } from '@/components/ThreatHunting/ThreatHuntingDashboard'
import { IOCScanner } from '@/components/ThreatHunting/IOCScanner'
import { BehaviorAnalysis } from '@/components/ThreatHunting/BehaviorAnalysis'
import { YARARules } from '@/components/ThreatHunting/YARARules'
import { SIGMARules } from '@/components/ThreatHunting/SIGMARules'
import { ThreatIntelligence } from '@/components/ThreatHunting/ThreatIntelligence'
import { HuntingPlaybooks } from '@/components/ThreatHunting/HuntingPlaybooks'
import { LiveHunting } from '@/components/ThreatHunting/LiveHunting'

export default function ThreatHuntingPage() {
  const [activeTab, setActiveTab] = useState('dashboard')
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
          Threat Hunting
        </h1>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'dashboard', name: 'Dashboard' },
            { id: 'ioc-scanner', name: 'IOC Scanner' },
            { id: 'behavior', name: 'Behavior Analysis' },
            { id: 'yara', name: 'YARA Rules' },
            { id: 'sigma', name: 'SIGMA Rules' },
            { id: 'intel', name: 'Threat Intel' },
            { id: 'playbooks', name: 'Hunting Playbooks' },
            { id: 'live', name: 'Live Hunting' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                border-b-2 py-4 px-1 text-sm font-medium
                ${activeTab === tab.id
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }
              `}
            >
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="mt-6">
        {activeTab === 'dashboard' && <ThreatHuntingDashboard />}
        {activeTab === 'ioc-scanner' && <IOCScanner />}
        {activeTab === 'behavior' && <BehaviorAnalysis />}
        {activeTab === 'yara' && <YARARules />}
        {activeTab === 'sigma' && <SIGMARules />}
        {activeTab === 'intel' && <ThreatIntelligence />}
        {activeTab === 'playbooks' && <HuntingPlaybooks />}
        {activeTab === 'live' && <LiveHunting />}
      </div>
    </div>
  )
} 