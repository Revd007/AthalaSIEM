'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { AIThreatAnalyzer } from '@/components/AI/SecurityAnalysis/AIThreatAnalyzer'
import { AnomalyDetection } from '@/components/AI/SecurityAnalysis/AnomalyDetection'
import { BehavioralAnalytics } from '@/components/AI/SecurityAnalysis/BehavioralAnalytics'
import { PredictiveAnalysis } from '@/components/AI/SecurityAnalysis/PredictiveAnalysis'
import { AutomatedResponse } from '@/components/AI/SecurityAnalysis/AutomatedResponse'
import { AIInsights } from '@/components/AI/SecurityAnalysis/AIInsights'
import { OSINTAnalysis } from '@/components/OSINT/OSINTAnalysis'
import { OSINTCorrelation } from '@/components/OSINT/OSINTCorrelation'
import { Brain, Activity, AlertTriangle, Shield, Zap, LineChart, Globe } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

export default function AIAnalysisPage() {
  const router = useRouter()

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    if (!token) {
      router.push('/login')
    }
  }, [router])

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold tracking-tight">AI Security Analysis</h1>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">
            <Brain className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="anomalies">
            <AlertTriangle className="w-4 h-4 mr-2" />
            Anomaly Detection
          </TabsTrigger>
          <TabsTrigger value="behavior">
            <Activity className="w-4 h-4 mr-2" />
            Behavioral Analytics
          </TabsTrigger>
          <TabsTrigger value="predictive">
            <LineChart className="w-4 h-4 mr-2" />
            Predictive Analysis
          </TabsTrigger>
          <TabsTrigger value="response">
            <Zap className="w-4 h-4 mr-2" />
            Automated Response
          </TabsTrigger>
          <TabsTrigger value="osint">
            <Globe className="w-4 h-4 mr-2" />
            OSINT Analysis
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <AIThreatAnalyzer />
            <AIInsights />
            <OSINTCorrelation />
          </div>
        </TabsContent>

        {/* Anomaly Detection Tab */}
        <TabsContent value="anomalies">
          <AnomalyDetection />
        </TabsContent>

        {/* Behavioral Analytics Tab */}
        <TabsContent value="behavior">
          <BehavioralAnalytics />
        </TabsContent>

        {/* Predictive Analysis Tab */}
        <TabsContent value="predictive">
          <PredictiveAnalysis />
        </TabsContent>

        {/* Automated Response Tab */}
        <TabsContent value="response">
          <AutomatedResponse />
        </TabsContent>

        {/* OSINT Analysis Tab */}
        <TabsContent value="osint">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <OSINTAnalysis />
            <OSINTCorrelation />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
} 