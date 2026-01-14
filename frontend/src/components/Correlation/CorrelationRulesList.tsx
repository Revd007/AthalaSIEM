'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { type CorrelationRule } from '@/hooks/useCorrelation'
import { Skeleton } from '@/components/ui/skeleton'
import { Shield, Clock, Target } from 'lucide-react'

interface CorrelationRulesListProps {
  rules: CorrelationRule[]
  isLoading: boolean
}

export function CorrelationRulesList({ rules, isLoading }: CorrelationRulesListProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
    )
  }

  if (rules.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No correlation rules configured
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {rules.map((rule, index) => (
        <Card key={index}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">{rule.name}</CardTitle>
              </div>
              <Badge variant="outline">Active</Badge>
            </div>
            <CardDescription>{rule.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Threshold:</span>
                <Badge variant="secondary">{rule.threshold} events</Badge>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Time Window:</span>
                <Badge variant="secondary">{rule.timeWindow} minutes</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
