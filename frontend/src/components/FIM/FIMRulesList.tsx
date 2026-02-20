'use client'

import { useFIMRules } from '@/services/fim-service'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

export function FIMRulesList() {
  const { data: rules, isLoading, isError } = useFIMRules()

  if (isLoading) {
    return (
      <Card>
        <CardHeader><CardTitle>FIM Rules</CardTitle></CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (isError || !rules) {
    return (
      <Card>
        <CardHeader><CardTitle>FIM Rules</CardTitle></CardHeader>
        <CardContent className="text-red-600 dark:text-red-400">
          Failed to load rules.
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>FIM Rules</CardTitle>
        <p className="text-sm text-muted-foreground">
          Monitored paths and scan settings. Configure via backend or future rule editor.
        </p>
      </CardHeader>
      <CardContent>
        {rules.length === 0 ? (
          <p className="text-muted-foreground text-sm">No FIM rules defined.</p>
        ) : (
          <ul className="space-y-3">
            {rules.map((r) => (
              <li
                key={r.id}
                className="flex flex-wrap items-center gap-2 rounded border dark:border-gray-700 p-3"
              >
                <span className="font-medium">{r.name}</span>
                <Badge variant={r.isEnabled ? 'default' : 'secondary'}>
                  {r.isEnabled ? 'Enabled' : 'Disabled'}
                </Badge>
                <span className="text-muted-foreground text-sm">
                  {r.monitoredPaths}
                </span>
                {r.targetAgents && (
                  <span className="text-xs text-muted-foreground">
                    Agents: {r.targetAgents}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}
