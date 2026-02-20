'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useFIMRules } from '@/services/fim-service'
import { Skeleton } from '@/components/ui/skeleton'

/**
 * Baseline management UI. The .NET backend (FileIntegrityController) exposes
 * rules with MonitoredPaths and ExcludePatterns; it does not yet expose
 * a dedicated baselines API or re-baseline trigger. This component shows
 * monitored paths from rules as the effective "baseline" configuration.
 */
export function FIMBaselineManager() {
  const { data: rules, isLoading, isError } = useFIMRules()

  if (isLoading) {
    return (
      <Card>
        <CardHeader><CardTitle>Baseline & monitored paths</CardTitle></CardHeader>
        <CardContent>
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (isError || !rules) {
    return (
      <Card>
        <CardHeader><CardTitle>Baseline & monitored paths</CardTitle></CardHeader>
        <CardContent className="text-red-600 dark:text-red-400 text-sm">
          Failed to load rules.
        </CardContent>
      </Card>
    )
  }

  const enabledRules = rules.filter((r) => r.isEnabled)
  const pathsByRule = enabledRules.map((r) => ({
    name: r.name,
    paths: r.monitoredPaths?.split(/[\n,;]/).map((p) => p.trim()).filter(Boolean) ?? [],
    exclude: r.excludePatterns,
    targetAgents: r.targetAgents,
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>Baseline & monitored paths</CardTitle>
        <p className="text-sm text-muted-foreground">
          Monitored directories are defined in FIM rules. Re-baseline and
          per-agent baseline APIs require backend support.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {pathsByRule.length === 0 ? (
          <p className="text-muted-foreground text-sm">No enabled rules with paths.</p>
        ) : (
          pathsByRule.map(({ name, paths, exclude, targetAgents }) => (
            <div
              key={name}
              className="rounded border dark:border-gray-700 p-3 text-sm space-y-1"
            >
              <p className="font-medium">{name}</p>
              <ul className="list-disc list-inside text-muted-foreground">
                {paths.map((path, i) => (
                  <li key={i}>{path}</li>
                ))}
              </ul>
              {exclude && (
                <p className="text-xs text-muted-foreground">Exclude: {exclude}</p>
              )}
              {targetAgents && (
                <p className="text-xs text-muted-foreground">Agents: {targetAgents}</p>
              )}
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
