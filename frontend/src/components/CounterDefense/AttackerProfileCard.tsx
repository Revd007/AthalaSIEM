'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { AttackerProfile } from '@/types/counter-defense'

interface AttackerProfileCardProps {
  profile: AttackerProfile
  onAction?: (action: string) => void
}

function countryToFlag(iso?: string): string {
  if (!iso || iso.length !== 2) return '🌐'
  const codePoints = iso.toUpperCase().split('').map((c) => 0x1f1e6 - 65 + c.charCodeAt(0))
  return String.fromCodePoint(...codePoints)
}

export function AttackerProfileCard({ profile, onAction }: AttackerProfileCardProps) {
  const geo = profile.geolocation
  const countryCode = geo?.country_code ?? ''
  const flag = countryToFlag(countryCode)
  const location = [geo?.city, geo?.country].filter(Boolean).join(', ') || 'Unknown'
  const riskPercent = Math.round((profile.risk_score ?? 0) * 100)

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <span>{flag}</span>
            <span>{profile.ip_addresses?.[0] ?? 'Unknown IP'}</span>
          </CardTitle>
          <Badge variant={riskPercent >= 70 ? 'destructive' : riskPercent >= 40 ? 'default' : 'secondary'}>
            Risk {riskPercent}%
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-muted-foreground">
          <strong>Location:</strong> {location}
        </p>
        {profile.asn_info?.org && (
          <p className="text-muted-foreground">
            <strong>ISP/Org:</strong> {profile.asn_info.org}
          </p>
        )}
        <p className="text-muted-foreground">
          <strong>First seen:</strong>{' '}
          {profile.first_seen ? new Date(profile.first_seen).toLocaleString() : '—'}
        </p>
        <p className="text-muted-foreground">
          <strong>Last seen:</strong>{' '}
          {profile.last_seen ? new Date(profile.last_seen).toLocaleString() : '—'}
        </p>
        <p className="text-muted-foreground">
          <strong>Attack count:</strong> {profile.attack_count ?? 0}
        </p>
        {profile.counter_measures_applied && profile.counter_measures_applied.length > 0 && (
          <p className="text-muted-foreground">
            <strong>Measures applied:</strong>{' '}
            {profile.counter_measures_applied.join(', ')}
          </p>
        )}
        {onAction && (
          <div className="pt-2 flex flex-wrap gap-2">
            <button
              type="button"
              className="text-xs px-2 py-1 rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-900/50"
              onClick={() => onAction('block_ip')}
            >
              Block IP
            </button>
            <button
              type="button"
              className="text-xs px-2 py-1 rounded bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-900/50"
              onClick={() => onAction('gather_intel')}
            >
              Gather intel
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
