'use client'

import { useState } from 'react'
import { AttackerProfileCard } from '@/components/CounterDefense/AttackerProfileCard'
import { CounterMeasurePanel } from '@/components/CounterDefense/CounterMeasurePanel'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { AttackerProfile } from '@/types/counter-defense'

// Placeholder: replace with useQuery when backend exposes GET /api/counter-defense/profiles
const MOCK_PROFILES: AttackerProfile[] = []

export default function CounterDefensePage() {
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null)
  const profiles = MOCK_PROFILES
  const selectedProfile = profiles.find((p) => p.id === selectedProfileId)

  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Counter-Defense & Active Defense
        </h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Active attacks</CardTitle>
          <p className="text-sm text-muted-foreground">
            Real-time panel when AI confirms attack (confidence &gt; 0.85). Wire to
            GET /api/counter-defense/profiles and WS /ws/counter-defense.
          </p>
        </CardHeader>
        <CardContent>
          {profiles.length === 0 ? (
            <p className="text-muted-foreground text-sm">
              No active attacker profiles. Profiles appear when attacks are confirmed
              and backend creates them via POST /api/counter-defense/profile.
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {profiles.map((profile) => (
                <AttackerProfileCard
                  key={profile.id}
                  profile={profile}
                  onAction={() => setSelectedProfileId(profile.id)}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Attacker profiles</CardTitle>
              <p className="text-sm text-muted-foreground">
                Geolocation map and infrastructure graph require backend attacker_profiles
                and optional map component (e.g. Leaflet).
              </p>
            </CardHeader>
            <CardContent>
              {selectedProfile ? (
                <AttackerProfileCard
                  profile={selectedProfile}
                  onAction={(action) => {
                    /* POST /api/counter-defense/block-ip etc. */
                  }}
                />
              ) : (
                <p className="text-muted-foreground text-sm">
                  Select an attacker above or wait for live data.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
        <div>
          <CounterMeasurePanel
            attackerId={selectedProfileId ?? undefined}
            onExecute={(action) => {
              console.log('Counter measure:', action)
              // POST to /api/counter-defense/block-ip, honeypot-redirect, etc.
            }}
          />
        </div>
      </div>
    </div>
  )
}
