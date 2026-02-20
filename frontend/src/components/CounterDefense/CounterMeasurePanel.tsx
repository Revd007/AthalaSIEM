'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Shield, Bug, Lock, Mail, FileWarning, Zap } from 'lucide-react'
import type { CounterMeasureType } from '@/types/counter-defense'

interface CounterMeasurePanelProps {
  attackerId?: string
  onExecute?: (action: CounterMeasureType) => void
  disabled?: boolean
}

const MEASURES: { id: CounterMeasureType; label: string; icon: React.ElementType }[] = [
  { id: 'block_ip', label: 'Block IP (firewall)', icon: Shield },
  { id: 'honeypot_redirect', label: 'Redirect to Honeypot', icon: Bug },
  { id: 'tarpit', label: 'Enable Tarpit', icon: Zap },
  { id: 'lock_accounts', label: 'Lock Targeted Accounts', icon: Lock },
  { id: 'gather_intel', label: 'Gather Intelligence', icon: FileWarning },
  { id: 'deploy_deception', label: 'Deploy Deception', icon: Mail },
  { id: 'execute_playbook', label: 'Execute Playbook', icon: Zap },
]

export function CounterMeasurePanel({
  attackerId,
  onExecute,
  disabled = false,
}: CounterMeasurePanelProps) {
  const [confirmAction, setConfirmAction] = useState<CounterMeasureType | null>(null)

  const handleClick = (action: CounterMeasureType) => {
    const offensive = ['honeypot_redirect', 'tarpit', 'gather_intel', 'deploy_deception']
    if (offensive.includes(action)) {
      setConfirmAction(action)
      return
    }
    onExecute?.(action)
  }

  const confirm = () => {
    if (confirmAction) {
      onExecute?.(confirmAction)
      setConfirmAction(null)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Counter-measures</CardTitle>
        <p className="text-sm text-muted-foreground">
          Require confirmation for offensive actions. All actions are logged.
        </p>
      </CardHeader>
      <CardContent className="space-y-2">
        {MEASURES.map(({ id, label, icon: Icon }) => (
          <Button
            key={id}
            variant="outline"
            className="w-full justify-start gap-2"
            disabled={disabled || (!attackerId && id !== 'execute_playbook')}
            onClick={() => handleClick(id)}
          >
            <Icon className="h-4 w-4" />
            {label}
          </Button>
        ))}
        {confirmAction && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <Card className="max-w-sm">
              <CardHeader>
                <CardTitle>Confirm action</CardTitle>
                <p className="text-sm text-muted-foreground">
                  This action may have legal implications. Proceed only if authorized.
                </p>
              </CardHeader>
              <CardContent className="flex gap-2">
                <Button onClick={confirm}>Confirm</Button>
                <Button variant="outline" onClick={() => setConfirmAction(null)}>
                  Cancel
                </Button>
              </CardContent>
            </Card>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
