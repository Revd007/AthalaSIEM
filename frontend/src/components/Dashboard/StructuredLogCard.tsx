'use client'

import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ChevronDown, ChevronUp, AlertTriangle, Shield, Network, FileText, User, Activity } from 'lucide-react'
import type { LogEntry } from '@/types/agent'

interface StructuredLogCardProps {
  log: LogEntry
  onExpand?: () => void
}

const getEventIcon = (eventType?: string, source?: string) => {
  const type = (eventType || source || '').toLowerCase()
  if (type.includes('auth') || type.includes('logon')) return <Shield className="h-4 w-4" />
  if (type.includes('network') || type.includes('connection')) return <Network className="h-4 w-4" />
  if (type.includes('file') || type.includes('fim')) return <FileText className="h-4 w-4" />
  if (type.includes('process')) return <Activity className="h-4 w-4" />
  return <AlertTriangle className="h-4 w-4" />
}

const getSeverityColor = (severity?: string, level?: string) => {
  const sev = (severity || level || 'low').toLowerCase()
  if (sev === 'critical' || sev === 'high') return 'destructive'
  if (sev === 'medium' || sev === 'warning') return 'default'
  return 'secondary'
}

const getSeverityText = (severity?: string, level?: string) => {
  return severity || level || 'Low'
}

export function StructuredLogCard({ log, onExpand }: StructuredLogCardProps) {
  const [expanded, setExpanded] = useState(false)
  
  const normalizedFields = (log.properties || {}) as any
  const sourceIp = normalizedFields?.sourceIp || normalizedFields?.source_ip || log.ipAddress
  const destIp = normalizedFields?.destinationIp || normalizedFields?.destination_ip
  const sourcePort = normalizedFields?.sourcePort || normalizedFields?.source_port
  const destPort = normalizedFields?.destinationPort || normalizedFields?.destination_port
  const userName = normalizedFields?.userName || normalizedFields?.user_name || log.username
  const processName = normalizedFields?.processName || normalizedFields?.process_name || log.processName
  const eventType = normalizedFields?.eventType || normalizedFields?.event_type || log.category
  const mitreTechniques = normalizedFields?.mitre_techniques || []
  
  const shortMessage = log.message?.length > 100 
    ? log.message.substring(0, 100) + '...' 
    : log.message || '(no message)'

  const handleExpand = () => {
    setExpanded(!expanded)
    onExpand?.()
  }

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          {/* Icon */}
          <div className="flex-shrink-0 mt-1">
            {getEventIcon(eventType, log.source)}
          </div>

          {/* Main Content */}
          <div className="flex-1 min-w-0">
            {/* Header Row */}
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <Badge variant="outline" className="text-xs">
                {log.source || 'Unknown'}
              </Badge>
              <Badge 
                variant={getSeverityColor(log.severity, log.level) as any}
                className="text-xs"
              >
                {getSeverityText(log.severity, log.level)}
              </Badge>
              {eventType && (
                <Badge variant="secondary" className="text-xs">
                  {eventType}
                </Badge>
              )}
              {mitreTechniques.length > 0 && (
                <Badge variant="destructive" className="text-xs">
                  MITRE: {mitreTechniques[0]?.techniqueId || 'N/A'}
                </Badge>
              )}
            </div>

            {/* Short Description */}
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              {shortMessage}
            </p>

            {/* Key Fields Row */}
            <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400 flex-wrap">
              {sourceIp && (
                <div className="flex items-center gap-1">
                  <Network className="h-3 w-3" />
                  <span>Src: {sourceIp}</span>
                  {sourcePort && <span>:{sourcePort}</span>}
                </div>
              )}
              {destIp && (
                <div className="flex items-center gap-1">
                  <span>→</span>
                  <span>Dst: {destIp}</span>
                  {destPort && <span>:{destPort}</span>}
                </div>
              )}
              {userName && (
                <div className="flex items-center gap-1">
                  <User className="h-3 w-3" />
                  <span>{userName}</span>
                </div>
              )}
              {processName && (
                <div className="flex items-center gap-1">
                  <Activity className="h-3 w-3" />
                  <span className="truncate max-w-[150px]" title={processName}>
                    {processName.split('\\').pop()?.split('/').pop()}
                  </span>
                </div>
              )}
              {log.eventId && log.eventId > 0 && (
                <span>Event ID: {log.eventId}</span>
              )}
            </div>

            {/* Expanded Details */}
            {expanded && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="space-y-2 text-xs">
                  <div>
                    <span className="font-semibold">Full Message:</span>
                    <p className="text-gray-600 dark:text-gray-400 mt-1 whitespace-pre-wrap">
                      {log.message || '(no message)'}
                    </p>
                  </div>
                  
                  {mitreTechniques.length > 0 && (
                    <div>
                      <span className="font-semibold">MITRE ATT&CK Techniques:</span>
                      <div className="mt-1 space-y-1">
                        {mitreTechniques.map((tech: any, idx: number) => (
                          <div key={idx} className="text-gray-600 dark:text-gray-400">
                            <Badge variant="outline" className="mr-2">
                              {tech.techniqueId}
                            </Badge>
                            {tech.techniqueName} - {tech.tactic}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {normalizedFields && Object.keys(normalizedFields).length > 0 && (
                    <div>
                      <span className="font-semibold">Structured Fields:</span>
                      <pre className="mt-1 p-2 bg-gray-50 dark:bg-gray-900 rounded text-xs overflow-x-auto">
                        {JSON.stringify(normalizedFields, null, 2)}
                      </pre>
                    </div>
                  )}

                  <div className="text-gray-500 dark:text-gray-500">
                    <span>Agent ID: {log.agentId}</span>
                    <span className="mx-2">•</span>
                    <span>Timestamp: {new Date(log.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Expand/Collapse Button */}
            <Button
              variant="ghost"
              size="sm"
              className="mt-2 h-6 px-2 text-xs"
              onClick={handleExpand}
            >
              {expanded ? (
                <>
                  <ChevronUp className="h-3 w-3 mr-1" />
                  Collapse
                </>
              ) : (
                <>
                  <ChevronDown className="h-3 w-3 mr-1" />
                  Expand
                </>
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
