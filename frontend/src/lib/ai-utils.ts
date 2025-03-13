export function getPatternSeverityColor(severity: number) {
    if (severity > 0.8) return 'bg-red-500'
    if (severity > 0.6) return 'bg-orange-500'
    if (severity > 0.4) return 'bg-yellow-500'
    return 'bg-green-500'
  }
  
  export function getThreatSeverityVariant(severity: number) {
    if (severity > 0.8) return 'destructive'
    if (severity > 0.6) return 'warning'
    return 'secondary'
  }