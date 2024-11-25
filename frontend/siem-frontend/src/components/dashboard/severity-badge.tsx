interface SeverityBadgeProps {
    severity: string;
  }
  
  export function SeverityBadge({ severity }: SeverityBadgeProps) {
    const colors = {
      high: 'bg-red-100 text-red-800',
      medium: 'bg-yellow-100 text-yellow-800',
      low: 'bg-green-100 text-green-800',
      default: 'bg-gray-100 text-gray-800'
    }
  
    const badgeColor = colors[severity.toLowerCase() as keyof typeof colors] || colors.default
  
    return (
      <span className={`px-2 py-1 rounded-full text-sm font-medium ${badgeColor}`}>
        {severity}
      </span>
    )
  }