export interface Activity {
    id: string;
    severity: 'low' | 'medium' | 'high';
    source: string;
    sourceIp: string;
    destinationIp?: string;
    user?: string;
    status: 'new' | 'in_progress' | 'resolved';
    category: string;
    timestamp: string;
    type: string;
    description: string;
  }