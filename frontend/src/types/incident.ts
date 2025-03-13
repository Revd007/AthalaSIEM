export interface TimelineEvent {
  id: string;
  time: string;
  action: string;
  user: string;
  details?: string;
  type: 'detection' | 'investigation' | 'response' | 'resolution';
}

export interface Incident {
  id: string;
  title: string;
  description: string;
  status: 'investigating' | 'containment' | 'eradication' | 'recovery' | 'resolved';
  priority: 'critical' | 'high' | 'medium' | 'low';
  category: 'security' | 'network' | 'system' | 'application';
  assignee: string;
  reporter: string;
  createdAt: string;
  updatedAt: string;
  timeline: TimelineEvent[];
  affectedSystems: string[];
  tags: string[];
  metrics: {
    mttr?: number; // Mean Time To Resolve
    mttd?: number; // Mean Time To Detect
    mtta?: number; // Mean Time To Acknowledge
  };
} 