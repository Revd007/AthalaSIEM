import { PredictionDetail } from '@/types/predictive'

export const mockPredictionDetails: PredictionDetail[] = [
  {
    id: '1',
    type: 'Advanced Persistent Threat',
    severity: 'critical',
    probability: 89.5,
    impact: {
      financial: 85,
      operational: 75,
      reputational: 90
    },
    timeline: {
      detected: new Date(Date.now() - 3600000).toISOString(),
      estimated: new Date(Date.now() + 86400000).toISOString(),
      window: '24-48 hours'
    },
    source: {
      ip: '192.168.1.100',
      location: 'Eastern Europe',
      actor: 'APT-29',
      technique: 'Supply Chain Compromise'
    },
    affectedSystems: [
      {
        id: 'sys-1',
        name: 'Primary Database Server',
        type: 'Database',
        criticality: 'High',
        status: 'At Risk'
      },
      {
        id: 'sys-2',
        name: 'Authentication Server',
        type: 'Identity Management',
        criticality: 'Critical',
        status: 'Monitoring'
      }
    ],
    indicators: [
      {
        id: 'ind-1',
        type: 'Network Traffic',
        value: 'Unusual data exfiltration pattern',
        confidence: 92,
        firstSeen: new Date(Date.now() - 86400000).toISOString(),
        lastSeen: new Date().toISOString()
      },
      {
        id: 'ind-2',
        type: 'System Access',
        value: 'Privileged account creation',
        confidence: 88,
        firstSeen: new Date(Date.now() - 43200000).toISOString(),
        lastSeen: new Date().toISOString()
      }
    ],
    mitigationSteps: [
      {
        id: 'mit-1',
        action: 'Isolate affected systems',
        priority: 'high',
        status: 'pending',
        assignedTo: 'Security Team',
        eta: new Date(Date.now() + 3600000).toISOString()
      },
      {
        id: 'mit-2',
        action: 'Block suspicious IPs',
        priority: 'high',
        status: 'in-progress',
        assignedTo: 'Network Team',
        eta: new Date(Date.now() + 1800000).toISOString()
      }
    ],
    analysis: {
      summary: 'High-confidence detection of APT activity targeting critical infrastructure',
      methodology: 'ML-based pattern recognition with behavioral analysis',
      confidence: 92,
      falsePositiveRisk: 8,
      dataPoints: 15420,
      modelVersion: '2.1.0',
      lastUpdated: new Date().toISOString()
    },
    relatedEvents: [
      {
        id: 'evt-1',
        type: 'Failed Login',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        description: 'Multiple failed login attempts from suspicious IP'
      }
    ],
    recommendations: [
      {
        id: 'rec-1',
        type: 'immediate',
        description: 'Enable additional authentication factors',
        impact: 'High',
        effort: 'Medium',
        status: 'proposed'
      }
    ]
  },
  {
    id: '2',
    type: 'Ransomware Campaign',
    severity: 'high',
    probability: 75.8,
    impact: {
      financial: 90,
      operational: 85,
      reputational: 70
    },
    timeline: {
      detected: new Date(Date.now() - 7200000).toISOString(),
      estimated: new Date(Date.now() + 43200000).toISOString(),
      window: '12-24 hours'
    },
    source: {
      ip: '45.67.89.123',
      location: 'Unknown',
      technique: 'Phishing Campaign'
    },
    affectedSystems: [
      {
        id: 'sys-3',
        name: 'File Server',
        type: 'Storage',
        criticality: 'High',
        status: 'Vulnerable'
      }
    ],
    indicators: [
      {
        id: 'ind-3',
        type: 'File Activity',
        value: 'Mass file encryption attempts',
        confidence: 85,
        firstSeen: new Date(Date.now() - 14400000).toISOString(),
        lastSeen: new Date().toISOString()
      }
    ],
    mitigationSteps: [
      {
        id: 'mit-3',
        action: 'Backup critical data',
        priority: 'high',
        status: 'in-progress',
        assignedTo: 'IT Team',
        eta: new Date(Date.now() + 1800000).toISOString()
      }
    ],
    analysis: {
      summary: 'Potential ransomware attack targeting file storage systems',
      methodology: 'Behavioral analysis and IOC matching',
      confidence: 85,
      falsePositiveRisk: 15,
      dataPoints: 8750,
      modelVersion: '2.0.1',
      lastUpdated: new Date().toISOString()
    },
    relatedEvents: [
      {
        id: 'evt-2',
        type: 'Suspicious Email',
        timestamp: new Date(Date.now() - 28800000).toISOString(),
        description: 'Phishing email with malicious attachment'
      }
    ],
    recommendations: [
      {
        id: 'rec-2',
        type: 'immediate',
        description: 'Deploy ransomware protection',
        impact: 'High',
        effort: 'High',
        status: 'approved'
      }
    ]
  }
] 