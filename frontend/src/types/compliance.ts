export type ComplianceFramework = 'ISO27001' | 'PCIDSS' | 'HIPAA' | 'GDPR' | 'SOC2';

export type ComplianceStatus = 'compliant' | 'non-compliant' | 'partial' | 'in-progress' | 'not-applicable';

export type ControlStatus = 'compliant' | 'non-compliant' | 'partial' | 'not-applicable'

export interface ComplianceControl {
  id: string;
  name: string;
  description: string;
  status: ControlStatus;
  framework: string;
  category: string;
  lastAssessed: string;
  evidence?: string[];
}

export interface ComplianceEvidence {
  id: string;
  controlId: string;
  type: 'document' | 'screenshot' | 'log' | 'report' | 'policy';
  name: string;
  url: string;
  uploadedAt: string;
  uploadedBy: string;
  description: string;
}

export interface ComplianceAudit {
  id: string;
  framework: ComplianceFramework;
  startDate: string;
  endDate: string;
  status: 'planned' | 'in-progress' | 'completed' | 'delayed';
  auditor: string;
  findings: ComplianceAuditFinding[];
  summary: string;
  score: number;
}

export interface ComplianceAuditFinding {
  id: string;
  controlId: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  recommendation: string;
  dueDate: string;
  status: 'open' | 'in-progress' | 'resolved' | 'accepted-risk';
}

export const statusConfig: Record<ControlStatus, { icon: any; color: string; bg: string }> = {
  'compliant': {
    icon: 'CheckCircle',
    color: 'text-green-600',
    bg: 'bg-green-50'
  },
  'non-compliant': {
    icon: 'XCircle',
    color: 'text-red-600',
    bg: 'bg-red-50'
  },
  'partial': {
    icon: 'AlertCircle',
    color: 'text-yellow-600',
    bg: 'bg-yellow-50'
  },
  'not-applicable': {
    icon: 'MinusCircle',
    color: 'text-gray-600',
    bg: 'bg-gray-50'
  }
} 