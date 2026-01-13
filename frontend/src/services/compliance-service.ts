import { api } from '@/lib/api';

export interface ComplianceControl {
  id: string;
  title: string;
  status: 'compliant' | 'non-compliant' | 'in-progress';
  lastAssessed: string;
  nextAssessment: string;
  evidence: string[];
  assignee: string;
  framework: string;
  section: string;
}

export interface ComplianceAudit {
  id: string;
  title: string;
  status: 'completed' | 'in-progress' | 'scheduled';
  startDate: string;
  endDate: string;
  auditor: string;
  score?: number;
  findings: number;
  framework: string;
}

export interface ComplianceMetrics {
  overallCompliance: number;
  controlsAtRisk: number;
  pendingReviews: number;
  nextAuditDate?: string;
  totalControls: number;
  compliantControls: number;
  nonCompliantControls: number;
}

export const complianceService = {
  async getControls(framework: string): Promise<ComplianceControl[]> {
    try {
      const { data } = await api.get<ComplianceControl[]>(`/api/compliance/${framework}/controls`);
      return data ?? [];
    } catch (error) {
      console.error('Error fetching compliance controls:', error);
      return [];
    }
  },

  async getAudits(framework: string): Promise<ComplianceAudit[]> {
    try {
      const { data } = await api.get<ComplianceAudit[]>(`/api/compliance/${framework}/audits`);
      return data ?? [];
    } catch (error) {
      console.error('Error fetching compliance audits:', error);
      return [];
    }
  },

  async getMetrics(framework: string): Promise<ComplianceMetrics | null> {
    try {
      const { data } = await api.get<ComplianceMetrics>(`/api/compliance/${framework}/metrics`);
      return data ?? null;
    } catch (error) {
      console.error('Error fetching compliance metrics:', error);
      return null;
    }
  }
};
