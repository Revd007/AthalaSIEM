export interface ThreatActor {
  id: string;
  name: string;
  aliases: string[];
  location: string;
  associatedEmails: string[];
  infrastructure: string[];
  riskLevel: 'low' | 'medium' | 'high';
  lastSeen: string;
  recentActivities?: string[];
  ttps?: string[]; // Tactics, Techniques, and Procedures
  indicators?: {
    ips: string[];
    domains: string[];
    hashes: string[];
  };
}