import { z } from 'zod';
import { User } from '../types/auth';

export const AuditEventSchema = z.object({
  id: z.string().uuid(),
  timestamp: z.string().datetime(),
  actor: z.object({
    id: z.string(),
    username: z.string(),
    role: z.string(),
  }),
  action: z.string(),
  resource: z.string(),
  details: z.record(z.unknown()),
  status: z.enum(['success', 'failure']),
  ipAddress: z.string(),
});

export type AuditEvent = z.infer<typeof AuditEventSchema>;

class AuditLogger {
  private static instance: AuditLogger;
  private readonly logs: AuditEvent[] = [];

  private constructor() {}

  static getInstance(): AuditLogger {
    if (!AuditLogger.instance) {
      AuditLogger.instance = new AuditLogger();
    }
    return AuditLogger.instance;
  }

  log(user: User, action: string, resource: string, details: Record<string, unknown>, status: 'success' | 'failure', ipAddress: string) {
    const event: AuditEvent = {
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      actor: {
        id: user.id,
        username: user.username,
        role: user.role,
      },
      action,
      resource,
      details,
      status,
      ipAddress,
    };

    this.logs.push(event);
    console.log('Audit Event:', event); // In production, send to secure logging service
  }

  getAuditTrail(filters?: Partial<AuditEvent>): AuditEvent[] {
    if (!filters) return this.logs;

    return this.logs.filter(log => 
      Object.entries(filters).every(([key, value]) => 
        log[key as keyof AuditEvent] === value
      )
    );
  }
}

export const auditLogger = AuditLogger.getInstance();