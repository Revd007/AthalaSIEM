import { AES, enc } from 'crypto-js';
import { z } from 'zod';

// Security constants
const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY || 'default-key-replace-in-production';
const MAX_LOGIN_ATTEMPTS = 3;
const PASSWORD_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$/;

// Input validation schemas
export const LogEventSchema = z.object({
  id: z.string().uuid(),
  timestamp: z.string().datetime(),
  severity: z.enum(['low', 'medium', 'high', 'critical']),
  source: z.string().min(1),
  message: z.string().min(1),
  sourceIp: z.string().ip().optional(),
  destinationIp: z.string().ip().optional(),
});

// Security utility functions
export const sanitizeInput = (input: string): string => {
  return input.replace(/[<>]/g, ''); // Basic XSS prevention
};

export const encryptData = (data: string): string => {
  return AES.encrypt(data, ENCRYPTION_KEY).toString();
};

export const decryptData = (encryptedData: string): string => {
  const bytes = AES.decrypt(encryptedData, ENCRYPTION_KEY);
  return bytes.toString(enc.Utf8);
};

export const validatePassword = (password: string): boolean => {
  return PASSWORD_REGEX.test(password);
};

// CSRF token management
export const generateCSRFToken = (): string => {
  return Math.random().toString(36).substring(2);
};

// Rate limiting
const requestCounts = new Map<string, number>();
const requestTimestamps = new Map<string, number>();

export const checkRateLimit = (ip: string): boolean => {
  const now = Date.now();
  const windowMs = 15 * 60 * 1000; // 15 minutes
  const maxRequests = 100;

  const count = requestCounts.get(ip) || 0;
  const lastTimestamp = requestTimestamps.get(ip) || 0;

  if (now - lastTimestamp > windowMs) {
    requestCounts.set(ip, 1);
    requestTimestamps.set(ip, now);
    return true;
  }

  if (count >= maxRequests) {
    return false;
  }

  requestCounts.set(ip, count + 1);
  return true;
};