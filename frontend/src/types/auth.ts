export type UserRole = 'admin' | 'analyst' | 'auditor' | 'operator';

export interface User {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  permissions: string[];
  lastLogin: string;
  twoFactorEnabled: boolean;
}

export interface Permission {
  id: string;
  name: string;
  description: string;
  roles: UserRole[];
}