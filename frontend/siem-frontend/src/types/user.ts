export type UserRole = 'admin' | 'analyst' | 'operator' | 'viewer';

export interface User {
  id: string;
  email: string;
  username: string;
  role: UserRole;
  full_name?: string;
  isActive: boolean;
  lastLogin?: string;
  createdAt: string;
  updatedAt: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  full_name?: string;
  role: UserRole;
}
