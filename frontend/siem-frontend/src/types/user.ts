export interface User {
  id: string
  email: string
  name: string
  avatar?: string
  role: 'admin' | 'user' | 'analyst'
  createdAt: string
  updatedAt: string
  lastLogin?: string
  isActive: boolean
  permissions: string[]
  department?: string
  position?: string
}
