'use client'

import React, { useState } from 'react';
import { Users, Plus, Edit2, Trash2 } from 'lucide-react';
import { UserRole, Permission } from '../../types/auth';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Skeleton } from '@/components/ui/skeleton';

// Default permissions - these are system-defined
const systemPermissions: Permission[] = [
  {
    id: '1',
    name: 'view_dashboard',
    description: 'View main dashboard',
    roles: ['admin', 'analyst', 'auditor', 'operator'],
  },
  {
    id: '2',
    name: 'manage_users',
    description: 'Create and manage users',
    roles: ['admin'],
  },
  {
    id: '3',
    name: 'view_reports',
    description: 'View security reports',
    roles: ['admin', 'analyst', 'auditor'],
  },
  {
    id: '4',
    name: 'manage_alerts',
    description: 'Manage security alerts',
    roles: ['admin', 'analyst'],
  },
  {
    id: '5',
    name: 'view_logs',
    description: 'View system logs',
    roles: ['admin', 'analyst', 'auditor'],
  },
];

export function RoleManagement() {
  const [selectedRole, setSelectedRole] = useState<UserRole>('admin');

  // Fetch roles from backend
  const { data: rolesData, isLoading } = useQuery({
    queryKey: ['roles'],
    queryFn: async () => {
      try {
        const { data } = await api.get<{ roles: string[] }>('/api/auth/roles');
        return data?.roles || ['Admin', 'User', 'Analyst', 'Operator'];
      } catch {
        // Fallback to default roles
        return ['Admin', 'User', 'Analyst', 'Operator'];
      }
    }
  });

  const roles = rolesData || ['Admin', 'User', 'Analyst', 'Operator'];
  const permissions = systemPermissions;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Users className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Role Management</h2>
        </div>
        <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center">
          <Plus className="h-4 w-4 mr-2" />
          Create Role
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="col-span-1">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Roles</h3>
          <div className="space-y-2">
            {isLoading ? (
              <>
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </>
            ) : roles.map((role) => (
              <button
                key={role}
                onClick={() => setSelectedRole(role.toLowerCase() as UserRole)}
                className={`w-full text-left px-4 py-2 rounded-lg flex items-center justify-between ${
                  selectedRole === role.toLowerCase()
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                <span className="capitalize">{role}</span>
                <div className="flex space-x-2">
                  <Edit2 className="h-4 w-4 text-gray-400" />
                  <Trash2 className="h-4 w-4 text-gray-400" />
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="col-span-2">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Permissions</h3>
          <div className="space-y-4">
            {permissions.map((permission) => (
              <div key={permission.id} className="border dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white">{permission.name}</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">{permission.description}</p>
                  </div>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={permission.roles.includes(selectedRole)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                  </label>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}