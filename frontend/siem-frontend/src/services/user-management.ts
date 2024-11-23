import { axiosInstance } from '../lib/axios';

interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  last_login: string;
  groups: Group[];
}

interface Group {
  id: string;
  name: string;
  description: string;
  users: User[];
}

export const userManagementService = {
  // User Operations
  async getUsers(): Promise<User[]> {
    const response = await axiosInstance.get('/users');
    return response.data;
  },

  async createUser(userData: Partial<User>): Promise<User> {
    const response = await axiosInstance.post('/users', userData);
    return response.data;
  },

  // Group Operations
  async getGroups(): Promise<Group[]> {
    const response = await axiosInstance.get('/groups');
    return response.data;
  },

  async addUserToGroup(userId: string, groupId: string): Promise<void> {
    await axiosInstance.post(`/groups/${groupId}/users`, { user_id: userId });
  }
};