const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9595';

interface LoginCredentials {
  username: string;
  password: string;
}

interface RegisterCredentials {
  username: string;
  email: string;
  full_name?: string;
  role?: string;
  password: string;
}

interface User {
  id: string;
  email: string;
  username: string;
  role: string;
  full_name?: string;
}

export interface LoginResponse {
  token: string;
  refreshToken: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    username: string;
    role: string;
    full_name?: string;
  };
}

export const authService = {
  async login(credentials: LoginCredentials): Promise<LoginResponse> {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
      }

      const data = await response.json();
      
      // Store both tokens in localStorage
      localStorage.setItem('token', data.token);
      localStorage.setItem('refreshToken', data.refreshToken);
      
      // Also store token in cookie for server-side auth
      document.cookie = `token=${data.token}; path=/;`;
      
      return data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },

  async register(credentials: RegisterCredentials) {
    const response = await fetch('/api/auth/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({
        ...credentials,
        is_active: true
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Registration failed');
    }

    return response.json();
  },

  async getCurrentUser(): Promise<User> {
    const token = localStorage.getItem('token');
    if (!token) throw new Error('No token found');

    try {
      const response = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        credentials: 'include',
      });

      if (!response.ok) {
        if (response.status === 401) {
          // Try to refresh token
          try {
            const refreshResponse = await fetch('/api/auth/refresh', {
              method: 'POST',
              credentials: 'include',
              headers: {
                'Content-Type': 'application/json',
              }
            });

            if (refreshResponse.ok) {
              const refreshData = await refreshResponse.json();
              if (refreshData.token) {
                localStorage.setItem('token', refreshData.token);
                localStorage.setItem('refreshToken', refreshData.refreshToken);
                document.cookie = `token=${refreshData.token}; path=/;`;
                
                // Retry the request with new token
                return this.getCurrentUser();
              }
            }
          } catch (refreshError) {
            console.error('Token refresh failed:', refreshError);
          }
          
          // If refresh failed, clear tokens and redirect
          this.logout();
          throw new Error('Session expired');
        }
        throw new Error('Failed to get current user');
      }

      const data = await response.json();
      return {
        ...data,
        id: String(data.id)
      };
    } catch (error) {
      throw error;
    }
  },

  async logout() {
    const token = localStorage.getItem('token');
    if (!token) return;

    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
    } finally {
      // Clear both localStorage and cookie
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
      document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
    }
  }
};