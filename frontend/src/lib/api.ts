import { QueryClient } from '@tanstack/react-query'
import type { ApiResponse } from '../types/api'
import { env } from '../config/env'

// Use environment configuration for base URL
const baseURL = env.NEXT_PUBLIC_API_URL

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
})

interface RequestOptions extends RequestInit {
  responseType?: 'json' | 'blob';
  skipAuth?: boolean;
}

// Track if we're currently refreshing the token
let isRefreshingToken = false;
// Store pending requests that are waiting for token refresh
let pendingRequests: Array<() => void> = [];

// Function to process pending requests after token refresh
const processPendingRequests = () => {
  pendingRequests.forEach(callback => callback());
  pendingRequests = [];
};

const refreshToken = async () => {
  try {
    const refreshToken = localStorage.getItem('refreshToken');
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${baseURL}/api/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({ refreshToken })
    });

    if (!response.ok) {
      throw new Error('Failed to refresh token');
    }

    const data = await response.json();
    if (data.token) {
      localStorage.setItem('token', data.token);
      localStorage.setItem('refreshToken', data.refreshToken);
      return data.token;
    }
    throw new Error('No token in refresh response');
  } catch (error) {
    console.error('Token refresh failed:', error);
    throw error;
  }
};

const makeRequest = async <T>(url: string, options: RequestOptions = {}): Promise<ApiResponse<T>> => {
  // Get token fresh from localStorage each time
  const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
  const { skipAuth, responseType, ...restOptions } = options;

  // Debug logging for token
  if (!skipAuth && typeof window !== 'undefined') {
    if (!token && !window.location.pathname.includes('/login')) {
      console.warn('[API] Making authenticated request without token:', url);
       // BREAKPOINT: Making API call without token
    } else if (token) {
      console.log(`[API] Making request to ${url}, Token present: true, Token length: ${token.length}`);
      console.log(`[API] Token preview: ${token.substring(0, 30)}...`);
       // BREAKPOINT: Making API call with token
    }
  }

  const defaultOptions: RequestInit = {
    mode: 'cors',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'Accept': responseType === 'blob' ? '*/*' : 'application/json',
      ...(!skipAuth && token ? { 'Authorization': `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
    ...restOptions,
  };

  try {
    const response = await fetch(`${baseURL}${url}`, defaultOptions);
    
    // Debug: Log response status
    if (!skipAuth && typeof window !== 'undefined') {
      console.log(`[API] Response from ${url}: ${response.status} ${response.statusText}`);
      if (response.status === 401) {
        console.error(`[API] 401 Unauthorized for ${url} - Check if token is being sent correctly`);
      }
    }

    if (!response) {
      throw new Error('Network error - Failed to connect to the server');
    }

    // Handle 401 errors
    if (!skipAuth && response.status === 401) {
       // BREAKPOINT: Received 401 Unauthorized response
      
      if (url.includes('/auth/login') || url.includes('/auth/register')) {
        throw new Error('Authentication failed');
      }
      
      // Check if we have a token - if not, user is not logged in
      const existingToken = localStorage.getItem('token');
      const existingRefreshToken = localStorage.getItem('refreshToken');
      
      console.warn('[API] 401 Unauthorized for:', url, 'Token exists:', !!existingToken, 'RefreshToken exists:', !!existingRefreshToken);
       // BREAKPOINT: After checking for tokens on 401
      
      if (!existingToken || !existingRefreshToken) {
        // No tokens - user is not logged in, clear everything and throw
        console.warn('[API] No tokens found, redirecting to login');
         // BREAKPOINT: About to redirect (no tokens on 401)
        localStorage.removeItem('token');
        localStorage.removeItem('refreshToken');
        queryClient.clear();
        // Only redirect if we're not already on the login page
        // Use a flag to prevent multiple redirects
        if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
          // Check if we're already redirecting to prevent loop
          const redirectingKey = '__athala_redirecting_to_login';
          if (!sessionStorage.getItem(redirectingKey)) {
            sessionStorage.setItem(redirectingKey, 'true');
            console.log('[API] Redirecting to login from:', window.location.pathname);
            // Clear the flag after a delay
            setTimeout(() => sessionStorage.removeItem(redirectingKey), 1000);
            window.location.href = '/login';
          }
        }
        throw new Error('Not authenticated');
      }
      
      if (!isRefreshingToken) {
        isRefreshingToken = true;
        
        try {
          const newToken = await refreshToken();
          
          // Process pending requests
          processPendingRequests();
          
          // Retry the current request with the new token
          return makeRequest<T>(url, {
            ...options,
            headers: {
              ...defaultOptions.headers,
              'Authorization': `Bearer ${newToken}`,
            },
          });
        } catch (error) {
          // If refresh failed, clear both tokens and redirect to login
          localStorage.removeItem('token');
          localStorage.removeItem('refreshToken');
          queryClient.clear();
          // Only redirect if we're not already on the login page
          if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
            window.location.href = '/login';
          }
          throw new Error('Session expired');
        } finally {
          isRefreshingToken = false;
        }
      }
      
      // If we're already refreshing the token, add this request to the queue
      return new Promise<ApiResponse<T>>(resolve => {
        pendingRequests.push(() => {
          resolve(makeRequest<T>(url, options));
        });
      });
    }

    if (!response.ok) {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        const errorData = await response.json();
        throw new Error(errorData.message || `Request failed with status ${response.status}`);
      }
      throw new Error(`Request failed with status ${response.status}`);
    }

    if (responseType === 'blob') {
      const blob = await response.blob();
      return { data: blob } as ApiResponse<T>;
    }

    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      const data = await response.json();
      return { data } as ApiResponse<T>;
    }

    const text = await response.text();
    return { data: text as any } as ApiResponse<T>;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to connect to the server');
  }
};

/**
 * Generic API client for making HTTP requests to the backend
 */
export const api = {
  /**
   * Make a GET request to the API
   * @param endpoint - The API endpoint to request
   * @param options - Optional fetch options
   * @returns Promise with the response data
   */
  get: <T>(url: string, options?: RequestOptions) => makeRequest<T>(url, { method: 'GET', ...options }),

  /**
   * Make a POST request to the API
   * @param endpoint - The API endpoint to request
   * @param data - The data to send in the request body
   * @param options - Optional fetch options
   * @returns Promise with the response data
   */
  post: <T>(url: string, body?: any, options?: RequestOptions) => makeRequest<T>(url, { 
    method: 'POST', 
    body: body ? JSON.stringify(body) : undefined,
    ...options 
  }),

  /**
   * Make a PUT request to the API
   * @param endpoint - The API endpoint to request
   * @param data - The data to send in the request body
   * @param options - Optional fetch options
   * @returns Promise with the response data
   */
  put: <T>(url: string, body?: any, options?: RequestOptions) => makeRequest<T>(url, { 
    method: 'PUT', 
    body: body ? JSON.stringify(body) : undefined,
    ...options 
  }),

  /**
   * Make a DELETE request to the API
   * @param endpoint - The API endpoint to request
   * @param options - Optional fetch options
   * @returns Promise with the response data
   */
  delete: <T>(url: string, options?: RequestOptions) => makeRequest<T>(url, { method: 'DELETE', ...options }),

  /**
   * Make a PATCH request to the API
   * @param endpoint - The API endpoint to request
   * @param data - The data to send in the request body
   * @param options - Optional fetch options
   * @returns Promise with the response data
   */
  patch: <T>(url: string, body?: any, options?: RequestOptions) => makeRequest<T>(url, { method: 'PATCH', body: JSON.stringify(body), ...options }),
};

// API endpoints configuration
export const endpoints = {
  auth: {
    login: '/api/auth/login',
    logout: '/api/auth/logout',
    register: '/api/auth/register',
    me: '/api/auth/me',
    refresh: '/api/auth/refresh'
  },
  agents: {
    list: '/api/agents',
    add: '/api/agents/add',
    details: (id: string) => `/api/agents/${id}`,
    update: (id: string) => `/api/agents/${id}`,
    delete: (id: string) => `/api/agents/${id}`,
    download: (os: string) => `/api/agents/download/${os}`
  },
  events: {
    list: '/api/logs',
    details: (id: string) => `/api/logs/${id}`
  },
  dashboard: {
    overview: '/api/dashboard',
    metrics: '/api/dashboard/metrics'
  },
  ai: {
    analyze: '/api/ai/analyze',
    status: '/api/ai/status'
  },
  health: '/api/health'
};

export const login = async (username: string, password: string) => {
  try {
    const response = await fetch(`${baseURL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
      credentials: 'include'
    });

    if (!response.ok) {
      throw new Error('Login failed');
    }

    const data = await response.json();
    if (data.token && data.refreshToken) {
      localStorage.setItem('token', data.token);
      localStorage.setItem('refreshToken', data.refreshToken);
      return data;
    }
    throw new Error('Invalid login response - missing token or refresh token');
  } catch (error) {
    console.error('Login failed:', error);
    throw error;
  }
};

export const logout = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('refreshToken');
  queryClient.clear();
  window.location.href = '/login';
}; 