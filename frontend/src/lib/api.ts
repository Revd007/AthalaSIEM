import { QueryClient } from '@tanstack/react-query'
import type { ApiResponse } from '../types/api'

// Use environment variable with fallback
const baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5135'

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

const makeRequest = async <T>(url: string, options: RequestOptions = {}): Promise<ApiResponse<T>> => {
  const token = localStorage.getItem('token');
  const { skipAuth, ...restOptions } = options;

  const defaultOptions: RequestOptions = {
    mode: 'cors',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...(!skipAuth && token ? { 'Authorization': `Bearer ${token}` } : {}),
      ...options.headers,
    },
    ...restOptions,
  };

  try {
    const response = await fetch(`${baseURL}${url}`, defaultOptions);
    
    // Network error handling
    if (!response) {
      throw new Error('Network error - Failed to connect to the server');
    }

    // Handle 401 errors
    if (!skipAuth && response.status === 401) {
      // For installer endpoints, just throw an error without redirecting
      if (url.includes('/installer-info/') || url.includes('/installer/')) {
        throw new Error('Authentication required. Please ensure you are logged in.');
      }
      
      // For other endpoints, try to refresh the token or redirect to login
      if (!isRefreshingToken) {
        isRefreshingToken = true;
        
        try {
          // Try to silently refresh the token (you would need to implement this endpoint)
          const refreshResponse = await fetch(`${baseURL}/api/auth/refresh`, {
            method: 'POST',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json',
            }
          });
          
          if (refreshResponse.ok) {
            const refreshData = await refreshResponse.json();
            if (refreshData.token) {
              // Store the new token
              localStorage.setItem('token', refreshData.token);
              
              // Process pending requests
              processPendingRequests();
              
              // Retry the current request with the new token
              return makeRequest<T>(url, options);
            }
          }
          
          // If refresh failed, redirect to login
          localStorage.removeItem('token');
          window.location.href = '/login';
          throw new Error('Session expired. Please login again.');
        } catch (error) {
          // If refresh failed, redirect to login
          localStorage.removeItem('token');
          window.location.href = '/login';
          throw new Error('Session expired. Please login again.');
        } finally {
          isRefreshingToken = false;
        }
      } else {
        // If we're already refreshing the token, add this request to the queue
        return new Promise<ApiResponse<T>>(resolve => {
          pendingRequests.push(() => {
            resolve(makeRequest<T>(url, options));
          });
        });
      }
    }

    if (!response.ok) {
      try {
        const errorData = await response.json();
        const errorMessage = errorData.message || errorData.title || errorData.detail || `Request failed with status ${response.status}`;
        console.error('API Error:', errorData);
        throw new Error(errorMessage);
      } catch (e) {
        throw new Error(`Request failed with status ${response.status}`);
      }
    }

    if (options.responseType === 'blob') {
      return { data: await response.blob() } as ApiResponse<T>;
    }

    const data = await response.json();
    return { data } as ApiResponse<T>;
  } catch (error) {
    console.error('API Request failed:', error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to connect to the server. Please check your connection.');
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
  post: <T>(url: string, body?: any, options?: RequestOptions) => makeRequest<T>(url, { method: 'POST', body: JSON.stringify(body), ...options }),

  /**
   * Make a PUT request to the API
   * @param endpoint - The API endpoint to request
   * @param data - The data to send in the request body
   * @param options - Optional fetch options
   * @returns Promise with the response data
   */
  put: <T>(url: string, body?: any, options?: RequestOptions) => makeRequest<T>(url, { method: 'PUT', body: JSON.stringify(body), ...options }),

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
    me: '/api/auth/me'
  },
  agents: {
    list: '/api/agents',
    add: '/api/agents/add',
    details: (id: string) => `/api/agents/${id}`,
    update: (id: string) => `/api/agents/${id}`,
    delete: (id: string) => `/api/agents/${id}`,
    installer: (type: string) => `/api/agents/installer/${type}`
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