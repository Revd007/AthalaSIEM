import { QueryClient } from '@tanstack/react-query'
import type { ApiResponse } from '../types/api'
import { ENV } from '../config/env'

// Use environment configuration for base URL
const baseURL = ENV.API_URL

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

  // For authentication endpoints, use specific CORS settings to avoid preflight issues
  const isAuthEndpoint = url.includes('/auth/login') || url.includes('/auth/register');
  
  const defaultOptions: RequestOptions = {
    mode: 'cors',
    // Always include credentials to ensure cookies are sent
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
    // Log the request for debugging
    console.debug(`Making API request to: ${baseURL}${url}`, {
      method: defaultOptions.method || 'GET',
      headers: defaultOptions.headers,
      credentials: defaultOptions.credentials,
    });
    
    const response = await fetch(`${baseURL}${url}`, defaultOptions);
    
    // Network error handling
    if (!response) {
      throw new Error('Network error - Failed to connect to the server');
    }

    console.debug(`Response status: ${response.status} ${response.statusText}`);

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
        // Check if response can be parsed as JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json();
          const errorMessage = errorData.message || errorData.title || errorData.detail || `Request failed with status ${response.status}`;
          console.error('API Error:', errorData);
          throw new Error(errorMessage);
        } else {
          // Handle non-JSON responses
          const text = await response.text();
          console.error('API Error (non-JSON):', text);
          throw new Error(`Request failed with status ${response.status}: ${text.substring(0, 100)}`);
        }
      } catch (e) {
        console.error('Error parsing error response:', e);
        throw new Error(`Request failed with status ${response.status}`);
      }
    }

    if (options.responseType === 'blob') {
      return { data: await response.blob() } as ApiResponse<T>;
    }

    try {
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        return { data } as ApiResponse<T>;
      } else {
        // Handle non-JSON responses
        const text = await response.text();
        console.warn('Received non-JSON response:', text.substring(0, 100));
        return { data: text as any } as ApiResponse<T>;
      }
    } catch (e) {
      console.error('Error parsing success response:', e);
      throw new Error('Failed to parse response data');
    }
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