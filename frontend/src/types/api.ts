export interface ApiResponse<T = any> {
  data: T;
  error?: string;
  message?: string;
  status: number;
  statusText: string;
  headers: Headers;
  token?: string;
  token_type?: string;
} 