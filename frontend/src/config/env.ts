import { PORTS } from './ports';

/**
 * Environment configuration
 * Centralizes access to environment variables with fallbacks
 */

export const ENV = {
  /**
   * API URL for backend communication
   */
  API_URL: process.env.NEXT_PUBLIC_API_URL || `http://localhost:${PORTS.BACKEND.HTTP}`,
  
  /**
   * Whether to use HTTPS for API communication
   */
  USE_HTTPS: process.env.NEXT_PUBLIC_USE_HTTPS === 'true',
  
  /**
   * gRPC URL for streaming services
   */
  GRPC_URL: process.env.NEXT_PUBLIC_GRPC_URL || `http://localhost:${PORTS.BACKEND.GRPC}`,
  
  /**
   * Whether to use secure connection for frontend
   */
  SECURE_CONNECTION: process.env.SECURE_CONNECTION === 'true',
  
  /**
   * Current environment
   */
  NODE_ENV: process.env.NODE_ENV || 'development',
  
  /**
   * Determines if we're in a production environment
   */
  IS_PRODUCTION: process.env.NODE_ENV === 'production',
  
  /**
   * Determines if we're in a development environment
   */
  IS_DEVELOPMENT: process.env.NODE_ENV === 'development' || !process.env.NODE_ENV,
  
  /**
   * Determines if we're in a test environment
   */
  IS_TEST: process.env.NODE_ENV === 'test',
};

export default ENV; 