/**
 * Application port configuration
 * Using non-standard ports to improve security through obscurity
 */

export const PORTS = {
  /**
   * Frontend ports
   */
  FRONTEND: {
    DEVELOPMENT: 7654,    // For local development
    PRODUCTION: 7655,     // For production builds
    SECURE: 7656,         // For HTTPS production
    TEST: 7657            // For testing environment
  },
  
  /**
   * Backend ports
   */
  BACKEND: {
    HTTP: 9595,           // HTTP API endpoint
    HTTPS: 9596,          // HTTPS API endpoint
    GRPC: 50051           // gRPC services
  }
};

/**
 * Get the current frontend port based on environment
 */
export const getCurrentPort = (): number => {
  // Using direct environment variable checks to avoid circular dependencies
  if (process.env.NODE_ENV === 'production') {
    return process.env.SECURE_CONNECTION === 'true' 
      ? PORTS.FRONTEND.SECURE 
      : PORTS.FRONTEND.PRODUCTION;
  }
  
  if (process.env.NODE_ENV === 'test') {
    return PORTS.FRONTEND.TEST;
  }
  
  return PORTS.FRONTEND.DEVELOPMENT;
};

/**
 * Get the base URL for the backend API
 */
export const getBackendBaseUrl = (): string => {
  // Using direct environment variable checks to avoid circular dependencies
  const isSecure = process.env.NEXT_PUBLIC_USE_HTTPS === 'true';
  const protocol = isSecure ? 'https' : 'http';
  const port = isSecure ? PORTS.BACKEND.HTTPS : PORTS.BACKEND.HTTP;
  
  return `${protocol}://localhost:${port}`;
};

/**
 * Get the gRPC endpoint URL
 */
export const getGrpcEndpoint = (): string => {
  return `http://localhost:${PORTS.BACKEND.GRPC}`;
};

export default PORTS; 