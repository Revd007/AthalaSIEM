/**
 * Environment configuration
 */

// Default values in case environment variables aren't set
const defaults = {
  NEXT_PUBLIC_API_URL: 'http://localhost:9598',
  NEXT_PUBLIC_USE_HTTPS: 'false',
  NEXT_PUBLIC_GRPC_URL: 'http://localhost:50051',
  SECURE_CONNECTION: 'false'
};

// Environment object to store the loaded variables
const env: Record<string, string> = {};

// Ensure all required variables are set by using defaults for any missing
Object.entries(defaults).forEach(([key, value]) => {
  env[key] = process.env[key] || value;
  process.env[key] = process.env[key] || value;
});

// Log the loaded environment variables
console.log('Environment variables loaded:');
console.log(`- NEXT_PUBLIC_API_URL: ${env.NEXT_PUBLIC_API_URL}`);
console.log(`- NEXT_PUBLIC_USE_HTTPS: ${env.NEXT_PUBLIC_USE_HTTPS}`);
console.log(`- NEXT_PUBLIC_GRPC_URL: ${env.NEXT_PUBLIC_GRPC_URL}`);
console.log(`- SECURE_CONNECTION: ${env.SECURE_CONNECTION}`);

export { env }; 