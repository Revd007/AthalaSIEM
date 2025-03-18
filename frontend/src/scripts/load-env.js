/**
 * Helper script to ensure environment variables are properly loaded
 * This addresses issues with Next.js rewrites and environment variables
 */

const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

// Path to .env.local file
const envLocalPath = path.resolve(process.cwd(), '.env.local');

// Default values in case environment variables aren't set
const defaults = {
  NEXT_PUBLIC_API_URL: 'http://localhost:9595',
  NEXT_PUBLIC_USE_HTTPS: 'false',
  NEXT_PUBLIC_GRPC_URL: 'http://localhost:50051',
  SECURE_CONNECTION: 'false'
};

// Environment object to store the loaded variables
const env = {};

// Try to load from .env.local file
if (fs.existsSync(envLocalPath)) {
  console.log('Loading environment variables from .env.local');
  
  try {
    // Parse the file content
    const fileContent = fs.readFileSync(envLocalPath, 'utf8');
    const parsed = dotenv.parse(fileContent);
    
    // Copy values to env object and process.env
    Object.entries(parsed).forEach(([key, value]) => {
      env[key] = value;
      if (!process.env[key]) {
        process.env[key] = value;
      }
    });
  } catch (error) {
    console.error('Error loading .env.local:', error.message);
  }
}

// Ensure all required variables are set by using defaults for any missing
Object.entries(defaults).forEach(([key, value]) => {
  if (!env[key]) {
    env[key] = process.env[key] || value;
    process.env[key] = process.env[key] || value;
  }
});

// Log the loaded environment variables
console.log('Environment variables loaded:');
console.log(`- NEXT_PUBLIC_API_URL: ${env.NEXT_PUBLIC_API_URL}`);
console.log(`- NEXT_PUBLIC_USE_HTTPS: ${env.NEXT_PUBLIC_USE_HTTPS}`);
console.log(`- NEXT_PUBLIC_GRPC_URL: ${env.NEXT_PUBLIC_GRPC_URL}`);
console.log(`- SECURE_CONNECTION: ${env.SECURE_CONNECTION}`);

module.exports = { env }; 