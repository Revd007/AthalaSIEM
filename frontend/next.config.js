/** @type {import('next').NextConfig} */

// Load environment variables from .env.local
const { env } = require('./src/scripts/load-env');

const nextConfig = {
  reactStrictMode: true,
  
  // Pass environment variables to the client
  env: env,
  
  async rewrites() {
    // Ensure we have a valid API URL, falling back to the default HTTP port
    const apiUrl = env.NEXT_PUBLIC_API_URL || 'http://localhost:9595';
    
    return [
      {
        source: '/api/auth/:path*',
        destination: `${apiUrl}/auth/:path*`
      },
      {
        source: '/auth/:path*',
        destination: `${apiUrl}/auth/:path*`
      }
    ]
  },
  // Add CORS headers
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Credentials', value: 'true' },
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,DELETE,PATCH,POST,PUT' },
          { key: 'Access-Control-Allow-Headers', value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization' },
        ]
      }
    ]
  },
  // Security enhancement - disable X-Powered-By header
  poweredByHeader: false
}

module.exports = nextConfig