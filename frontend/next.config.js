/** @type {import('next').NextConfig} */

// Load environment variables
const { env } = require('./src/config/env');

const nextConfig = {
  reactStrictMode: true,
  
  // Pass environment variables to the client
  env: env,
  
  async rewrites() {
    // Ensure we have a valid API URL, falling back to the default HTTP port
    const apiUrl = env.NEXT_PUBLIC_API_URL || 'http://localhost:9595';
    
    return [
      // Auth endpoints with correct casing
      {
        source: '/api/auth/:path*',
        destination: `${apiUrl}/api/Auth/:path*`
      },
      {
        source: '/api/Auth/:path*',
        destination: `${apiUrl}/api/Auth/:path*`
      },
      // Additional API routes
      {
        source: '/api/agents/:path*',
        destination: `${apiUrl}/api/Agents/:path*`
      },
      {
        source: '/api/alerts/:path*',
        destination: `${apiUrl}/api/Alerts/:path*`
      },
      {
        source: '/api/logs/:path*',
        destination: `${apiUrl}/api/Logs/:path*`
      },
      {
        source: '/api/users/:path*',
        destination: `${apiUrl}/api/Users/:path*`
      },
      {
        source: '/api/dashboard/:path*',
        destination: `${apiUrl}/api/Dashboard/:path*`
      },
      {
        source: '/api/reports/:path*',
        destination: `${apiUrl}/api/Reports/:path*`
      },
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`
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
          { key: 'Access-Control-Allow-Methods', value: 'GET,DELETE,PATCH,POST,PUT,OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization' },
        ]
      }
    ]
  },
  // Security enhancement - disable X-Powered-By header
  poweredByHeader: false
}

module.exports = nextConfig