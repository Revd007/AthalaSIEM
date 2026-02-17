# 🛡️ AthalaSIEM Frontend

**Modern, production-ready SIEM dashboard built with Next.js**

[![Next.js](https://img.shields.io/badge/Next.js-15+-black)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## 📖 Overview

The AthalaSIEM Frontend is a modern, responsive web application built with Next.js 15+ that provides a comprehensive dashboard for security information and event management. It offers real-time monitoring, agent management, alert handling, and security analytics.

### Key Capabilities

- **Real-time Dashboard**: Live system metrics and security posture
- **Agent Management**: Deploy, configure, and monitor agents
- **Security Monitoring**: Real-time event monitoring and alerting
- **User Management**: Role-based access control and user administration
- **Analytics & Reporting**: Security analytics and compliance reporting
- **Responsive Design**: Optimized for desktop, tablet, and mobile

## ✨ Features

### Dashboard & Monitoring
-  Real-time system health metrics
-  Agent status monitoring
-  Log volume visualization
-  Security posture scoring
-  Recent alerts and incidents
-  Performance metrics

### Agent Management
-  Agent deployment and registration
-  Configuration management
-  Health monitoring
-  Log collection status
-  Agent metrics and statistics

### Security Monitoring
-  Real-time event monitoring
-  Alert management and triage
-  Threat detection visualization
-  Incident response workflows
-  Compliance monitoring

### User Management
-  User administration (Admin)
-  Role management
-  Password policy enforcement
-  Two-factor authentication (2FA)
-  Session management
-  User hardening settings

### Analytics & Reporting
-  Custom security reports
-  Data visualization with charts
-  Trend analysis
-  Compliance reporting
-  Performance analytics

## 🛠️ Tech Stack

### Core Framework
- **Next.js 15+**: React framework with App Router
- **TypeScript**: Type-safe development
- **React 18+**: UI library

### UI & Styling
- **TailwindCSS**: Utility-first CSS framework
- **shadcn/ui**: High-quality component library
- **Radix UI**: Accessible component primitives
- **Lucide React**: Icon library

### State Management
- **React Query (TanStack Query)**: Server state management
- **Zustand**: Client state management
- **React Hook Form**: Form state management

### Data Fetching
- **Axios**: HTTP client
- **React Query**: Data fetching and caching
- **SWR**: Alternative data fetching (optional)

### Validation & Forms
- **Zod**: Schema validation
- **React Hook Form**: Form handling

### Charts & Visualization
- **Recharts**: Chart library
- **TanStack Table**: Data tables

### Utilities
- **date-fns**: Date manipulation
- **clsx**: Conditional class names
- **class-variance-authority**: Component variants

##  Prerequisites

### Required
- Node.js 18+ (LTS recommended)
- npm 9+ or yarn 1.22+ or pnpm 8+
- Git

### Recommended
- VS Code with extensions:
  - ESLint
  - Prettier
  - TypeScript
  - Tailwind CSS IntelliSense

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/athalasiem/athalasiem.git
cd athalasiem/frontend
```

### 2. Install Dependencies

```bash
npm install
# or
yarn install
# or
pnpm install
```

### 3. Configure Environment

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:9595
NEXT_PUBLIC_GRPC_URL=http://localhost:9595
```

### 4. Run Development Server

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:7654](http://localhost:7654) in your browser.

## 📦 Installation

### Development Setup

```bash
# 1. Install dependencies
npm install

# 2. Set up environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# 3. Run development server
npm run dev
```

### Production Build

```bash
# 1. Build for production
npm run build

# 2. Start production server
npm start

# 3. Or use PM2
pm2 start npm --name "athala-frontend" -- start
```

## ⚙️ Configuration

### Environment Variables

Create `.env.local` (or `.env.production` for production):

```env
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:9595

# gRPC URL (if using gRPC)
NEXT_PUBLIC_GRPC_URL=http://localhost:9595

# Frontend URL (for CORS)
NEXT_PUBLIC_FRONTEND_URL=http://localhost:7654

# Feature flags (optional)
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_2FA=true
```

### next.config.js

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9595',
  },
  // Add other configuration as needed
}

module.exports = nextConfig
```

## 🛠️ Development

### Available Scripts

```bash
# Development
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run format       # Format code with Prettier
npm run type-check   # Run TypeScript type checking
```

### Code Style

- **TypeScript**: Strict mode enabled
- **ESLint**: Extended Next.js and React rules
- **Prettier**: Code formatting
- **Functional Components**: Use hooks, avoid classes
- **Named Exports**: Prefer named over default exports

### Component Structure

```typescript
// components/Feature/ComponentName.tsx
import { ComponentProps } from './types'

export function ComponentName({ prop1, prop2 }: ComponentProps) {
  // Component implementation
  return <div>...</div>
}
```

### API Integration

```typescript
// lib/api.ts
import api from './api'

// Usage in components
const { data, isLoading, error } = useQuery({
  queryKey: ['agents'],
  queryFn: () => api.get('/api/agents'),
})
```

### State Management

```typescript
// Server state (React Query)
const { data } = useQuery({
  queryKey: ['users'],
  queryFn: fetchUsers,
})

// Client state (Zustand)
const useStore = create((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
}))
```

## 🚢 Deployment

### Vercel (Recommended)

```bash
# 1. Install Vercel CLI
npm i -g vercel

# 2. Deploy
vercel

# 3. Set environment variables in Vercel dashboard
```

### Docker

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 7654
CMD ["node", "server.js"]
```

```bash
# Build and run
docker build -t athala-frontend .
docker run -p 7654:7654 -e NEXT_PUBLIC_API_URL=http://backend:9595 athala-frontend
```

### Traditional Server (Node.js)

```bash
# 1. Build
npm run build

# 2. Start with PM2
pm2 start npm --name "athala-frontend" -- start

# 3. Or use systemd
sudo systemctl start athala-frontend
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name siem.yourdomain.com;

    location / {
        proxy_pass http://localhost:7654;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── dashboard/         # Dashboard pages
│   │   │   ├── admin/         # Admin pages
│   │   │   ├── agents/        # Agent management
│   │   │   ├── alerts/        # Alert management
│   │   │   ├── logs/          # Log viewing
│   │   │   └── profile/       # User profile
│   │   ├── login/             # Authentication
│   │   ├── register/          # Registration
│   │   └── layout.tsx         # Root layout
│   ├── components/            # React components
│   │   ├── Agents/           # Agent components
│   │   ├── Dashboard/        # Dashboard components
│   │   ├── Security/         # Security components
│   │   ├── Admin/           # Admin components
│   │   ├── Profile/         # Profile components
│   │   └── ui/              # shadcn/ui components
│   ├── lib/                  # Utilities
│   │   ├── api.ts           # API client
│   │   ├── utils.ts         # Helper functions
│   │   └── constants.ts     # Constants
│   ├── types/               # TypeScript types
│   ├── hooks/               # Custom hooks
│   ├── contexts/            # React contexts
│   └── styles/              # Global styles
├── public/                  # Static assets
├── .env.local              # Environment variables
├── next.config.js         # Next.js configuration
├── tailwind.config.js     # Tailwind configuration
└── tsconfig.json          # TypeScript configuration
```

## 🐛 Troubleshooting

### Build Errors

```bash
# Clear Next.js cache
rm -rf .next
npm run build

# Clear node_modules
rm -rf node_modules package-lock.json
npm install
```

### API Connection Issues

1. **Check environment variables:**
   ```bash
   echo $NEXT_PUBLIC_API_URL
   ```

2. **Verify backend is running:**
   ```bash
   curl http://localhost:9595/api/health
   ```

3. **Check CORS configuration** in backend

### TypeScript Errors

```bash
# Run type checking
npm run type-check

# Fix auto-fixable issues
npm run lint -- --fix
```

### Performance Issues

1. **Enable production build:**
   ```bash
   npm run build
   npm start
   ```

2. **Check bundle size:**
   ```bash
   npm run build
   # Check .next/analyze for bundle analysis
   ```

3. **Optimize images:**
   - Use Next.js Image component
   - Optimize image formats (WebP)

## 📚 Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Query Documentation](https://tanstack.com/query/latest)
- [TailwindCSS Documentation](https://tailwindcss.com/docs)
- [shadcn/ui Documentation](https://ui.shadcn.com)

## 📄 License

Copyright © 2025 Athala Security Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**🎉 Modern SIEM dashboard!** Monitor and manage your security infrastructure with ease.
