# ATHALA SIEM Frontend

The frontend component of the ATHALA SIEM (Security Information and Event Management) platform provides a modern, responsive user interface for security monitoring, incident response, and system management.

## Overview

The ATHALA SIEM Frontend offers a comprehensive dashboard and management interface for:
- Real-time security event monitoring
- Agent deployment and management
- Alert investigation and response
- Log analysis and threat hunting
- System health monitoring
- Compliance reporting
- User and permission management

## Tech Stack

- **Framework**: Next.js 15.0 with App Router
- **Language**: TypeScript
- **UI Components**: 
  - Radix UI primitives
  - Tailwind CSS for styling
  - Tremor for data visualization
  - Tanstack React Table for data displays
- **State Management**: 
  - Zustand for global state
  - React Context for theme and authentication
- **Data Fetching**: 
  - Tanstack React Query
  - Axios for API requests
- **Charts and Visualization**:
  - Recharts
  - D3.js
  - React Leaflet for geospatial mapping
- **Deployment**: Containerized with Docker

## Project Structure

```
frontend/
├── src/
│   ├── app/                   # Next.js App Router routes
│   ├── components/            # UI Components
│   │   ├── ui/                # Shared UI components
│   │   ├── Dashboard/         # Dashboard-specific components
│   │   ├── Agents/            # Agent management components
│   │   ├── SecurityEvents/    # Event monitoring components
│   │   └── ...                # Other feature-specific components
│   ├── services/              # API service clients
│   ├── contexts/              # React Context providers
│   ├── hooks/                 # Custom React hooks
│   ├── types/                 # TypeScript type definitions
│   ├── utils/                 # Utility functions
│   ├── lib/                   # Shared libraries and configurations
│   └── styles/                # Global styles
├── public/                    # Static assets
├── next.config.js             # Next.js configuration
└── tailwind.config.js         # Tailwind CSS configuration
```

## Key Features

### Dashboard
- **Security Overview**: Real-time security posture with threat indicators
- **Agent Status**: Live monitoring of deployed agents
- **Alert Statistics**: Visual representations of alert trends and severity
- **System Health**: Performance metrics and health status
- **Recent Activities**: Timeline of recent security events

### Security Monitoring
- **Live Event Stream**: Real-time security event monitoring
- **Event Filtering**: Advanced filtering and search capabilities
- **Alert Management**: View, assign, and resolve security alerts
- **Incident Response**: Guided workflows for handling security incidents
- **SIEM Correlation**: Visual representation of event correlations

### Agent Management
- **Agent Deployment**: Generate deployment tokens and installer packages
- **Agent Configuration**: Remote configuration of data collection parameters
- **Health Monitoring**: Track agent performance and status
- **Log Collection**: Configure log sources and collection policies

### Analytics and Reporting
- **Threat Intelligence**: Integration with threat feeds
- **Compliance Reporting**: Pre-configured compliance report templates
- **Custom Reports**: Configurable reporting with export options
- **Security Metrics**: Key security indicators and metrics
- **Predictive Analysis**: ML-powered risk prediction

### Administration
- **User Management**: Create and manage users and roles
- **System Configuration**: Configure system-wide settings
- **Audit Trail**: Comprehensive logging of administrative actions
- **API Key Management**: Generate and manage API keys for integrations

## Component Architecture

The frontend follows a modular component architecture:

### Core Layout Components
- `Layout`: Base layout with navigation
- `Header`: Application header with user controls
- `Sidebar`: Main navigation sidebar
- `Dashboard`: Main dashboard container

### Feature-specific Components
- Each major feature has its dedicated component directory
- Components follow a hierarchical structure (containers -> elements)
- Shared UI elements are in the `ui` directory

### Service Layer
- API communication is abstracted through service modules
- Each service focuses on a specific domain (auth, agents, events, etc.)
- Services handle data fetching, caching, and error management

## Getting Started

### Prerequisites
- Node.js 18.0+ 
- npm or yarn
- ATHALA SIEM Backend running and accessible

### Development Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   # or
   yarn
   ```
3. Configure the backend API URL in `.env.local`:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:9595
   NEXT_PUBLIC_GRPC_URL=http://localhost:50051
   ```
4. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```
5. Open [http://localhost:7654](http://localhost:7654) in your browser

### Building for Production
```bash
npm run build
# or
yarn build
```

### Docker Deployment
```bash
docker build -t athala-siem-frontend .
docker run -p 7654:7655 -e NEXT_PUBLIC_API_URL=<backend_url> athala-siem-frontend
```

## Authentication

The frontend uses JWT authentication with the following flow:
1. User login with credentials
2. JWT token stored in memory and HTTP-only cookies
3. Automatic token refresh
4. Protected routes with authentication checks

## Theme System

The application supports both light and dark themes:
- Theme selection persisted in local storage
- Automatic system preference detection
- Consistent design across both themes

## Responsive Design

The UI is fully responsive and optimized for:
- Desktop workstations (1920x1080+)
- Laptops (1366x768+)
- Tablets (768x1024+)
- Mobile devices (320x568+)

## Extension Points

The frontend is designed for extensibility:
- Custom dashboard widgets
- Additional visualization components
- New report templates
- Integration with external security tools

## Browser Compatibility

The application is tested and supported on:
- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Edge (latest 2 versions)
- Safari (latest 2 versions)

## Performance Optimizations

- Code splitting for route-based chunking
- Static page generation where applicable
- Image optimization with Next.js Image component
- Efficient data fetching with React Query
- Virtualized lists for large datasets

## License

Copyright © 2025 Athala Security Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 