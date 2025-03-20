# AthalaSIEM Frontend

Frontend application for AthalaSIEM, built with Next.js, TypeScript, and TailwindCSS.

## Project Structure

```
frontend/
├── src/
│   ├── app/                   # Next.js 13+ App Router routes
│   │   ├── dashboard/         # Dashboard pages
│   │   ├── login/            # Authentication pages
│   │   ├── register/         # Registration pages
│   │   ├── settings/         # Settings pages
│   │   ├── security/         # Security monitoring pages
│   │   └── compliance/       # Compliance reporting pages
│   ├── components/           # React components
│   │   ├── Agents/          # Agent management components
│   │   ├── Dashboard/       # Dashboard components
│   │   ├── Security/        # Security monitoring components
│   │   ├── Analytics/       # Analytics and reporting components
│   │   ├── Admin/          # Administrative components
│   │   └── ui/             # Shared UI components (shadcn/ui)
│   ├── services/            # API service layer
│   │   ├── agent-service.ts    # Agent management service
│   │   ├── auth-service.ts     # Authentication service
│   │   ├── monitoring-service.ts # System monitoring service
│   │   └── security-alert-service.ts # Security alert service
│   ├── lib/                 # Core utilities
│   │   ├── api.ts          # API client configuration
│   │   ├── api-endpoints.ts # API endpoint definitions
│   │   └── utils.ts        # Utility functions
│   ├── types/              # TypeScript type definitions
│   │   ├── agent.ts       # Agent related types
│   │   ├── auth.ts        # Authentication types
│   │   └── api.ts         # API response types
│   ├── hooks/             # Custom React hooks
│   ├── contexts/          # React context providers
│   ├── providers/         # Global providers
│   └── styles/            # Global styles
├── public/                # Static assets
└── [config files]         # Various configuration files
```

## Tech Stack

- **Framework**: Next.js 15+ (App Router)
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **UI Components**: shadcn/ui
- **State Management**: 
  - React Query (Server State)
  - Zustand (Client State)
- **Authentication**: JWT with refresh tokens
- **Form Handling**: React Hook Form
- **Validation**: Zod
- **Icons**: Lucide React
- **Charts**: Recharts
- **Tables**: TanStack Table
- **Date Handling**: date-fns

## Key Features

### 1. Authentication & Authorization
- JWT-based authentication
- Role-based access control
- Secure token management
- Protected routes

### 2. Dashboard & Monitoring
- Real-time system monitoring
- Customizable dashboards
- Performance metrics
- Health status indicators

### 3. Agent Management
- Agent deployment
- Configuration management
- Status monitoring
- Health checks
- Log collection

### 4. Security Monitoring
- Real-time event monitoring
- Alert management
- Incident response
- Threat detection
- Compliance monitoring

### 5. Analytics & Reporting
- Custom reports
- Data visualization
- Trend analysis
- Compliance reporting
- Performance analytics

### 6. System Administration
- User management
- Role management
- System configuration
- Audit logging
- API key management

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env.local
   ```
   Edit `.env.local` with your configuration:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:9595
   NEXT_PUBLIC_GRPC_URL=http://localhost:50051
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:7654](http://localhost:7654) in your browser.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier

## Development Guidelines

### Code Style
- Use TypeScript for type safety
- Follow ESLint and Prettier configurations
- Use functional components with hooks
- Implement proper error handling
- Write meaningful comments

### Component Structure
- Place components in appropriate feature folders
- Use shared UI components from shadcn/ui
- Implement proper prop types
- Handle loading and error states

### State Management
- Use React Query for server state
- Use Zustand for global client state
- Use local state for component-specific state
- Implement proper caching strategies

### API Integration
- Use the centralized API client
- Implement proper error handling
- Use TypeScript types for API responses
- Handle loading and error states

## Responsive Design

The UI is fully responsive and optimized for:
- Desktop workstations (1920x1080+)
- Laptops (1366x768+)
- Tablets (768x1024+)
- Mobile devices (320x568+)

## Security Considerations

- Implement proper authentication
- Use secure token management
- Implement proper authorization
- Handle sensitive data properly
- Follow security best practices

## Browser Support

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