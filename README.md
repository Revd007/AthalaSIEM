# Athala SIEM - Security Information and Event Management System

## Overview
Athala SIEM is a comprehensive Security Information and Event Management system designed to collect, analyze, and visualize security events and logs from multiple sources. The system helps organizations detect, investigate, and respond to security threats in real-time, providing a centralized platform for security monitoring and incident response.

![Athala SIEM Architecture](docs/images/architecture.png)

## Key Features

- **Centralized Log Management**: Collect and store logs from various sources in a unified platform
- **Real-time Security Monitoring**: Monitor security events across your infrastructure
- **Advanced Analytics**: Analyze logs with correlation rules to identify threats
- **Alert Management**: Generate, prioritize, and manage security alerts
- **Dashboard Visualization**: Visualize security data with interactive dashboards
- **Threat Intelligence Integration**: Enrich security data with threat intelligence
- **Agent-based Collection**: Deploy lightweight agents for efficient log collection
- **Compliance Reporting**: Generate reports for regulatory compliance
- **User and Role Management**: Granular access control with role-based permissions
- **AI-powered Analysis**: Advanced machine learning models for threat detection

## System Architecture

Athala SIEM follows a distributed architecture consisting of four main components:

- **Backend (.NET)**: Primary .NET Core API server that processes and stores security events
- **Backend (Python)**: Secondary Python-based AI and analytics engine for advanced threat detection
- **Frontend**: Next.js-based web application providing a user interface for monitoring and analysis
- **Agent**: .NET Core application that collects logs from systems and forwards them to the Backend

Communication between components:
- Agents ↔ Backend: gRPC for high-performance log streaming, REST API for configuration
- Frontend ↔ Backend: REST API with JWT authentication for secure communication
- Backend (.NET) ↔ Backend (Python): Internal API for AI analytics and advanced processing

### Architecture Diagram

```
┌───────────────┐     HTTP/gRPC      ┌──────────────┐    REST API     ┌──────────────┐
│               │  ───────────────>  │              │  ──────────────> │              │
│  SIEM Agents  │                    │  Backend.NET │                  │   Frontend   │
│               │  <───────────────  │              │  <────────────── │              │
└───────────────┘     Config/Auth    └──────────────┘     JWT Auth     └──────────────┘
                                            │
                                            │                          ┌──────────────┐
                                            │              REST API    │              │
                                            ├─────────────────────────>│ BackendPy    │
                                            │                          │ (AI Engine)  │
                                            │                          │              │
                                            ▼                          └──────────────┘
                                     ┌──────────────┐
                                     │  PostgreSQL  │
                                     │  Database    │
                                     └──────────────┘
```

## Components

### Backend (.NET)
The primary .NET Backend serves as the core of the system, responsible for:
- Processing and storing security events and logs
- Authenticating users and agents
- Generating alerts based on predefined rules
- Providing REST APIs for the Frontend
- Log analysis and correlation

**Tech Stack**:
- .NET 8.0
- Entity Framework Core
- PostgreSQL Database
- JWT Authentication
- gRPC for agent communication
- Hosted services for background processing

[Read more about the .NET Backend](backend/README.md)

### Backend (Python)
The Python Backend provides advanced analytics and AI-driven capabilities:
- AI-driven security event analysis
- Anomaly detection and behavioral analysis
- Machine learning models for threat detection
- Automated incident response recommendations
- Integration with threat intelligence feeds

**Tech Stack**:
- FastAPI framework
- MS SQL Server with SQLAlchemy
- Custom ML models
- Data processing pipelines
- OpenAPI documentation

[Read more about the Python Backend](backendpy/README.md)

### Frontend (Web UI)
The Frontend provides a modern, responsive interface for:
- Real-time monitoring of security events
- Viewing and managing alerts
- Analyzing log data through dashboards and visualizations
- Configuring system settings and alert rules
- User management

**Tech Stack**:
- Next.js 15.0
- React 18
- TypeScript
- Tailwind CSS
- Radix UI components
- Tremor and Recharts for visualizations

[Read more about the Frontend](frontend/README.md)

### Agent
The Agent component can be deployed on various systems to:
- Collect system logs, event logs, and security events
- Monitor system health and metrics
- Forward collected data to the Backend
- Execute response actions when triggered

**Tech Stack**:
- .NET 8.0
- Windows Service / Linux daemon capabilities
- Cross-platform compatibility (Windows/Linux)
- gRPC for efficient communication
- Configurable collectors for different log sources

[Read more about the Agent](agent/README.md)

## Prerequisites

### For Development
- .NET SDK 8.0 or later
- Python 3.9+ with virtual environment
- Node.js 18+ and npm
- PostgreSQL 14+
- MS SQL Server (for Python backend)
- Visual Studio 2022 or VS Code
- Git

### For Production Deployment
- Windows or Linux server for Backend hosting
- PostgreSQL database server
- MS SQL Server (for Python backend)
- Web server for Frontend hosting (or cloud service)
- Target systems for Agent deployment
- Outbound connectivity from Agents to Backend server

## Installation & Setup

### .NET Backend Setup
1. Clone the repository
   ```bash
   git clone https://github.com/athala-security/AthalaSIEM.git
   cd AthalaSIEM
   ```

2. Navigate to the backend directory
   ```bash
   cd backend
   ```

3. Update the database connection string in `appsettings.json`
   ```json
   "ConnectionStrings": {
     "DefaultConnection": "Host=localhost;Port=5432;Database=siem-db;Username=user;Password=password;"
   }
   ```

4. Apply database migrations
   ```bash
   dotnet ef database update
   ```

5. Run the backend
   ```bash
   dotnet run
   ```

   The API server will be available at:
   - HTTP: http://localhost:9595
   - HTTPS: https://localhost:9596
   - gRPC: http://localhost:50051

### Python Backend Setup
1. Navigate to the Python backend directory
   ```bash
   cd backendpy
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the database connection in `.env`
   ```
   DATABASE_URL=mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server
   ```

5. Run the Python backend
   ```bash
   uvicorn main:app --reload
   ```

   The AI API will be available at http://localhost:8000

### Frontend Setup
1. Navigate to the frontend directory
   ```bash
   cd frontend
   ```

2. Install dependencies
   ```bash
   npm install
   # or
   yarn
   ```

3. Configure the backend API endpoint in `.env.local`
   ```
   NEXT_PUBLIC_API_URL=http://localhost:9595
   NEXT_PUBLIC_GRPC_URL=http://localhost:50051
   NEXT_PUBLIC_AI_API_URL=http://localhost:8000
   ```

4. Run the frontend development server
   ```bash
   npm run dev
   # or
   yarn dev
   ```

   The web interface will be available at http://localhost:3000

### Agent Setup
1. Navigate to the agent directory
   ```bash
   cd agent
   ```

2. Build the agent
   ```bash
   dotnet build -c Release
   ```

3. For Windows installer:
   ```bash
   cd Installer/Windows
   .\build-installer.ps1
   ```

   The MSI installer will be generated in the `bin/Release` directory.

4. For Linux packages:
   ```bash
   ./build-linux-packages.sh
   ```

   The DEB and RPM packages will be generated in the `bin/Release` directory.

## Development Workflow

### Backend Development
1. Make changes to the backend code
2. Run tests: `dotnet test`
3. Start the backend: `dotnet run`
4. Test API endpoints using Swagger at https://localhost:9596/swagger

### Python Backend Development
1. Make changes to the Python backend code
2. Run the development server: `uvicorn main:app --reload`
3. Test API endpoints using Swagger at http://localhost:8000/docs

### Frontend Development
1. Make changes to the frontend code
2. Start the development server: `npm run dev`
3. Access the application at http://localhost:3000
4. Build for production: `npm run build`

### Agent Development
1. Make changes to the agent code
2. Build the agent: `dotnet build`
3. Test locally: `dotnet run`
4. Build installers using the provided scripts

## Docker Deployment

Docker Compose is available for easy deployment of the complete system:

```bash
docker-compose up -d
```

Individual components can also be built and run separately:

```bash
# .NET Backend
docker build -t athala-siem-backend ./backend
docker run -p 9595:9595 -p 9596:9596 -p 50051:50051 athala-siem-backend

# Python Backend
docker build -t athala-siem-backendpy ./backendpy
docker run -p 8000:8000 athala-siem-backendpy

# Frontend
docker build -t athala-siem-frontend ./frontend
docker run -p 3000:3000 athala-siem-frontend
```

## Documentation

Detailed documentation for each component is available in their respective directories:

- [.NET Backend Documentation](backend/README.md)
- [Python Backend Documentation](backendpy/README.md)
- [Frontend Documentation](frontend/README.md)
- [Agent Documentation](agent/README.md)

## License

Copyright © 2025 Athala Security Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
