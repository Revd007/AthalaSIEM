# ATHALA SIEM Backend

ATHALA SIEM (Security Information and Event Management) is a comprehensive security monitoring solution designed to collect, analyze, and respond to security events across your infrastructure.

## Overview

The backend component of ATHALA SIEM provides the core functionality for:
- Agent management and deployment
- Log collection, storage, and analysis
- Alert generation and management
- User authentication and authorization
- Real-time event processing
- Reporting and dashboard metrics
- Threat intelligence integration

## Tech Stack

- **Framework**: ASP.NET Core 8.0
- **Database**: PostgreSQL with Entity Framework Core
- **Authentication**: JWT (JSON Web Tokens)
- **API Documentation**: Swagger/OpenAPI
- **Communication Protocol**: gRPC for agent-server communication
- **Background Processing**: Hosted Services
- **Logging**: Structured logging with Serilog

## Project Structure

```
backend/
├── Controllers/               # API endpoints for REST communication
├── Services/                  # Business logic implementation
│   └── Background/            # Background services (agent monitoring, cleanup)
├── Data/                      # Database access
│   └── Repositories/          # Repository pattern implementation
├── Models/                    # Domain entities
├── DTOs/                      # Data Transfer Objects
├── Protos/                    # gRPC service definitions
├── Migrations/                # EF Core database migrations
├── Properties/                # Launch and project properties
└── Installers/                # Installation scripts
```

### Core Components

#### Controllers
The Controllers directory contains all REST API endpoints for:

- **`AgentsController.cs`**: Agent registration, configuration, and management
- **`AlertsController.cs`**: Alert query, management, and rule configuration
- **`AuthController.cs`**: User authentication and token generation
- **`DashboardsController.cs`**: Dashboard metrics and visualization data
- **`HealthController.cs`**: System health status endpoint
- **`LogsController.cs`**: Log ingestion, querying, and analysis
- **`ReportsController.cs`**: Security report generation and management
- **`UsersController.cs`**: User management and role assignments

#### Services
The Services directory contains the core business logic for:

- **`AgentService.cs`**: Manages agent registration, configuration, and status tracking
- **`AgentMonitoringService.cs`**: Monitors agent health and availability
- **`AlertService.cs`**: Generates and manages security alerts
- **`AlertProcessingService.cs`**: Processes incoming events for alert generation
- **`AuthService.cs`**: Handles user authentication and JWT token generation
- **`DashboardService.cs`**: Provides metrics for dashboard visualization
- **`InstallerService.cs`**: Manages agent installer packages
- **`LogAnalysisService.cs`**: Analyzes logs for security events and patterns
- **`LogService.cs`**: Processes and stores log entries
- **`ReportService.cs`**: Generates security reports and analysis
- **`SiemService.cs`**: gRPC implementation for agent communication
- **`UserService.cs`**: Manages user accounts and permissions

**Background Services**:
- **`AgentMonitoringService.cs`**: Monitors agent heartbeats and status
- **`LogCleanupService.cs`**: Handles log retention policies
- **`AlertCleanupService.cs`**: Manages alert lifecycle and cleanup

#### Data Layer
The Data directory contains database context and repository implementations:

- **`ApplicationDbContext.cs`**: EF Core database context
- **Repository Implementations**:
  - **`AgentRepository.cs`**: Agent data operations
  - **`AlertRepository.cs`**: Alert data operations
  - **`DashboardRepository.cs`**: Dashboard metrics operations
  - **`LogEntryRepository.cs`**: Log data operations
  - **`ReportRepository.cs`**: Report data operations
  - **`UserRepository.cs`**: User data operations
  - **`AgentDeploymentTokenRepository.cs`**: Token management
  - **`SystemConfigurationRepository.cs`**: System config operations

#### Models
The Models directory contains domain entities for:

- **Agent Management**: `AgentModels.cs`, `AgentConfigModels.cs`, `AgentStatus.cs`, `AgentHeartbeatModels.cs`
- **Alert Management**: `AlertModels.cs`, `AlertEnums.cs`, `AlertRuleModels.cs`
- **User Management**: `UserModels.cs`, `UserRoleModels.cs`, `RoleModels.cs`
- **Log Management**: `LogEntryModels.cs`, `LogSeverityModels.cs`
- **Dashboard & Reporting**: `DashboardModels.cs`, `ReportModels.cs`, `ComplianceReport.cs`
- **System Metrics**: `HealthMetricModels.cs`, `SystemConfiguration.cs`
- **Security**: `SecurityEventModels.cs`, `ThreatIntelligence.cs`

#### DTOs (Data Transfer Objects)
The DTOs directory contains objects used for API communications:

- **Agent DTOs**: `AgentDTOs.cs`, `AgentConfigDto.cs`, `AgentHeartbeatDto.cs`
- **Alert DTOs**: `AlertDTOs.cs`, `AlertDto.cs`
- **Log DTOs**: `LogDTOs.cs`, `LogEntryDTO.cs`
- **Health DTOs**: `HealthReportDTO.cs`, `HeartbeatDto.cs`, `SystemMetricsDto.cs`
- **Authentication DTOs**: `UserLoginDto.cs`, `UserRegisterDto.cs`

#### gRPC Services
The Protos directory contains protocol buffer definitions for agent communication:

- **`siem.proto`**: Defines gRPC services for agent-server communication including:
  - Agent registration and authentication
  - Configuration management
  - Log forwarding
  - Heartbeat monitoring
  - System metrics reporting

## API Endpoints

### Authentication
- `POST /api/auth/login`: Authenticate users
- `POST /api/auth/register`: Register new users (admin only)
- `POST /api/auth/refresh-token`: Refresh JWT token

### Agents
- `GET /api/agents`: List all registered agents
- `GET /api/agents/{id}`: Get agent details
- `POST /api/agents/register`: Register a new agent
- `PUT /api/agents/{id}`: Update agent configuration
- `DELETE /api/agents/{id}`: Deactivate an agent
- `POST /api/agents/token`: Generate agent deployment token
- `GET /api/agents/{id}/logs`: Get logs for a specific agent
- `GET /api/agents/{id}/health`: Get health status for a specific agent

### Logs
- `GET /api/logs`: Query logs with filtering
- `POST /api/logs/ingest`: Ingest single log entry
- `POST /api/logs/batch`: Ingest batch log entries
- `GET /api/logs/statistics`: Get log statistics
- `GET /api/logs/search`: Search logs with advanced filtering

### Alerts
- `GET /api/alerts`: Query alerts with filtering
- `GET /api/alerts/{id}`: Get alert details
- `PUT /api/alerts/{id}/status`: Update alert status
- `POST /api/alerts/rules`: Create/update alert rules
- `GET /api/alerts/rules`: Get all alert rules
- `GET /api/alerts/rules/{id}`: Get specific alert rule
- `DELETE /api/alerts/rules/{id}`: Delete alert rule

### Dashboard
- `GET /api/dashboards/summary`: Get dashboard summary statistics
- `GET /api/dashboards/recent-alerts`: Get recent alerts
- `GET /api/dashboards/agent-status`: Get agent status overview
- `GET /api/dashboards/log-volume`: Get log volume metrics
- `GET /api/dashboards/security-posture`: Get security posture score

### Reports
- `GET /api/reports`: List available reports
- `POST /api/reports/generate`: Generate a new report
- `GET /api/reports/{id}`: Get report details
- `GET /api/reports/templates`: Get available report templates
- `GET /api/reports/download/{id}`: Download report in requested format

### Users
- `GET /api/users`: List all users (admin only)
- `GET /api/users/{id}`: Get user details
- `POST /api/users`: Create a new user
- `PUT /api/users/{id}`: Update user details
- `DELETE /api/users/{id}`: Delete user
- `PUT /api/users/{id}/role`: Update user role

## Configuration

The application uses `appsettings.json` for configuration:

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Port=5432;Database=siem-db;Username=user;Password=password;"
  },
  "Jwt": {
    "Key": "your-secret-key",
    "Issuer": "AthalaSIEM",
    "Audience": "AthalaSIEMUsers",
    "ExpireMinutes": 60
  },
  "ApiKey": "your-api-key",
  "InstallerDownloadCode": "secure-download-code",
  "Kestrel": {
    "Endpoints": {
      "Http": { "Url": "http://0.0.0.0:9595" },
      "Https": { "Url": "https://0.0.0.0:9596" },
      "Grpc": {
        "Url": "http://0.0.0.0:50051",
        "Protocols": "Http2"
      }
    }
  }
}
```

## Getting Started

### Prerequisites
- .NET 8.0 SDK
- PostgreSQL 14+
- Visual Studio 2022 or VS Code

### Development Setup
1. Clone the repository
2. Update the connection string in `appsettings.json`
3. Run database migrations:
   ```
   dotnet ef database update
   ```
4. Start the application:
   ```
   dotnet run
   ```
5. Access Swagger UI at https://localhost:9596/swagger

### Docker Setup
```bash
docker build -t athala-siem-backend .
docker run -p 9595:9595 -p 9596:9596 -p 50051:50051 athala-siem-backend
```

## Architecture Principles

The backend follows these principles:
- **Clean Architecture**: Separation of concerns with clear boundaries
- **Repository Pattern**: Abstraction of data access
- **Dependency Injection**: For loose coupling between components
- **SOLID Principles**: Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion
- **RESTful API Design**: Consistent API patterns
- **Asynchronous Programming**: Non-blocking I/O operations
- **Domain-Driven Design**: Core entities reflect business domain

## License

Copyright © 2025 Athala Security Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 