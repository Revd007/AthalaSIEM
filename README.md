# 🛡️ AthalaSIEM Agent (.NET)

<p align="center">
  <em>Cross-Platform Log Collection Agent for AthalaSIEM</em>
</p>

## 📋 Overview

The AthalaSIEM Agent is a high-performance .NET 8 application designed to collect security logs and events from various sources and forward them to the AthalaSIEM Backend. It provides real-time log collection, intelligent parsing, and secure communication with the central SIEM system.

## ✨ Key Features

- 🔍 **Multi-Source Collection**: Collect logs from Windows Event Logs, Syslog, containers, cloud services, and more
- 🚀 **High-Performance**: Built with .NET 8 for optimal performance and low resource usage
- 🔐 **Secure Communication**: gRPC and HTTPS communication with JWT authentication
- 🌐 **Cross-Platform**: Runs on Windows, Linux, and containerized environments
- 📊 **Real-time Processing**: Immediate log forwarding with configurable batching
- 🛠️ **Modular Design**: Pluggable collectors for different log sources
- 📈 **Intelligent Parsing**: Smart log parsing and normalization
- 🔔 **Health Monitoring**: Built-in health checks and monitoring capabilities
- 📋 **Configuration Management**: Centralized configuration from the backend
- 🔄 **Automatic Recovery**: Resilient design with automatic reconnection and retry logic

## 🏗️ Architecture

```
┌─────────────────┐    Log Collection    ┌─────────────────┐    gRPC/HTTP    ┌─────────────────┐
│                 │ ──────────────────► │                 │ ──────────────► │                 │
│  Log Sources    │                     │  AthalaSIEM     │                 │  AthalaSIEM     │
│  (Various)      │                     │  Agent          │                 │  Backend        │
│                 │                     │                 │ ◄────────────── │                 │
└─────────────────┘                     └─────────────────┘   Config/Auth   └─────────────────┘
```

### Supported Log Sources

- **Windows Event Logs**: Application, System, Security, and custom logs
- **Linux Syslog**: System logs via rsyslog/syslog-ng
- **Container Logs**: Docker and Kubernetes container logs
- **Cloud Services**: AWS CloudTrail, Azure Activity Logs, GCP Audit Logs
- **Database Logs**: SQL Server, MySQL, PostgreSQL audit logs
- **IoT Devices**: MQTT, Modbus, and custom IoT protocols
- **Network Devices**: Cisco, Juniper, and other network equipment logs
- **File Integrity**: File system monitoring and integrity checking

## 🛠️ Tech Stack

- **.NET 8.0**: Latest .NET framework for high performance
- **gRPC**: High-performance communication with backend
- **System.Text.Json**: Fast JSON serialization
- **Microsoft.Extensions.Hosting**: Background service hosting
- **Microsoft.Extensions.Configuration**: Configuration management
- **Microsoft.Extensions.Logging**: Structured logging
- **System.IO.FileSystemWatcher**: File system monitoring
- **System.Diagnostics.EventLog**: Windows Event Log access

## 📊 Current Status

| Component | Status | Build | Tests | Production Ready |
|-----------|--------|-------|-------|------------------|
| Core Agent | ✅ Complete | ✅ 0 Errors | ✅ Passing | ✅ Ready |
| Log Collectors | ✅ Complete | ✅ 0 Errors | ✅ Passing | ✅ Ready |
| Communication | ✅ Complete | ✅ 0 Errors | ✅ Passing | ✅ Ready |
| Configuration | ✅ Complete | ✅ 0 Errors | ✅ Passing | ✅ Ready |

## 🚀 Quick Start

### Prerequisites

- .NET 8.0 Runtime (or SDK for development)
- Windows 10+ or Linux (Ubuntu 18.04+, CentOS 7+)
- Network connectivity to AthalaSIEM Backend
- Appropriate permissions for log access

### 1. Download and Install

#### Windows
```powershell
# Download the latest release
Invoke-WebRequest -Uri "https://github.com/yourusername/AthalaSIEM/releases/latest/download/athala-agent-win-x64.zip" -OutFile "athala-agent.zip"
Expand-Archive -Path "athala-agent.zip" -DestinationPath "C:\Program Files\AthalaSIEM\Agent"

# Install as Windows Service
cd "C:\Program Files\AthalaSIEM\Agent"
.\AthalaSIEM.Agent.exe install
```

#### Linux
```bash
# Download and extract
wget https://github.com/yourusername/AthalaSIEM/releases/latest/download/athala-agent-linux-x64.tar.gz
sudo tar -xzf athala-agent-linux-x64.tar.gz -C /opt/athala-agent

# Install as systemd service
sudo cp /opt/athala-agent/athala-agent.service /etc/systemd/system/
sudo systemctl enable athala-agent
```

### 2. Configure Agent

Edit `appsettings.json`:

```json
{
  "AgentSettings": {
    "AgentName": "Production-Agent-01",
    "BackendApiUrl": "https://your-backend-server:9596",
    "BackendGrpcUrl": "https://your-backend-server:50051",
    "ApiKey": "your-agent-api-key",
    "LogBatchSize": 100,
    "LogSendingIntervalSeconds": 30,
    "HeartbeatIntervalSeconds": 60
  },
  "Collectors": [
    {
      "Type": "WindowsEventLog",
      "Enabled": true,
      "IntervalSeconds": 10,
      "Properties": {
        "EventLogs": "Application,System,Security",
        "MaxEvents": 1000
      }
    },
    {
      "Type": "FileIntegrity",
      "Enabled": true,
      "IntervalSeconds": 300,
      "Properties": {
        "WatchPaths": "C:\\Windows\\System32,C:\\Program Files",
        "IncludeSubdirectories": "true"
      }
    }
  ]
}
```

### 3. Start Agent

#### Windows
```powershell
# Start as service
Start-Service "AthalaSIEM Agent"

# Or run directly
.\AthalaSIEM.Agent.exe
```

#### Linux
```bash
# Start as service
sudo systemctl start athala-agent

# Or run directly
./AthalaSIEM.Agent
```

## 🔧 Configuration

### appsettings.json

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=localhost;Database=AthalaSIEM;Trusted_Connection=true;TrustServerCertificate=true;"
  },
  "JwtSettings": {
    "SecretKey": "your-256-bit-secret-key-here",
    "Issuer": "AthalaSIEM",
    "Audience": "AthalaSIEM-Users",
    "ExpirationMinutes": 60
  },
  "GrpcSettings": {
    "Port": 50051,
    "EnableTls": true,
    "MaxReceiveMessageSize": 4194304,
    "MaxSendMessageSize": 4194304
  },
  "LogSettings": {
    "MaxBatchSize": 1000,
    "BatchTimeoutSeconds": 30,
    "EnableCompression": true
  },
  "AlertSettings": {
    "EnableRealTimeAlerts": true,
    "MaxAlertsPerMinute": 100,
    "AlertRetentionDays": 90
  },
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  }
}
```

### Environment Variables

```bash
# Database
ATHALA_DB_CONNECTION="Server=localhost;Database=AthalaSIEM;Trusted_Connection=true;"

# JWT
ATHALA_JWT_SECRET="your-secret-key"
ATHALA_JWT_ISSUER="AthalaSIEM"

# gRPC
ATHALA_GRPC_PORT="50051"
ATHALA_GRPC_TLS="true"
```

## 📡 API Endpoints

### Authentication
- `POST /api/auth/login` - User authentication
- `POST /api/auth/refresh` - Token refresh
- `POST /api/auth/logout` - User logout

### Logs
- `GET /api/logs` - Retrieve logs with filtering
- `POST /api/logs/batch` - Submit log batch
- `GET /api/logs/{id}` - Get specific log entry
- `DELETE /api/logs/{id}` - Delete log entry

### Alerts
- `GET /api/alerts` - Retrieve alerts
- `POST /api/alerts` - Create new alert
- `PUT /api/alerts/{id}` - Update alert
- `DELETE /api/alerts/{id}` - Delete alert

### Agents
- `GET /api/agents` - List registered agents
- `POST /api/agents/register` - Register new agent
- `PUT /api/agents/{id}` - Update agent configuration
- `DELETE /api/agents/{id}` - Unregister agent

### Dashboard
- `GET /api/dashboard/stats` - System statistics
- `GET /api/dashboard/metrics` - Performance metrics
- `GET /api/dashboard/threats` - Threat summary

## 🔌 gRPC Services

### LogService
- `SubmitLogs` - High-performance log submission
- `SubmitLogBatch` - Batch log submission
- `GetLogStream` - Real-time log streaming

### AgentService
- `RegisterAgent` - Agent registration
- `GetConfiguration` - Agent configuration retrieval
- `SendHeartbeat` - Agent health monitoring

### AlertService
- `GetAlertStream` - Real-time alert streaming
- `AcknowledgeAlert` - Alert acknowledgment

## 🗄️ Database Schema

### Core Tables
- **Logs**: Primary log storage table
- **Alerts**: Alert definitions and instances
- **Agents**: Registered agent information
- **Users**: User accounts and authentication
- **Roles**: Role-based access control
- **AuditLogs**: System audit trail

### Key Relationships
- Logs → Agents (Many-to-One)
- Alerts → Logs (Many-to-Many)
- Users → Roles (Many-to-Many)
- AuditLogs → Users (Many-to-One)

## 🚀 Production Deployment

### 1. Build for Production

```bash
dotnet publish -c Release -o ./publish
```

### 2. Docker Deployment

```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY publish/ .
EXPOSE 80 443 50051
ENTRYPOINT ["dotnet", "AthalaSIEM.Backend.dll"]
```

```bash
docker build -t athala-siem-backend .
docker run -d -p 9595:80 -p 9596:443 -p 50051:50051 athala-siem-backend
```

### 3. IIS Deployment

1. Install .NET 8 Hosting Bundle on IIS server
2. Create new IIS application
3. Copy published files to application directory
4. Configure application pool for .NET Core
5. Set up SSL certificates for HTTPS

### 4. Cloud Deployment

#### Azure App Service
```bash
az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name athala-siem-backend --runtime "DOTNETCORE|8.0"
az webapp deployment source config-zip --resource-group myResourceGroup --name athala-siem-backend --src publish.zip
```

#### AWS Elastic Beanstalk
```bash
eb init athala-siem-backend --platform "64bit Amazon Linux 2 v2.2.0 running .NET Core"
eb create production
eb deploy
```

## 🔒 Security Features

### Authentication & Authorization
- **JWT Tokens**: Secure, stateless authentication
- **Role-Based Access Control**: Granular permission management
- **API Key Authentication**: For service-to-service communication
- **Multi-Factor Authentication**: Optional 2FA support

### Data Protection
- **TLS Encryption**: All communications encrypted in transit
- **Data Encryption**: Sensitive data encrypted at rest
- **Input Validation**: Comprehensive input sanitization
- **SQL Injection Protection**: Parameterized queries and ORM

### Audit & Compliance
- **Comprehensive Logging**: All actions logged and auditable
- **Data Retention**: Configurable data retention policies
- **Compliance Reports**: Built-in compliance reporting
- **Access Monitoring**: Real-time access monitoring and alerting

## 📈 Performance Optimization

### Database Optimization
- **Indexing Strategy**: Optimized indexes for query performance
- **Connection Pooling**: Efficient database connection management
- **Query Optimization**: Optimized Entity Framework queries
- **Caching**: Redis caching for frequently accessed data

### API Performance
- **Async/Await**: Non-blocking asynchronous operations
- **Response Compression**: Gzip compression for API responses
- **Rate Limiting**: API rate limiting to prevent abuse
- **Load Balancing**: Support for horizontal scaling

### gRPC Optimization
- **Message Compression**: Efficient binary serialization
- **Connection Multiplexing**: Single connection for multiple streams
- **Streaming**: Support for bidirectional streaming
- **Keep-Alive**: Connection keep-alive for persistent connections

## 🔍 Monitoring & Diagnostics

### Health Checks
- **Database Connectivity**: SQL Server connection health
- **External Services**: Third-party service availability
- **Memory Usage**: Application memory monitoring
- **Disk Space**: Available disk space monitoring

### Metrics
- **Request Metrics**: API request rates and response times
- **Error Rates**: Application error tracking
- **Performance Counters**: System performance metrics
- **Custom Metrics**: Business-specific metrics

### Logging
- **Structured Logging**: JSON-formatted logs with Serilog
- **Log Levels**: Configurable log levels (Debug, Info, Warning, Error)
- **Log Sinks**: Multiple output destinations (File, Console, Database)
- **Correlation IDs**: Request correlation for distributed tracing

## 🧪 Testing

### Unit Tests
```bash
dotnet test --configuration Release --logger trx --results-directory TestResults
```

### Integration Tests
```bash
dotnet test --configuration Release --filter Category=Integration
```

### Load Testing
```bash
# Using NBomber for load testing
dotnet run --project LoadTests --configuration Release
```

## 🛠️ Development

### Prerequisites
- .NET 8.0 SDK
- Visual Studio 2022 or VS Code
- SQL Server Developer Edition
- Git

### Setup Development Environment
```bash
git clone https://github.com/yourusername/AthalaSIEM.git
cd AthalaSIEM/backend
dotnet restore
dotnet ef database update
dotnet run
```

### Code Style
- Follow .NET coding conventions
- Use EditorConfig for consistent formatting
- Run code analysis with `dotnet analyze`
- Use XML documentation for public APIs

### Database Migrations
```bash
# Add new migration
dotnet ef migrations add MigrationName

# Update database
dotnet ef database update

# Generate SQL script
dotnet ef migrations script
```

## 📚 Additional Resources

- [.NET 8 Documentation](https://docs.microsoft.com/en-us/dotnet/)
- [ASP.NET Core Documentation](https://docs.microsoft.com/en-us/aspnet/core/)
- [Entity Framework Core Documentation](https://docs.microsoft.com/en-us/ef/core/)
- [gRPC Documentation](https://grpc.io/docs/)
- [JWT.io](https://jwt.io/) - JWT token debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

<p align="center">
  <strong>AthalaSIEM Backend - Secure, Scalable, Production-Ready</strong>
</p>
