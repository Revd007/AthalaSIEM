# 🛡️ AthalaSIEM Backend

**Enterprise-grade SIEM backend with production-ready architecture**

[![.NET](https://img.shields.io/badge/.NET-8.0-purple)](https://dotnet.microsoft.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-blue)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Database](#database)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## 📖 Overview

The AthalaSIEM Backend is a production-grade ASP.NET Core 8.0 application that provides the core SIEM functionality including agent management, log ingestion, alert generation, user authentication, and real-time event processing. It follows Clean Architecture principles with clear separation of concerns.

### Key Capabilities

- **Agent Management**: Registration, configuration, health monitoring
- **Log Ingestion**: High-throughput log processing (REST API and gRPC)
- **Alert Generation**: Real-time alert generation and management
- **User Management**: JWT authentication, role-based access control
- **Dashboard Metrics**: Real-time system metrics and statistics
- **Reporting**: Security reports and compliance documentation
- **Threat Intelligence**: Integration with threat intelligence feeds

## ✨ Features

### Core Functionality
- ✅ REST API for frontend and agent communication
- ✅ gRPC for high-performance agent communication
- ✅ PostgreSQL database with Entity Framework Core
- ✅ JWT authentication with refresh tokens
- ✅ Role-based access control (Admin, User, Analyst)
- ✅ Background services for monitoring and cleanup
- ✅ Structured logging with Serilog
- ✅ Health checks and metrics

### Agent Management
- ✅ Agent registration and authentication
- ✅ Configuration management
- ✅ Health monitoring and heartbeat tracking
- ✅ Installer package generation
- ✅ Deployment token management

### Log Processing
- ✅ High-throughput log ingestion
- ✅ Log normalization and enrichment
- ✅ Real-time alert generation
- ✅ Log archiving and retention
- ✅ Advanced search and filtering

### Security
- ✅ JWT token authentication
- ✅ Password hashing (BCrypt)
- ✅ Two-factor authentication (2FA)
- ✅ Password policy enforcement
- ✅ Session management
- ✅ User hardening settings

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                        │
│                    Port: 3000/7654                           │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API (HTTP/HTTPS)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend API Layer                          │
│                    Port: 9595 (HTTP) / 9596 (HTTPS)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Controllers  │  │   Services   │  │ Background   │        │
│  │  (REST API)  │  │  (Business   │  │   Services   │        │
│  │              │  │    Logic)    │  │  (Workers)   │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                 │                 │
│         └─────────────────┴─────────────────┘                 │
│                            │                                   │
│                            ▼                                   │
│                  ┌──────────────────┐                          │
│                  │  Repositories    │                          │
│                  │  (Data Access)   │                          │
│                  └────────┬─────────┘                          │
└───────────────────────────┼───────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL Database                       │
│                    Port: 5432                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Agents  │  │   Logs   │  │  Alerts  │  │  Users   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ gRPC (Port: 9595)
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Agents (Universal Agent)                   │
│                    Multiple Agents                            │
└─────────────────────────────────────────────────────────────┘
```

### Clean Architecture Layers

1. **Controllers**: API endpoints (REST)
2. **Services**: Business logic implementation
3. **Repositories**: Data access abstraction
4. **Models**: Domain entities
5. **DTOs**: Data transfer objects
6. **Infrastructure**: External service integrations

## 📋 Prerequisites

### Required
- .NET 8.0 SDK (for development) or Runtime (for production)
- PostgreSQL 14+ database
- Windows 10+, Linux (Ubuntu 18.04+, CentOS 7+), or Docker

### Optional
- Redis (for caching)
- Nginx/Apache (reverse proxy)
- SSL certificates (for HTTPS)

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/athalasiem/athalasiem.git
cd athalasiem/backend
```

### 2. Configure Database

```bash
# Create PostgreSQL database
createdb siem-db

# Or using psql
psql -U postgres -c "CREATE DATABASE \"siem-db\";"
```

### 3. Configure Application

Edit `appsettings.json`:

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Port=5432;Database=siem-db;Username=youruser;Password=yourpassword;"
  },
  "Jwt": {
    "Key": "your-256-bit-secret-key-here-change-in-production",
    "Issuer": "AthalaSIEM",
    "Audience": "AthalaSIEMUsers",
    "ExpireMinutes": 60
  }
}
```

### 4. Run Database Migrations

```bash
dotnet ef database update
```

### 5. Run Application

```bash
dotnet run
```

The backend will be available at:
- **HTTP**: http://localhost:9595
- **Swagger UI**: http://localhost:9595/swagger
- **gRPC**: http://localhost:9595 (same port)

## 📦 Installation

### Method 1: Docker (Recommended)

```bash
# Build image
docker build -t athala-siem-backend:latest .

# Run container
docker run -d \
  --name athala-backend \
  -p 9595:9595 \
  -e ConnectionStrings__DefaultConnection="Host=db;Port=5432;Database=siem-db;Username=postgres;Password=password;" \
  -e Jwt__Key="your-secret-key" \
  athala-siem-backend:latest

# Or use docker-compose
docker-compose up -d
```

### Method 2: Windows Service

```powershell
# 1. Publish application
dotnet publish -c Release -r win-x64 --self-contained -o ./publish

# 2. Install as Windows Service
# Use NSSM or similar tool
nssm install AthalaSIEMBackend "C:\Program Files\AthalaSIEM\Backend\backend.exe"
nssm set AthalaSIEMBackend AppDirectory "C:\Program Files\AthalaSIEM\Backend"
nssm start AthalaSIEMBackend
```

### Method 3: Linux systemd

```bash
# 1. Publish application
dotnet publish -c Release -r linux-x64 --self-contained -o ./publish

# 2. Copy to installation directory
sudo mkdir -p /opt/athala-backend
sudo cp -r ./publish/* /opt/athala-backend/

# 3. Create systemd service
sudo nano /etc/systemd/system/athala-backend.service
```

Service file (`/etc/systemd/system/athala-backend.service`):

```ini
[Unit]
Description=AthalaSIEM Backend
After=network.target postgresql.service

[Service]
Type=notify
WorkingDirectory=/opt/athala-backend
ExecStart=/opt/athala-backend/backend
Restart=always
RestartSec=10
User=athala
Environment=ASPNETCORE_ENVIRONMENT=Production
Environment=ConnectionStrings__DefaultConnection=Host=localhost;Port=5432;Database=siem-db;Username=athala;Password=password;

[Install]
WantedBy=multi-user.target
```

```bash
# 4. Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable athala-backend
sudo systemctl start athala-backend
```

### Method 4: IIS (Windows)

1. Install .NET 8.0 Hosting Bundle
2. Create IIS application pool (No Managed Code)
3. Create IIS application pointing to published folder
4. Configure SSL certificates
5. Set environment variables in web.config

## ⚙️ Configuration

### appsettings.json

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Port=5432;Database=siem-db;Username=user;Password=password;"
  },
  "Jwt": {
    "Key": "your-256-bit-secret-key-change-in-production",
    "Issuer": "AthalaSIEM",
    "Audience": "AthalaSIEMUsers",
    "ExpireMinutes": 60
  },
  "Kestrel": {
    "Endpoints": {
      "Http": {
        "Url": "http://0.0.0.0:9595"
      },
      "Https": {
        "Url": "https://0.0.0.0:9596",
        "Certificate": {
          "Path": "/path/to/certificate.pfx",
          "Password": "certificate-password"
        }
      }
    }
  },
  "Cors": {
    "AllowedOrigins": [
      "http://localhost:3000",
      "http://localhost:7654",
      "https://your-frontend-domain.com"
    ]
  },
  "GrpcServer": {
    "Url": "http://0.0.0.0:9595"
  },
  "LogArchiving": {
    "IntervalHours": 24,
    "RetentionDays": 90,
    "BatchSize": 1000,
    "Directory": "archives/logs",
    "EnableCompression": true
  }
}
```

### Environment Variables

Override configuration via environment variables:

```bash
# Database
export ConnectionStrings__DefaultConnection="Host=db;Port=5432;Database=siem-db;Username=user;Password=pass;"

# JWT
export Jwt__Key="your-secret-key"
export Jwt__ExpireMinutes=60

# Kestrel
export Kestrel__Endpoints__Http__Url="http://0.0.0.0:9595"

# CORS
export Cors__AllowedOrigins__0="http://localhost:3000"
export Cors__AllowedOrigins__1="https://your-domain.com"
```

### Production Configuration

Create `appsettings.Production.json`:

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Warning",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "Kestrel": {
    "Endpoints": {
      "Https": {
        "Url": "https://0.0.0.0:9596"
      }
    }
  }
}
```

## 🚢 Deployment

### Docker Compose (Full Stack)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: siem-db
      POSTGRES_USER: athala
      POSTGRES_PASSWORD: secure-password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: .
    environment:
      ConnectionStrings__DefaultConnection: "Host=postgres;Port=5432;Database=siem-db;Username=athala;Password=secure-password;"
      Jwt__Key: "your-secret-key"
      ASPNETCORE_ENVIRONMENT: Production
    ports:
      - "9595:9595"
    depends_on:
      - postgres
    volumes:
      - ./archives:/app/archives

volumes:
  postgres-data:
```

```bash
docker-compose up -d
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: athala-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: athala-backend
  template:
    metadata:
      labels:
        app: athala-backend
    spec:
      containers:
      - name: backend
        image: athalasiem/backend:latest
        ports:
        - containerPort: 9595
        env:
        - name: ConnectionStrings__DefaultConnection
          valueFrom:
            secretKeyRef:
              name: athala-secrets
              key: database-connection
        - name: Jwt__Key
          valueFrom:
            secretKeyRef:
              name: athala-secrets
              key: jwt-key
---
apiVersion: v1
kind: Service
metadata:
  name: athala-backend
spec:
  selector:
    app: athala-backend
  ports:
  - port: 9595
    targetPort: 9595
  type: LoadBalancer
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name siem.yourdomain.com;

    location / {
        proxy_pass http://localhost:9595;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection keep-alive;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📡 API Documentation

### Swagger UI

Access interactive API documentation at:
- **Development**: http://localhost:9595/swagger
- **Production**: https://your-domain.com/swagger

### Key Endpoints

#### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh-token` - Refresh JWT token

#### Agents
- `GET /api/agents` - List all agents
- `POST /api/agents/register` - Register new agent
- `GET /api/agents/{id}` - Get agent details
- `PUT /api/agents/{id}` - Update agent
- `DELETE /api/agents/{id}` - Delete agent
- `GET /api/agents/{id}/health` - Get agent health

#### Logs
- `GET /api/logs` - Query logs
- `POST /api/logs/ingest` - Ingest single log
- `POST /api/logs/batch` - Ingest log batch
- `GET /api/logs/statistics` - Get log statistics

#### Alerts
- `GET /api/alerts` - Query alerts
- `GET /api/alerts/{id}` - Get alert details
- `PUT /api/alerts/{id}/status` - Update alert status

#### Users
- `GET /api/users` - List users (Admin only)
- `POST /api/users` - Create user (Admin only)
- `PUT /api/users/{id}` - Update user
- `DELETE /api/users/{id}` - Delete user

### gRPC Services

- `RegisterAgent` - Agent registration
- `ForwardLogs` - Log forwarding
- `SendHeartbeat` - Agent heartbeat
- `GetAgentConfiguration` - Get agent config

## 🗄️ Database

### Schema Overview

- **Agents**: Registered agent information
- **LogEntries**: Ingested log entries
- **Alerts**: Generated security alerts
- **Users**: User accounts and authentication
- **Roles**: Role definitions
- **UserRoles**: User-role assignments
- **AgentConfigs**: Agent configurations
- **HealthReports**: Agent health reports

### Migrations

```bash
# Add new migration
dotnet ef migrations add MigrationName

# Update database
dotnet ef database update

# Generate SQL script
dotnet ef migrations script

# Rollback migration
dotnet ef database update PreviousMigrationName
```

### Backup and Restore

```bash
# Backup
pg_dump -U postgres -d siem-db -F c -f backup.dump

# Restore
pg_restore -U postgres -d siem-db -c backup.dump
```

## 🔒 Security

### Authentication
- JWT tokens with configurable expiration
- Refresh token support
- Password hashing with BCrypt
- Two-factor authentication (2FA)

### Authorization
- Role-based access control (RBAC)
- Admin, User, Analyst roles
- Endpoint-level authorization

### Data Protection
- HTTPS/TLS encryption
- SQL injection protection (parameterized queries)
- Input validation and sanitization
- CORS configuration

### Best Practices
1. **Change default JWT key** in production
2. **Use strong database passwords**
3. **Enable HTTPS** in production
4. **Configure CORS** properly
5. **Regular security updates**
6. **Monitor access logs**

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Test PostgreSQL connection
psql -h localhost -U postgres -d siem-db

# Check connection string format
# Should be: Host=localhost;Port=5432;Database=siem-db;Username=user;Password=pass;
```

### Application Won't Start

1. **Check logs:**
   ```bash
   # View application logs
   tail -f logs/backend-*.log
   ```

2. **Verify .NET Runtime:**
   ```bash
   dotnet --version
   # Should show 8.0.x
   ```

3. **Check port availability:**
   ```bash
   # Windows
   netstat -ano | findstr :9595
   
   # Linux
   sudo lsof -i :9595
   ```

### Migration Errors

```bash
# Reset database (development only)
dotnet ef database drop
dotnet ef database update

# Check migration status
dotnet ef migrations list
```

### Performance Issues

1. **Database optimization:**
   - Add indexes for frequently queried columns
   - Analyze query performance
   - Consider connection pooling

2. **Application optimization:**
   - Enable response compression
   - Use async/await properly
   - Monitor memory usage

## 🛠️ Development

### Prerequisites

- .NET 8.0 SDK
- PostgreSQL 14+
- Visual Studio 2022 or VS Code
- Git

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/athalasiem/athalasiem.git
cd athalasiem/backend

# Restore dependencies
dotnet restore

# Update database
   dotnet ef database update

# Run application
   dotnet run
   ```

### Project Structure

```
backend/
├── Controllers/          # REST API controllers
├── Services/             # Business logic services
├── Repositories/         # Data access repositories
├── Models/              # Domain entities
├── DTOs/                # Data transfer objects
├── Data/                # Database context
├── Infrastructure/      # External integrations
├── Workers/             # Background services
├── Protos/              # gRPC definitions
└── Migrations/          # EF Core migrations
```

### Running Tests

```bash
# Run all tests
dotnet test

# Run with coverage
dotnet test /p:CollectCoverage=true
```

### Code Style

- Follow .NET coding conventions
- Use EditorConfig for formatting
- XML documentation for public APIs
- Async/await for I/O operations

## 📚 Additional Resources

- [API Documentation](http://localhost:9595/swagger) - Interactive API docs
- [Architecture Analysis](../ATHALASIEM_BACKEND_ARCHITECTURE_ANALYSIS.md) - Detailed architecture
- [FIM Documentation](README_FIM_ENHANCED.md) - File Integrity Monitoring

## 📄 License

Copyright © 2025 Athala Security Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

---

**🎉 Production-ready SIEM backend!** Deploy enterprise security monitoring with confidence.
