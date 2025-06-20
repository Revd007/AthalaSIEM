# 🛡️ AthalaSIEM

<p align="center">
  <img src="docs/assets/athala-logo.png" alt="AthalaSIEM Logo" width="200"/>
  <br>
  <em>Next Generation Security Information and Event Management System</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#documentation">Documentation</a>
</p>

## ✨ Features

- 🔍 **Real-time Log Collection** - Multi-source log aggregation with intelligent parsing
- 🤖 **AI-Powered Threat Detection** - Advanced machine learning models for anomaly detection
- 📊 **Interactive Dashboard** - Modern Next.js-based UI with real-time monitoring
- 🚀 **High-Performance Architecture** - .NET 8 backend with gRPC communication
- 🔐 **Enterprise Security** - JWT authentication, TLS encryption, and RBAC
- 🌐 **Scalable Design** - Microservices architecture built for enterprise scale
- 📈 **Advanced Analytics** - Comprehensive log analysis and correlation
- 🔔 **Intelligent Alerting** - Smart alert generation with customizable rules
- 🛠️ **Multi-Platform Agents** - Cross-platform log collection agents
- 📋 **Compliance Ready** - Built-in compliance reporting and audit trails

## 🏗️ Architecture

AthalaSIEM uses a modern, distributed architecture with three main components:

```
┌─────────────────┐    gRPC/HTTP     ┌─────────────────┐    REST API     ┌─────────────────┐
│                 │ ──────────────► │                 │ ──────────────► │                 │
│  SIEM Agents    │                 │  .NET Backend   │                 │  Next.js        │
│  (.NET 8)       │ ◄────────────── │  (API Server)   │ ◄────────────── │  Frontend       │
│                 │   Config/Auth   │                 │   JWT Auth      │                 │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   SQL Server    │
                                    │   Database      │
                                    └─────────────────┘
```

### Components

- **Backend (.NET 8)**: High-performance API server with gRPC and REST endpoints
- **Frontend (Next.js 15)**: Modern React-based dashboard with TypeScript
- **Agent (.NET 8)**: Cross-platform log collection service
- **Database**: SQL Server with Entity Framework Core

## 🚀 Quick Start

### Prerequisites

- .NET 8.0 SDK
- Node.js 18+
- SQL Server 2019+ (or SQL Server Express)
- 8GB RAM minimum

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/AthalaSIEM.git
cd AthalaSIEM
```

### 2. Setup Backend

```bash
cd backend
dotnet restore
dotnet ef database update
dotnet run
```

Backend will be available at:
- HTTP API: http://localhost:9595
- HTTPS API: https://localhost:9596
- gRPC: http://localhost:50051

### 3. Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at: http://localhost:3000

### 4. Deploy Agent

```bash
cd agent
dotnet publish -c Release
# Copy published files to target systems
```

## 🚀 Production Deployment

### Backend Deployment

```bash
cd backend
dotnet publish -c Release -o ./publish
# Deploy to IIS, Docker, or cloud service
```

### Frontend Deployment

```bash
cd frontend
npm run build
npm start
# Or deploy to Vercel, Netlify, or cloud service
```

### Agent Deployment

```bash
cd agent
dotnet publish -c Release --self-contained -r win-x64
# For Windows systems

dotnet publish -c Release --self-contained -r linux-x64
# For Linux systems
```

## 📊 Current Status

| Component | Status | Build | Tests | Production Ready |
|-----------|--------|-------|-------|------------------|
| Backend | ✅ Complete | ✅ 0 Errors | ✅ Passing | ✅ Ready |
| Frontend | ✅ Complete | ✅ 0 Errors | ✅ Passing | ✅ Ready |
| Agent | ✅ Complete | ✅ 0 Errors | ✅ Passing | ✅ Ready |
| Database | ✅ Complete | ✅ Migrated | ✅ Ready | ✅ Ready |

## 🔧 Configuration

### Backend Configuration (`appsettings.json`)

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=localhost;Database=AthalaSIEM;Trusted_Connection=true;"
  },
  "JwtSettings": {
    "SecretKey": "your-secret-key",
    "Issuer": "AthalaSIEM",
    "Audience": "AthalaSIEM-Users"
  },
  "GrpcSettings": {
    "Port": 50051,
    "EnableTls": true
  }
}
```

### Agent Configuration (`appsettings.json`)

```json
{
  "AgentSettings": {
    "AgentName": "Production-Agent-01",
    "BackendApiUrl": "https://your-backend-server:9596",
    "BackendGrpcUrl": "https://your-backend-server:50051",
    "LogBatchSize": 100,
    "LogSendingIntervalSeconds": 30
  },
  "Collectors": [
    {
      "Type": "WindowsEventLog",
      "Enabled": true,
      "Properties": {
        "EventLogs": "Application,System,Security"
      }
    }
  ]
}
```

## 📚 Documentation

- [Backend API Documentation](backend/README.md)
- [Agent Configuration Guide](agent/README.md)
- [Frontend User Guide](frontend/README.md)
- [Deployment Guide](docs/deployment.md)
- [Security Features](docs/security.md)

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **TLS Encryption**: End-to-end encryption for all communications
- **Role-Based Access Control**: Granular permission management
- **Audit Logging**: Comprehensive audit trail
- **Data Integrity**: Hash-based log integrity verification
- **Secure Configuration**: Encrypted configuration storage

## 🌟 Key Capabilities

### Log Collection
- Windows Event Logs
- Linux Syslog
- Container Logs (Docker/Kubernetes)
- Cloud Services (AWS, Azure, GCP)
- Database Audit Logs
- IoT Device Logs
- Network Device Logs

### Analytics & Detection
- Real-time log analysis
- Anomaly detection
- Threat correlation
- Behavioral analysis
- Custom rule engine
- Machine learning models

### Monitoring & Alerting
- Real-time dashboards
- Custom alert rules
- Email/SMS notifications
- Incident management
- Compliance reporting
- Performance metrics

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- 📧 Email: support@athala-siem.com
- 💬 Discord: [Join our community](https://discord.gg/athala-siem)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/AthalaSIEM/issues)

---

<p align="center">
  Made with ❤️ by the AthalaSIEM Team
</p>