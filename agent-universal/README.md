# 🛡️ AthalaSIEM Universal Agent

**Enterprise-grade, cross-platform SIEM log collection agent**

[![.NET](https://img.shields.io/badge/.NET-8.0-purple)](https://dotnet.microsoft.com/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE.rtf)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Management](#management)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## 📖 Overview

The AthalaSIEM Universal Agent is a production-grade, cross-platform log collection agent designed for enterprise SIEM deployments. It collects security logs from Windows Event Logs, Linux syslog, containers, network devices, and cloud services, then forwards them to the AthalaSIEM backend for analysis and alerting.

### Key Capabilities

- **Multi-Platform**: Windows (Service/Console), Linux (systemd/Console), Docker containers
- **High Performance**: 1000+ events/second, efficient batching, queue management
- **Secure Communication**: gRPC and HTTP with JWT authentication
- **Intelligent Processing**: Pipeline-based architecture (Collector → Parser → Normalizer → Exporter)
- **Self-Contained**: Single-file deployment, minimal dependencies
- **Enterprise Ready**: MSI installer, Windows Service, systemd integration

## ✨ Features

### Log Collection
- ✅ Windows Event Log (Security, System, Application, Sysmon)
- ✅ Linux syslog (journalctl, /var/log/*)
- ✅ Docker container events
- ✅ Network device logs (firewalls, routers, switches)
- ✅ File Integrity Monitoring (FIM)
- ✅ Windows Registry monitoring
- ✅ Cloud service logs (AWS, Azure, GCP)

### Processing Pipeline
- ✅ **Collector**: Raw log acquisition (platform-specific)
- ✅ **Parser**: Structured log parsing (Windows, Syslog, Docker, Network)
- ✅ **Normalizer**: Athala ECS-lite schema mapping
- ✅ **Buffer**: Memory/disk buffering with backpressure handling
- ✅ **Exporter**: HTTP, gRPC, File, Console output

### Enterprise Features
- ✅ Agent registration and authentication
- ✅ Centralized configuration from backend
- ✅ Health monitoring and heartbeat
- ✅ Automatic reconnection and retry logic
- ✅ Log compression and batching
- ✅ Performance metrics reporting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Log Sources                              │
│  Windows Event Log | Linux Syslog | Docker | Network       │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Collector Layer                          │
│  WindowsEventLogCollector | SyslogCollector | etc.          │
└──────────────────────┬─────────────────────────────────────┘
                       │ RawEvent
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Parser Layer                              │
│  WindowsEventLogParser | SyslogParser | DockerParser         │
└──────────────────────┬─────────────────────────────────────┘
                       │ ParsedEvent
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Normalizer Layer                          │
│  AthalaEcsLiteNormalizer (ECS-lite schema mapping)          │
└──────────────────────┬─────────────────────────────────────┘
                       │ AthalaEcsLiteEvent
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Buffer Layer                              │
│  Memory Buffer → Disk Fallback (backpressure handling)      │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Exporter Layer                           │
│  HttpExporter | GrpcExporter | FileExporter | Console       │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  AthalaSIEM     │
              │  Backend        │
              └─────────────────┘
```

## 📋 Prerequisites

### Windows
- Windows 10/11 or Windows Server 2016+
- .NET 8.0 Runtime (or SDK for development)
- Administrator privileges (for Windows Event Log access)
- Network connectivity to AthalaSIEM backend

### Linux
- Ubuntu 18.04+, CentOS 7+, RHEL 7+, Debian 10+
- .NET 8.0 Runtime (or SDK for development)
- systemd (for service mode)
- Network connectivity to AthalaSIEM backend

### Docker
- Docker 20.10+ or compatible container runtime
- Network access to AthalaSIEM backend

## 🚀 Quick Start

### Windows (MSI Installer)

```powershell
# 1. Download MSI installer
# 2. Run installer (or use silent install)
.\AthalaSIEM-UniversalAgent-1.0.0-x64.msi

# 3. Configure (edit appsettings.json)
# Location: C:\Program Files\AthalaSIEM\UniversalAgent\appsettings.json

# 4. Start service
Start-Service AthalaSIEMUniversalAgent

# 5. Check status
Get-Service AthalaSIEMUniversalAgent
```

### Linux (systemd)

```bash
# 1. Download and extract
wget https://github.com/athalasiem/agent/releases/latest/download/athala-agent-linux-x64.tar.gz
sudo tar -xzf athala-agent-linux-x64.tar.gz -C /opt/athala-agent

# 2. Install systemd service
sudo cp /opt/athala-agent/athala-siem-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable athala-siem-agent

# 3. Configure
sudo nano /opt/athala-agent/appsettings.json

# 4. Start service
sudo systemctl start athala-siem-agent

# 5. Check status
sudo systemctl status athala-siem-agent
```

### Docker

```bash
# 1. Pull image
docker pull athalasiem/agent:latest

# 2. Run container
docker run -d \
  --name athala-agent \
  -v ./appsettings.json:/app/appsettings.json \
  -e BACKEND_URL=http://your-backend:9595 \
  athalasiem/agent:latest

# 3. Check logs
docker logs -f athala-agent
```

## 📦 Installation

### Method 1: MSI Installer (Windows)

1. **Download** the latest MSI installer from releases
2. **Run** the installer (requires Administrator privileges)
3. **Configure** `appsettings.json` in installation directory
4. **Start** the Windows Service

**Silent Installation:**
```powershell
msiexec /i AthalaSIEM-UniversalAgent-1.0.0-x64.msi /quiet /norestart BACKEND_URL=http://your-backend:9595
```

### Method 2: PowerShell Deployment Script

```powershell
# Download deployment script
.\deploy-agent.ps1 -BackendUrl "http://your-backend:9595" -StartService

# With custom agent name
.\deploy-agent.ps1 -BackendUrl "http://your-backend:9595" -AgentName "Production-Agent-01" -StartService

# Silent installation
.\deploy-agent.ps1 -BackendUrl "http://your-backend:9595" -SilentInstall -StartService
```

### Method 3: Manual Installation

#### Windows

```powershell
# 1. Build or download agent
dotnet publish -c Release -r win-x64 --self-contained -o ./publish

# 2. Copy to installation directory
Copy-Item -Path ./publish/* -Destination "C:\Program Files\AthalaSIEM\UniversalAgent" -Recurse

# 3. Install as Windows Service
cd "C:\Program Files\AthalaSIEM\UniversalAgent"
.\athala-agent.exe install

# 4. Start service
Start-Service AthalaSIEMUniversalAgent
```

#### Linux

```bash
# 1. Build or download agent
dotnet publish -c Release -r linux-x64 --self-contained -o ./publish

# 2. Copy to installation directory
sudo mkdir -p /opt/athala-agent
sudo cp -r ./publish/* /opt/athala-agent/

# 3. Install systemd service
sudo cp athala-siem-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable athala-siem-agent

# 4. Start service
sudo systemctl start athala-siem-agent
```

## ⚙️ Configuration

### appsettings.json

The agent configuration is managed via `appsettings.json`:

```json
{
  "SiemManager": {
    "ManagerIP": "192.168.1.100",
    "ManagerPort": 9595,
    "UseHTTPS": false,
    "AutoDiscovery": true
  },
  "Agent": {
    "Id": "AGENT-001",
    "Name": "Production-Agent-01",
    "ManagerUrl": "http://192.168.1.100:9595",
    "DeploymentToken": "your-deployment-token",
    "AutoRegister": true,
    "BatchSize": 500,
    "BatchIntervalSeconds": 30,
    "HeartbeatIntervalSeconds": 60
  },
  "Collectors": [
    {
      "Type": "WindowsEventLog",
      "Enabled": true,
      "Properties": {
        "LogSources": ["Security", "System", "Application"],
        "CollectAllEvents": true
      }
    }
  ]
}
```

### Environment Variables

Override configuration via environment variables:

```bash
# Backend URL
export ATHALA_SiemManager__ManagerIP=192.168.1.100
export ATHALA_SiemManager__ManagerPort=9595

# Agent Settings
export ATHALA_Agent__Name=Production-Agent-01
export ATHALA_Agent__DeploymentToken=your-token

# Logging
export ATHALA_Logging__LogLevel__Default=Information
```

### Configuration Sources

The agent supports multiple configuration sources (priority order):

1. **Backend Configuration** (highest priority)
   - Fetched from backend on startup
   - Refreshed periodically
   - Overrides local configuration

2. **Environment Variables**
   - Override specific settings
   - Useful for containerized deployments

3. **appsettings.json**
   - Local configuration file
   - Fallback if backend unavailable

## 🚢 Deployment

### Windows Service Deployment

```powershell
# Install service
.\athala-agent.exe install

# Start service
Start-Service AthalaSIEMUniversalAgent

# Stop service
Stop-Service AthalaSIEMUniversalAgent

# Uninstall service
.\athala-agent.exe uninstall
```

### Linux systemd Deployment

```bash
# Enable service
sudo systemctl enable athala-siem-agent

# Start service
sudo systemctl start athala-siem-agent

# Check status
sudo systemctl status athala-siem-agent

# View logs
sudo journalctl -u athala-siem-agent -f
```

### Docker Deployment

```bash
# Build image
docker build -t athala-agent:latest .

# Run container
docker run -d \
  --name athala-agent \
  --restart unless-stopped \
  -v ./appsettings.json:/app/appsettings.json \
  -v ./logs:/app/logs \
  -e BACKEND_URL=http://backend:9595 \
  athala-agent:latest

# Docker Compose
docker-compose up -d
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: athala-agent
spec:
  selector:
    matchLabels:
      app: athala-agent
  template:
    metadata:
      labels:
        app: athala-agent
    spec:
      containers:
      - name: agent
        image: athalasiem/agent:latest
        env:
        - name: BACKEND_URL
          value: "http://athala-backend:9595"
        volumeMounts:
        - name: config
          mountPath: /app/appsettings.json
          subPath: appsettings.json
      volumes:
      - name: config
        configMap:
          name: athala-agent-config
```

### Mass Deployment (Enterprise)

#### PowerShell (Windows)

```powershell
$servers = @("server1", "server2", "server3")
$backendUrl = "http://siem-backend:9595"

foreach ($server in $servers) {
    Invoke-Command -ComputerName $server -ScriptBlock {
        param($url)
        .\deploy-agent.ps1 -BackendUrl $url -SilentInstall -StartService
    } -ArgumentList $backendUrl
}
```

#### Ansible (Linux)

```yaml
- name: Deploy AthalaSIEM Agent
  hosts: all
  tasks:
    - name: Download agent
      get_url:
        url: https://github.com/athalasiem/agent/releases/latest/download/athala-agent-linux-x64.tar.gz
        dest: /tmp/athala-agent.tar.gz
    
    - name: Extract agent
      unarchive:
        src: /tmp/athala-agent.tar.gz
        dest: /opt/athala-agent
    
    - name: Install systemd service
      copy:
        src: athala-siem-agent.service
        dest: /etc/systemd/system/
    
    - name: Configure agent
      template:
        src: appsettings.json.j2
        dest: /opt/athala-agent/appsettings.json
    
    - name: Start service
      systemd:
        name: athala-siem-agent
        enabled: yes
        state: started
```

## 🔧 Management

### Command Line Options

```bash
# Run in console mode (for testing)
./athala-agent --console

# Test connection to backend
./athala-agent --test-connection

# Show configuration
./athala-agent --config

# Show version
./athala-agent --version

# Run UAT tests
./athala-agent --run-uat
```

### Service Management

#### Windows

```powershell
# Check service status
Get-Service AthalaSIEMUniversalAgent

# View service logs
Get-EventLog -LogName Application -Source "AthalaSIEM Universal Agent" -Newest 50

# Restart service
Restart-Service AthalaSIEMUniversalAgent
```

#### Linux

```bash
# Check service status
sudo systemctl status athala-siem-agent

# View logs
sudo journalctl -u athala-siem-agent -f

# Restart service
sudo systemctl restart athala-siem-agent
```

### Health Monitoring

The agent reports health status to the backend:

- **Heartbeat**: Every 60 seconds (configurable)
- **Health Metrics**: CPU, memory, disk usage
- **Collection Status**: Active collectors, log counts
- **Connection Status**: Backend connectivity

View health status in the AthalaSIEM frontend dashboard.

## 🐛 Troubleshooting

### Agent Not Connecting to Backend

1. **Check network connectivity:**
   ```bash
   # Test backend connection
   curl http://your-backend:9595/api/health
   ```

2. **Verify configuration:**
   ```bash
   # Check appsettings.json
   cat appsettings.json | grep ManagerUrl
   ```

3. **Check firewall rules:**
   ```bash
   # Windows
   netsh advfirewall firewall show rule name=all | findstr 9595
   
   # Linux
   sudo ufw status | grep 9595
   ```

4. **View agent logs:**
   ```bash
   # Windows
   Get-Content "C:\Program Files\AthalaSIEM\UniversalAgent\logs\agent-*.log" -Tail 50
   
   # Linux
   sudo journalctl -u athala-siem-agent -n 50
   ```

### Agent Not Collecting Logs

1. **Check collector status:**
   - Verify collectors are enabled in `appsettings.json`
   - Check collector-specific logs

2. **Verify permissions:**
   ```bash
   # Windows - Check Event Log access
   wevtutil el | Select-String "Security"
   
   # Linux - Check syslog access
   ls -la /var/log/syslog
   ```

3. **Test collector manually:**
   ```bash
   # Run in console mode with verbose logging
   ./athala-agent --console --log-level Debug
   ```

### High Memory Usage

1. **Reduce batch size:**
   ```json
   {
     "Agent": {
       "BatchSize": 100,
       "BatchIntervalSeconds": 15
     }
   }
   ```

2. **Enable log compression:**
   ```json
   {
     "Security": {
       "EnableCompression": true
     }
   }
   ```

3. **Reduce queue size:**
   ```json
   {
     "Agent": {
       "MaxQueueSize": 10000
     }
   }
   ```

### Service Won't Start

1. **Check service logs:**
   ```bash
   # Windows
   Get-EventLog -LogName Application -Source "AthalaSIEM" -Newest 20
   
   # Linux
   sudo journalctl -u athala-siem-agent -n 20
   ```

2. **Verify .NET Runtime:**
   ```bash
   dotnet --version
   # Should show 8.0.x
   ```

3. **Check file permissions:**
   ```bash
   # Linux
   ls -la /opt/athala-agent/
   sudo chown -R athala-agent:athala-agent /opt/athala-agent/
   ```

## 🛠️ Development

### Prerequisites

- .NET 8.0 SDK
- Visual Studio 2022 or VS Code
- Git

### Building from Source

```bash
# Clone repository
git clone https://github.com/athalasiem/athalasiem.git
cd athalasiem/agent-universal

# Restore dependencies
dotnet restore

# Build
dotnet build -c Release

# Run tests
dotnet test

# Publish
dotnet publish -c Release -r win-x64 --self-contained -o ./publish
```

### Project Structure

```
agent-universal/
├── Collectors/          # Log collectors (Windows, Linux, Docker, etc.)
├── Core/                # Core pipeline components
│   ├── Collector/       # Collector interfaces and adapters
│   ├── Parser/          # Log parsers
│   ├── Normalizer/      # Event normalizers (ECS-lite)
│   ├── Exporter/        # Export handlers (HTTP, gRPC, File)
│   └── Pipeline/        # Pipeline orchestration
├── Services/            # Background services
├── Models/              # Data models
├── Protos/              # gRPC protocol definitions
├── Program.cs           # Entry point
└── appsettings.json     # Configuration
```

### Running in Development

```bash
# Run in console mode
dotnet run -- --console

# Run with specific configuration
dotnet run -- --config appsettings.Development.json

# Run UAT tests
dotnet run -- --run-uat
```

## 📚 Additional Resources

- [Installation Guide](INSTALLATION-GUIDE.md) - Detailed installation instructions
- [Deployment Guide](README-DEPLOYMENT.md) - Production deployment scenarios
- [Linux Deployment](docs/LINUX_DEPLOYMENT_GUIDE.md) - Linux-specific deployment
- [Architecture Documentation](source/README.md) - Technical architecture details

## 📄 License

Copyright © 2025 Athala Security Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**🎉 Ready for production!** Deploy enterprise-grade security monitoring in minutes.
