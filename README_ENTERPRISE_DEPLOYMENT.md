# AthalaSIEM Enterprise Deployment Guide

## Overview
AthalaSIEM adalah solusi SIEM enterprise yang mendukung multi-platform dengan kemampuan threat intelligence yang comprehensive. Sistem ini dapat menerima log dari berbagai device termasuk Windows, Linux, FreeBSD, network devices, firewall, dan infrastruktur IT lainnya.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AthalaSIEM Enterprise Architecture                │
├─────────────────────────────────────────────────────────────────────┤
│  Frontend (React + Next.js)                                        │
│  ├── Dashboard Management    ├── Agent Deployment                   │
│  ├── Threat Intelligence     ├── Log Analysis                       │
│  ├── File Integrity         ├── Compliance Reports                 │
│  └── Alert Management       └── User Management                     │
├─────────────────────────────────────────────────────────────────────┤
│  Backend API (.NET Core)                                           │
│  ├── Agent Management       ├── Threat Intelligence Engine          │
│  ├── Log Processing         ├── Real-time Analytics                 │
│  ├── Alert Engine           ├── Compliance Engine                   │
│  └── Deployment Service     └── Report Generator                    │
├─────────────────────────────────────────────────────────────────────┤
│  Database Layer                                                     │
│  ├── SQL Server/PostgreSQL  ├── Time-Series DB (InfluxDB)          │
│  ├── Elasticsearch         └── Redis Cache                         │
├─────────────────────────────────────────────────────────────────────┤
│  Agents & Data Sources                                             │
│  ├── Windows Agents         ├── Network Devices (Cisco, Juniper)   │
│  ├── Linux/Unix Agents      ├── Firewalls (Fortinet, Palo Alto)    │
│  ├── FreeBSD Agents         ├── Load Balancers (F5, HAProxy)       │
│  ├── Container Agents       ├── Cloud Services (AWS, Azure)        │
│  └── Syslog Receivers       └── IoT/Industrial Devices             │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start Deployment

### 1. Backend Deployment

```bash
# Clone repository
git clone https://github.com/your-org/AthalaSIEM.git
cd AthalaSIEM/backend

# Configure database connection
# Edit appsettings.json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=localhost;Database=AthalaSIEM;Trusted_Connection=true;",
    "Redis": "localhost:6379"
  }
}

# Run database migrations
dotnet ef database update

# Start backend service
dotnet run
```

### 2. Frontend Deployment

```bash
cd ../frontend

# Install dependencies
npm install

# Configure backend URL
# Edit .env.local
NEXT_PUBLIC_API_URL=https://your-backend-url

# Build and start
npm run build
npm start
```

### 3. Agent Deployment

## 📋 Multi-Platform Agent Deployment

### Windows Agent Deployment

#### Automated Deployment (PowerShell)
```powershell
# Download and run deployment script
Invoke-WebRequest -Uri "https://your-siem-server/api/agentdeployment/scripts/windows?tokenId=your-token" -OutFile "deploy-agent.ps1"
.\deploy-agent.ps1
```

#### Manual Deployment
```powershell
# Download MSI installer
$installerUrl = "https://your-siem-server/downloads/agent/windows/athala-siem-agent.msi"
$tempPath = "$env:TEMP\athala-siem-agent.msi"
Invoke-WebRequest -Uri $installerUrl -OutFile $tempPath

# Install with configuration
msiexec /i $tempPath /quiet BACKEND_URL="https://your-siem-server" DEPLOYMENT_TOKEN="your-token"

# Start service
Start-Service -Name "AthalaSIEMAgent"
```

#### Configuration
```json
{
  "Agent": {
    "Collectors": [
      {
        "Type": "WindowsEventLog",
        "Enabled": true,
        "Properties": {
          "LogNames": "System,Application,Security,Setup,Forwarded Events",
          "QueryFilter": "*[System[(Level=1 or Level=2 or Level=3 or Level=4)]]"
        }
      },
      {
        "Type": "FileIntegrity",
        "Enabled": true,
        "Properties": {
          "MonitoredPaths": "C:\\Windows\\System32,C:\\Program Files",
          "CriticalPaths": "C:\\Windows\\System32\\drivers,C:\\Windows\\System32\\config"
        }
      }
    ]
  }
}
```

### Linux Agent Deployment

#### Automated Deployment (Ubuntu/Debian)
```bash
#!/bin/bash
# Download and install agent
curl -L "https://your-siem-server/api/agentdeployment/scripts/linux?tokenId=your-token" | sudo bash
```

#### Manual Deployment
```bash
# Download package
wget https://your-siem-server/downloads/agent/linux/athala-siem-agent.deb

# Install
sudo dpkg -i athala-siem-agent.deb
sudo apt-get install -f -y

# Configure
sudo tee /etc/athala-siem/agent.conf << EOF
backend_url=https://your-siem-server
deployment_token=your-token
EOF

# Start service
sudo systemctl enable athala-siem-agent
sudo systemctl start athala-siem-agent
```

#### Configuration
```json
{
  "Agent": {
    "Collectors": [
      {
        "Type": "Syslog",
        "Enabled": true,
        "Properties": {
          "UdpPort": "514",
          "TcpPort": "601",
          "BindAddress": "0.0.0.0"
        }
      },
      {
        "Type": "FileIntegrity",
        "Enabled": true,
        "Properties": {
          "MonitoredPaths": "/etc,/bin,/sbin,/usr/bin",
          "CriticalPaths": "/etc/passwd,/etc/shadow,/etc/hosts"
        }
      }
    ]
  }
}
```

### FreeBSD Agent Deployment

#### Automated Deployment
```bash
#!/bin/sh
# Download and install agent
fetch -o - "https://your-siem-server/api/agentdeployment/scripts/freebsd?tokenId=your-token" | sudo sh
```

#### Manual Deployment
```bash
# Install dependencies
sudo pkg install -y curl

# Download and install
fetch https://your-siem-server/downloads/agent/freebsd/athala-siem-agent.txz
sudo pkg add athala-siem-agent.txz

# Configure
sudo mkdir -p /usr/local/etc/athala-siem
sudo tee /usr/local/etc/athala-siem/agent.conf << EOF
backend_url=https://your-siem-server
deployment_token=your-token
EOF

# Enable and start service
sudo sysrc athala_siem_agent_enable="YES"
sudo service athala-siem-agent start
```

### Container Deployment (Docker)

#### Docker Compose
```yaml
version: '3.8'

services:
  athala-siem-agent:
    image: athala/siem-agent:latest
    container_name: athala-siem-agent
    restart: unless-stopped
    environment:
      - BACKEND_URL=https://your-siem-server
      - DEPLOYMENT_TOKEN=your-token
      - AGENT_NAME={{.Node.Hostname}}
    volumes:
      - /var/log:/host/var/log:ro
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    network_mode: host
    privileged: true
    pid: host
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: athala-siem-agent
  namespace: athala-siem
spec:
  selector:
    matchLabels:
      app: athala-siem-agent
  template:
    metadata:
      labels:
        app: athala-siem-agent
    spec:
      serviceAccountName: athala-siem-agent
      containers:
      - name: agent
        image: athala/siem-agent:latest
        env:
        - name: BACKEND_URL
          value: "https://your-siem-server"
        - name: DEPLOYMENT_TOKEN
          valueFrom:
            secretKeyRef:
              name: athala-siem-secrets
              key: deployment-token
        volumeMounts:
        - name: varlog
          mountPath: /host/var/log
          readOnly: true
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
```

## 🌐 Network Device Integration

### Cisco Devices

#### Configuration
```cisco
# Enable syslog
configure terminal
logging on
logging buffered 64000
logging console warnings
logging monitor warnings
logging trap informational
logging facility local0

# Configure syslog server
logging host your-siem-server transport udp port 514

# Save configuration
copy running-config startup-config
```

### Juniper Devices

#### Configuration
```junos
# Configure syslog
set system syslog host your-siem-server any any
set system syslog host your-siem-server port 514
set system syslog host your-siem-server facility-override local0

# Commit configuration
commit
```

### Fortinet FortiGate

#### Configuration
```fortigate
# Configure log settings
config log setting
    set status enable
    set localid 1
end

# Configure syslog server
config log syslogd setting
    set status enable
    set server "your-siem-server"
    set port 514
    set facility local0
    set format default
end
```

### Palo Alto Networks

#### Configuration
```panos
# Configure syslog server
configure
set deviceconfig system server-profile syslog athala-siem
set deviceconfig system server-profile syslog athala-siem server your-siem-server
set deviceconfig system server-profile syslog athala-siem port 514
set deviceconfig system server-profile syslog athala-siem transport UDP
set deviceconfig system server-profile syslog athala-siem facility LOG_LOCAL0

# Configure log forwarding
set log-settings syslog athala-siem server your-siem-server port 514 facility LOG_LOCAL0

commit
```

### pfSense Firewall

#### Configuration
```php
# Navigate to Status > System Logs > Settings
# Configure Remote Logging:
Remote log servers: your-siem-server:514
Remote Syslog Contents: Everything

# Save settings
```

## 🔍 Threat Intelligence Configuration

### Supported Feed Types

#### 1. MISP Integration
```json
{
  "Name": "MISP Threat Feed",
  "Type": "MISP",
  "FeedUrl": "https://your-misp-instance.com/events/json",
  "ApiKey": "your-misp-api-key",
  "UpdateInterval": 180,
  "Priority": "Critical",
  "Categories": ["IP", "Domain", "Hash", "URL", "Email"]
}
```

#### 2. AlienVault OTX
```json
{
  "Name": "AlienVault OTX",
  "Type": "OTX",
  "FeedUrl": "https://otx.alienvault.com/api/v1/indicators/export",
  "ApiKey": "your-otx-api-key",
  "UpdateInterval": 360,
  "Priority": "High",
  "Categories": ["IP", "Domain", "Hash", "URL"]
}
```

#### 3. Custom JSON Feeds
```json
{
  "Name": "Custom Threat Feed",
  "Type": "JSON",
  "FeedUrl": "https://your-threat-feed.com/api/indicators",
  "ApiKey": "your-api-key",
  "UpdateInterval": 60,
  "Priority": "High",
  "Configuration": {
    "JsonPath": "$.indicators",
    "IndicatorField": "value",
    "TypeField": "type",
    "ContextField": "context"
  }
}
```

#### 4. STIX/TAXII Feeds
```json
{
  "Name": "STIX Feed",
  "Type": "STIX",
  "FeedUrl": "https://your-taxii-server.com/taxii2/collections/indicators/objects/",
  "Username": "taxii-user",
  "Password": "taxii-password",
  "UpdateInterval": 1440,
  "Priority": "Medium"
}
```

### Real-time Threat Matching

Sistem secara real-time akan mencocokkan log yang masuk dengan indicator yang tersedia:

```json
{
  "ThreatIntelligence": {
    "EnableRealTimeMatching": true,
    "MatchingFields": ["source_ip", "destination_ip", "domain", "url", "file_hash"],
    "EnrichmentServices": ["VirusTotal", "PassiveTotal", "IPGeolocation"],
    "CacheTimeout": 3600,
    "MaxCacheSize": 10000
  }
}
```

## 📊 Management Dashboard Features

### 1. Agent Management
- **Multi-platform agent monitoring**
- **Real-time status tracking**
- **Remote configuration management**
- **Deployment token management**
- **Performance metrics**

### 2. Threat Intelligence
- **Feed management dan monitoring**
- **Real-time indicator matching**
- **Threat campaign tracking**
- **MITRE ATT&CK technique mapping**
- **IoC enrichment dan analysis**

### 3. File Integrity Monitoring
- **Real-time file change detection**
- **Critical path monitoring**
- **Baseline management**
- **Compliance reporting**
- **Batch processing optimization**

### 4. Compliance & Reporting
- **PCI DSS compliance reporting**
- **HIPAA audit trails**
- **SOX compliance monitoring**
- **Custom report generation**
- **Automated PDF/CSV exports**

## 🔧 Advanced Configuration

### Load Balancing Configuration

#### NGINX Configuration
```nginx
upstream athala_siem_backend {
    server backend1.internal:9598;
    server backend2.internal:9598;
    server backend3.internal:9598;
}

server {
    listen 443 ssl http2;
    server_name siem.your-domain.com;

    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;

    location / {
        proxy_pass http://athala_siem_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /syslog {
        proxy_pass http://athala_siem_backend;
        proxy_timeout 60s;
        proxy_udp_timeout 60s;
    }
}

# Syslog UDP load balancing
stream {
    upstream syslog_servers {
        server backend1.internal:514;
        server backend2.internal:514;
        server backend3.internal:514;
    }

    server {
        listen 514 udp;
        proxy_pass syslog_servers;
        proxy_timeout 1s;
        proxy_responses 1;
    }
}
```

### High Availability Setup

#### Database Clustering
```sql
-- SQL Server Always On Configuration
ALTER AVAILABILITY GROUP [AthalaSIEM-AG]
ADD DATABASE [AthalaSIEM];

-- Enable backup on secondary
ALTER AVAILABILITY GROUP [AthalaSIEM-AG]
MODIFY REPLICA ON 'SQL-Secondary'
WITH (BACKUP_PRIORITY = 50);
```

#### Redis Cluster
```redis
# Redis Sentinel configuration
sentinel monitor athala-siem-master 192.168.1.10 6379 2
sentinel down-after-milliseconds athala-siem-master 5000
sentinel failover-timeout athala-siem-master 10000
sentinel parallel-syncs athala-siem-master 1
```

### Performance Tuning

#### Database Optimization
```sql
-- Index optimization for large log tables
CREATE INDEX IX_LogEntries_Timestamp_Level 
ON log_entries (Timestamp DESC, Level) 
INCLUDE (Message, Source, AgentId);

-- Partitioning by date
CREATE PARTITION FUNCTION PF_LogEntries_Date (datetime2)
AS RANGE RIGHT FOR VALUES 
('2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01');

CREATE PARTITION SCHEME PS_LogEntries_Date
AS PARTITION PF_LogEntries_Date ALL TO ([PRIMARY]);

-- Apply partition scheme
ALTER TABLE log_entries
ADD CONSTRAINT PK_LogEntries_Partitioned
PRIMARY KEY CLUSTERED (Id, Timestamp)
ON PS_LogEntries_Date(Timestamp);
```

#### Application Performance
```json
{
  "Performance": {
    "MaxConcurrentAgents": 1000,
    "LogProcessingBatchSize": 500,
    "DatabaseConnectionPoolSize": 50,
    "RedisConnectionPoolSize": 20,
    "ThreatIntelligenceWorkers": 10,
    "FileIntegrityWorkers": 5
  }
}
```

## 🔐 Security Hardening

### SSL/TLS Configuration
```json
{
  "Kestrel": {
    "Endpoints": {
      "HttpsInlineCertFile": {
        "Url": "https://*:9598",
        "Certificate": {
          "Path": "/etc/ssl/certs/athala-siem.pfx",
          "Password": "your-certificate-password"
        }
      }
    }
  }
}
```

### Authentication & Authorization
```json
{
  "Authentication": {
    "JwtSettings": {
      "SecretKey": "your-jwt-secret-key-32-characters-minimum",
      "Issuer": "AthalaSIEM",
      "Audience": "AthalaSIEM-Users",
      "ExpirationMinutes": 480
    },
    "LdapSettings": {
      "Enabled": true,
      "Server": "ldap://your-domain-controller.com",
      "BaseDN": "DC=company,DC=com",
      "UserSearchFilter": "(sAMAccountName={0})",
      "GroupSearchFilter": "(member={0})"
    }
  }
}
```

### API Rate Limiting
```json
{
  "RateLimiting": {
    "GeneralRules": [
      {
        "Endpoint": "*",
        "Period": "1m",
        "Limit": 1000
      }
    ],
    "SpecificRules": [
      {
        "Endpoint": "/api/logs",
        "Period": "1m", 
        "Limit": 10000
      }
    ]
  }
}
```

## 🔄 Maintenance & Monitoring

### Health Checks
```csharp
public void ConfigureServices(IServiceCollection services)
{
    services.AddHealthChecks()
        .AddSqlServer(connectionString)
        .AddRedis(redisConnectionString)
        .AddUrlGroup(new Uri("https://external-threat-feed.com"), "threat-feeds")
        .AddCheck<CustomHealthCheck>("athala-siem-custom");
}
```

### Monitoring Endpoints
- **Health**: `https://your-siem-server/health`
- **Metrics**: `https://your-siem-server/metrics`
- **Status**: `https://your-siem-server/api/system/status`

### Log Retention Policy
```sql
-- Automated log cleanup job
CREATE PROCEDURE CleanupOldLogs
AS
BEGIN
    -- Keep detailed logs for 90 days
    DELETE FROM log_entries 
    WHERE Timestamp < DATEADD(DAY, -90, GETUTCDATE());
    
    -- Keep summary data for 1 year
    DELETE FROM threat_matches 
    WHERE DetectedAt < DATEADD(DAY, -365, GETUTCDATE())
    AND IsAcknowledged = 1;
END
```

## 📞 Support & Troubleshooting

### Common Issues

#### Agent Connectivity
```bash
# Test agent connectivity
curl -k https://your-siem-server/api/health
telnet your-siem-server 514

# Check agent logs
tail -f /var/log/athala-siem/agent.log
Get-EventLog -LogName Application -Source AthalaSIEMAgent
```

#### Performance Issues
```sql
-- Check database performance
SELECT 
    object_name,
    counter_name,
    cntr_value
FROM sys.dm_os_performance_counters
WHERE counter_name IN ('Batch Requests/sec', 'SQL Compilations/sec');

-- Check index usage
SELECT 
    i.name AS IndexName,
    s.user_seeks,
    s.user_scans,
    s.user_lookups
FROM sys.dm_db_index_usage_stats s
JOIN sys.indexes i ON s.object_id = i.object_id AND s.index_id = i.index_id;
```

### Support Contacts
- **Technical Support**: support@athala-siem.com
- **Documentation**: https://docs.athala-siem.com
- **Community**: https://community.athala-siem.com

---

**AthalaSIEM Enterprise** - Comprehensive SIEM solution untuk infrastruktur IT modern dengan kemampuan multi-platform dan threat intelligence yang advanced. 