# AthalaSIEM Linux Agent Deployment Guide

## Overview
The AthalaSIEM Linux Agent provides comprehensive security monitoring and log collection for Linux environments, supporting multiple distributions and deployment methods.

## System Requirements
- **Operating System**: Linux (Ubuntu, CentOS, RHEL, Debian, SUSE, Arch Linux)
- **.NET Runtime**: .NET 8.0 or later
- **Architecture**: x64, ARM64
- **Memory**: Minimum 512MB RAM
- **Disk**: 100MB free space (plus archive storage)
- **Network**: HTTP/HTTPS connectivity to SIEM backend
- **Privileges**: Root access for full functionality

## Features
### Log Collection
- **Syslog**: RFC 3164, RFC 5424, CEF, JSON, Key-Value parsing
- **Systemd Journal**: Real-time journal monitoring
- **File Integrity Monitoring (FIM)**: inotify-based file monitoring
- **System Metrics**: CPU, Memory, Disk I/O, Network, Process monitoring

### Communication
- **HTTP REST API**: Primary communication method
- **Batch Processing**: Configurable batch sizes
- **Compression**: Gzip compression for log transmission
- **Retry Logic**: Automatic retry with exponential backoff

## Installation Methods

### Method 1: Shell Script Deployment
```bash
# Download and run deployment script
curl -sSL https://your-siem-server/deploy/linux-deployment.sh | sudo bash

# Or download and run manually
wget https://your-siem-server/deploy/linux-deployment.sh
chmod +x linux-deployment.sh
sudo ./linux-deployment.sh
```

### Method 2: Package Installation

#### Debian/Ubuntu (.deb)
```bash
# Install package
sudo dpkg -i athala-siem-agent_1.0.0_amd64.deb
sudo apt-get install -f  # Fix dependencies if needed

# Start service
sudo systemctl enable athala-siem-agent
sudo systemctl start athala-siem-agent
```

#### RHEL/CentOS (.rpm)
```bash
# Install package
sudo rpm -ivh athala-siem-agent-1.0.0-1.x86_64.rpm

# Start service
sudo systemctl enable athala-siem-agent
sudo systemctl start athala-siem-agent
```

### Method 3: Manual Installation
```bash
# Create user and directories
sudo useradd -r -s /bin/false athala-siem
sudo mkdir -p /opt/athala-siem-agent
sudo mkdir -p /var/log/athala-siem-agent
sudo mkdir -p /etc/athala-siem-agent

# Extract agent files
sudo tar -xzf athala-siem-agent-linux.tar.gz -C /opt/athala-siem-agent/

# Set permissions
sudo chown -R athala-siem:athala-siem /opt/athala-siem-agent
sudo chown -R athala-siem:athala-siem /var/log/athala-siem-agent
sudo chmod +x /opt/athala-siem-agent/AthalaSIEM.UniversalAgent

# Install systemd service
sudo cp athala-siem-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable athala-siem-agent
```

## Configuration

### Environment Variables
The Linux agent supports configuration via environment variables:

```bash
# Core Configuration
export ATHALA_SIEM_MANAGER_IP="192.168.1.100"
export ATHALA_SIEM_MANAGER_PORT="9595"
export ATHALA_SIEM_USE_HTTPS="false"
export ATHALA_SIEM_DEPLOYMENT_TOKEN="athala-siem-agent-registration-2025"

# Agent Identity
export ATHALA_SIEM_AGENT_NAME="web-server-01"
export ATHALA_SIEM_AGENT_HOSTNAME="$(hostname)"
export ATHALA_SIEM_AGENT_DESCRIPTION="Production Web Server"

# System Metrics Collection
export ATHALA_SIEM_METRICS_ENABLED="true"
export ATHALA_SIEM_METRICS_INTERVAL="60"  # seconds
export ATHALA_SIEM_CPU_THRESHOLD="80.0"   # percentage
export ATHALA_SIEM_MEMORY_THRESHOLD="85.0" # percentage
export ATHALA_SIEM_DISK_THRESHOLD="90.0"   # percentage

# Syslog Configuration
export ATHALA_SIEM_SYSLOG_ENABLED="true"
export ATHALA_SIEM_SYSLOG_PATHS="/var/log/syslog,/var/log/messages,/var/log/auth.log"
export ATHALA_SIEM_JOURNAL_ENABLED="true"

# File Integrity Monitoring
export ATHALA_SIEM_FIM_ENABLED="true"
export ATHALA_SIEM_FIM_PATHS="/etc,/usr/bin,/usr/sbin,/home"
export ATHALA_SIEM_FIM_RECURSIVE="true"
export ATHALA_SIEM_FIM_HASH_ALGORITHMS="SHA256,MD5"

# Batch Processing
export ATHALA_SIEM_BATCH_SIZE="100"
export ATHALA_SIEM_BATCH_INTERVAL="30"  # seconds
export ATHALA_SIEM_MAX_QUEUE_SIZE="10000"

# Logging
export ATHALA_SIEM_LOG_LEVEL="Information"
export ATHALA_SIEM_LOG_PATH="/var/log/athala-siem-agent/agent.log"
```

### Configuration File
Alternative to environment variables, create `/etc/athala-siem-agent/appsettings.json`:

```json
{
  "SiemManager": {
    "ManagerIP": "192.168.1.100",
    "ManagerPort": 9595,
    "UseHTTPS": false
  },
  "Agent": {
    "Name": "web-server-01",
    "Hostname": "web-server-01.company.com",
    "Description": "Production Web Server",
    "DeploymentToken": "athala-siem-agent-registration-2025"
  },
  "LinuxSystemMetrics": {
    "Enabled": true,
    "IntervalSeconds": 60,
    "CPUThreshold": 80.0,
    "MemoryThreshold": 85.0,
    "DiskThreshold": 90.0
  },
  "LinuxSyslog": {
    "Enabled": true,
    "LogPaths": ["/var/log/syslog", "/var/log/messages", "/var/log/auth.log"],
    "JournalEnabled": true
  },
  "LinuxFIM": {
    "Enabled": true,
    "WatchPaths": ["/etc", "/usr/bin", "/usr/sbin", "/home"],
    "Recursive": true,
    "HashAlgorithms": ["SHA256", "MD5"]
  },
  "Communication": {
    "BatchSize": 100,
    "BatchIntervalSeconds": 30,
    "MaxQueueSize": 10000
  },
  "Logging": {
    "LogLevel": {
      "Default": "Information"
    }
  }
}
```

## Service Management

### SystemD Commands
```bash
# Start/Stop/Restart
sudo systemctl start athala-siem-agent
sudo systemctl stop athala-siem-agent
sudo systemctl restart athala-siem-agent

# Enable/Disable auto-start
sudo systemctl enable athala-siem-agent
sudo systemctl disable athala-siem-agent

# Check status
sudo systemctl status athala-siem-agent

# View logs
sudo journalctl -u athala-siem-agent -f
sudo journalctl -u athala-siem-agent --since "1 hour ago"
```

### Log Files
- **Agent Logs**: `/var/log/athala-siem-agent/agent.log`
- **System Journal**: `journalctl -u athala-siem-agent`

## Monitoring & Troubleshooting

### Health Checks
```bash
# Check agent status
curl -s http://localhost:8080/health || echo "Agent not responding"

# Check backend connectivity
curl -s http://your-siem-server:9595/api/health || echo "Backend not reachable"

# Verify log collection
sudo tail -f /var/log/athala-siem-agent/agent.log | grep "logs sent"
```

### Common Issues

#### 1. Permission Denied
```bash
# Fix file permissions
sudo chown -R athala-siem:athala-siem /opt/athala-siem-agent
sudo chmod +x /opt/athala-siem-agent/AthalaSIEM.UniversalAgent
```

#### 2. Network Connectivity
```bash
# Test backend connectivity
telnet your-siem-server 9595

# Check firewall rules
sudo iptables -L | grep 9595
sudo firewall-cmd --list-ports
```

#### 3. Missing Dependencies
```bash
# Install .NET runtime
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt update
sudo apt install dotnet-runtime-8.0
```

#### 4. Syslog Access
```bash
# Add user to required groups
sudo usermod -a -G adm,syslog athala-siem

# Check log file permissions
ls -la /var/log/syslog /var/log/messages
```

#### 5. inotify Limits
```bash
# Increase inotify limits for FIM
echo 'fs.inotify.max_user_watches=524288' | sudo tee -a /etc/sysctl.conf
echo 'fs.inotify.max_user_instances=512' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Security Considerations

### File Permissions
- Agent binary: `755` (athala-siem:athala-siem)
- Configuration files: `600` (athala-siem:athala-siem)
- Log directories: `750` (athala-siem:athala-siem)

### Network Security
- Use HTTPS in production environments
- Implement proper firewall rules
- Consider VPN or private networks for agent-backend communication

### Deployment Token
- Use unique tokens per environment
- Rotate tokens regularly
- Store tokens securely (environment variables, not in code)

## Performance Tuning

### Resource Optimization
```bash
# Adjust batch sizes based on load
export ATHALA_SIEM_BATCH_SIZE="50"  # Reduce for low-resource systems
export ATHALA_SIEM_BATCH_SIZE="200" # Increase for high-throughput systems

# Adjust collection intervals
export ATHALA_SIEM_METRICS_INTERVAL="120"  # Less frequent metrics
export ATHALA_SIEM_BATCH_INTERVAL="60"     # Less frequent batching
```

### Memory Management
```bash
# Monitor memory usage
ps aux | grep AthalaSIEM
top -p $(pgrep -f AthalaSIEM)

# Adjust .NET garbage collection
export DOTNET_gcServer=1
export DOTNET_GCRetainVM=1
```

## Multi-Agent Deployment

### Ansible Playbook Example
```yaml
---
- hosts: linux_servers
  become: yes
  vars:
    siem_server: "192.168.1.100"
    siem_port: 9595
    deployment_token: "athala-siem-agent-registration-2025"
  
  tasks:
    - name: Download AthalaSIEM Agent
      get_url:
        url: "http://{{ siem_server }}:{{ siem_port }}/deploy/athala-siem-agent-linux.tar.gz"
        dest: "/tmp/athala-siem-agent-linux.tar.gz"
    
    - name: Install AthalaSIEM Agent
      shell: |
        export ATHALA_SIEM_MANAGER_IP="{{ siem_server }}"
        export ATHALA_SIEM_MANAGER_PORT="{{ siem_port }}"
        export ATHALA_SIEM_DEPLOYMENT_TOKEN="{{ deployment_token }}"
        export ATHALA_SIEM_AGENT_NAME="{{ inventory_hostname }}"
        curl -sSL http://{{ siem_server }}:{{ siem_port }}/deploy/linux-deployment.sh | bash
```

## Support & Maintenance

### Log Rotation
```bash
# Configure logrotate
sudo cat > /etc/logrotate.d/athala-siem-agent << EOF
/var/log/athala-siem-agent/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

### Updates
```bash
# Update via package manager
sudo apt update && sudo apt upgrade athala-siem-agent  # Debian/Ubuntu
sudo yum update athala-siem-agent                       # RHEL/CentOS

# Manual update
sudo systemctl stop athala-siem-agent
# Replace binaries
sudo systemctl start athala-siem-agent
```

## Integration Examples

### Docker Deployment
```dockerfile
FROM mcr.microsoft.com/dotnet/runtime:8.0-alpine
COPY athala-siem-agent /app/
WORKDIR /app
ENV ATHALA_SIEM_MANAGER_IP=host.docker.internal
ENV ATHALA_SIEM_MANAGER_PORT=9595
ENTRYPOINT ["./AthalaSIEM.UniversalAgent"]
```

### Kubernetes DaemonSet
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: athala-siem-agent
spec:
  selector:
    matchLabels:
      app: athala-siem-agent
  template:
    metadata:
      labels:
        app: athala-siem-agent
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: athala-siem-agent
        image: athala-siem-agent:latest
        env:
        - name: ATHALA_SIEM_MANAGER_IP
          value: "siem-backend.monitoring.svc.cluster.local"
        - name: ATHALA_SIEM_AGENT_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        volumeMounts:
        - name: var-log
          mountPath: /var/log
          readOnly: true
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
      volumes:
      - name: var-log
        hostPath:
          path: /var/log
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
```

---

## Summary: Revian Ravil Athala

The AthalaSIEM Linux Agent provides enterprise-grade security monitoring with:
- **Multi-format log parsing** (Syslog, Journal, CEF, JSON)
- **Real-time file integrity monitoring** via inotify
- **Comprehensive system metrics** collection
- **Flexible deployment options** (packages, containers, scripts)
- **Production-ready architecture** with batching, compression, and retry logic

For support or questions, contact: Revian Ravil Athala
