# Enhanced File Integrity Monitoring (FIM) - AthalaSIEM

## Overview
Enhanced File Integrity Monitoring system that provides comprehensive file system change detection with batch processing, critical path monitoring, and advanced performance optimization features.

## 🚀 Features

### Core Features
- ✅ Real-time file system monitoring
- ✅ Batch processing for optimal performance
- ✅ Critical path immediate alerting
- ✅ Configurable exclude patterns
- ✅ SHA256 hash-based integrity verification
- ✅ Comprehensive API endpoints
- ✅ Modern responsive UI
- ✅ Database-backed event storage

### Advanced Features
- 🔥 **Batch Processing**: Efficient event batching to reduce system load
- 🔥 **Critical Path Monitoring**: Immediate alerts for critical system files
- 🔥 **Performance Optimization**: Configurable buffer sizes and scan intervals
- 🔥 **Detailed Logging**: Optional verbose logging for troubleshooting
- 🔥 **Overflow Protection**: Automatic buffer management to prevent memory issues

## 📋 Configuration

### Agent Configuration (appsettings.json)

```json
{
  "Type": "FileIntegrity",
  "Enabled": true,
  "Properties": {
    "MonitoredPaths": "C:\\Windows\\System32,C:\\Program Files,C:\\Program Files (x86),C:\\inetpub",
    "ExcludePatterns": "*.tmp,*.log,*.swp,*.lock,*~,*.bak,thumbs.db",
    "CriticalPaths": "C:\\Windows\\System32\\drivers,C:\\Windows\\System32\\config,C:\\inetpub\\wwwroot",
    "RealTimeMonitoring": "true",
    "ScanIntervalMinutes": "60",
    "MaxEventsPerBatch": "50",
    "BatchIntervalSeconds": "10",
    "MaxBufferSize": "1000",
    "EnableDetailedLogging": "false",
    "EnablePerformanceOptimization": "true"
  }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MonitoredPaths` | string | **Required** | Comma-separated list of paths to monitor |
| `ExcludePatterns` | string | Optional | File patterns to exclude from monitoring |
| `CriticalPaths` | string | Optional | Paths requiring immediate alerting |
| `RealTimeMonitoring` | bool | true | Enable real-time file system watching |
| `ScanIntervalMinutes` | int | 60 | Full scan interval in minutes |
| `MaxEventsPerBatch` | int | 50 | Maximum events per batch |
| `BatchIntervalSeconds` | int | 10 | Batch processing interval |
| `MaxBufferSize` | int | 1000 | Maximum buffer size before overflow protection |
| `EnableDetailedLogging` | bool | false | Enable verbose logging |
| `EnablePerformanceOptimization` | bool | true | Enable performance optimizations |

## 🔧 Installation Steps

### 1. Database Migration
```sql
-- Run the migration script
.\backend\Scripts\CreateFimMigration.sql
```

### 2. Backend Setup
```bash
# Add FIM entities to ApplicationDbContext (already configured)
# Run Entity Framework migration
dotnet ef migrations add AddFileIntegrityTables
dotnet ef database update
```

### 3. Agent Configuration
Update `agent/appsettings.json` with your specific monitoring requirements:

```json
{
  "MonitoredPaths": "YOUR_PATHS_HERE",
  "CriticalPaths": "YOUR_CRITICAL_PATHS_HERE"
}
```

### 4. Frontend Access
Navigate to the FIM interface:
```
https://localhost:3000/siem/file-integrity
```

## 🎯 Usage Examples

### Basic Monitoring Setup
```json
{
  "MonitoredPaths": "C:\\Windows\\System32,C:\\Program Files",
  "ExcludePatterns": "*.tmp,*.log",
  "RealTimeMonitoring": "true",
  "ScanIntervalMinutes": "60"
}
```

### High-Performance Setup
```json
{
  "MonitoredPaths": "C:\\inetpub\\wwwroot,C:\\app\\data",
  "MaxEventsPerBatch": "100",
  "BatchIntervalSeconds": "5",
  "MaxBufferSize": "2000",
  "EnablePerformanceOptimization": "true"
}
```

### Critical Infrastructure Monitoring
```json
{
  "MonitoredPaths": "C:\\Windows\\System32,C:\\Program Files",
  "CriticalPaths": "C:\\Windows\\System32\\drivers,C:\\Windows\\System32\\config",
  "MaxEventsPerBatch": "25",
  "BatchIntervalSeconds": "5"
}
```

## 📊 API Endpoints

### Events
- `GET /api/fileintegrity/events` - Get paginated FIM events
- `GET /api/fileintegrity/events/{id}` - Get specific event
- `POST /api/fileintegrity/events/acknowledge` - Acknowledge events

### Rules
- `GET /api/fileintegrity/rules` - Get FIM rules
- `POST /api/fileintegrity/rules` - Create new rule
- `PUT /api/fileintegrity/rules/{id}` - Update rule
- `DELETE /api/fileintegrity/rules/{id}` - Delete rule

### Statistics
- `GET /api/fileintegrity/statistics` - Get FIM metrics

## 🔍 Monitoring & Troubleshooting

### Performance Monitoring
Monitor these metrics in your logs:
- Batch processing times
- Buffer overflow warnings
- Event processing rates
- Critical path alerts

### Common Issues

#### High Memory Usage
```json
{
  "MaxBufferSize": "500",
  "BatchIntervalSeconds": "5",
  "MaxEventsPerBatch": "25"
}
```

#### Missing Events
```json
{
  "EnableDetailedLogging": "true",
  "RealTimeMonitoring": "true"
}
```

#### Too Many False Positives
```json
{
  "ExcludePatterns": "*.tmp,*.log,*.swp,*.lock,*~,*.bak,thumbs.db,*.dll.log,*.cache"
}
```

## 📈 Future Enhancements

### Planned Features
- [ ] Machine Learning anomaly detection
- [ ] Compliance report generation (PCI DSS, HIPAA)
- [ ] Real-time notifications via WebSocket
- [ ] Mobile-responsive UI improvements
- [ ] PDF/CSV export capabilities
- [ ] Agent auto-update mechanism
- [ ] Advanced charting with Chart.js/Recharts

### Configuration Templates

#### Web Server Monitoring
```json
{
  "MonitoredPaths": "C:\\inetpub\\wwwroot,C:\\Program Files\\IIS",
  "CriticalPaths": "C:\\inetpub\\wwwroot",
  "ExcludePatterns": "*.log,*.tmp,*access.log*",
  "ScanIntervalMinutes": "30"
}
```

#### Database Server Monitoring
```json
{
  "MonitoredPaths": "C:\\Program Files\\Microsoft SQL Server,C:\\Database\\Data",
  "CriticalPaths": "C:\\Database\\Data",
  "ExcludePatterns": "*.ldf,*.tmp,*.bak",
  "ScanIntervalMinutes": "15"
}
```

#### Development Environment
```json
{
  "MonitoredPaths": "C:\\Projects\\Production,C:\\Config",
  "EnableDetailedLogging": "true",
  "ScanIntervalMinutes": "120"
}
```

## 🎨 UI Features

### Dashboard
- Real-time event counts
- Severity distribution charts
- Agent status overview
- Acknowledgment statistics

### Event Management
- Paginated event listing
- Advanced filtering (severity, type, agent, status)
- Bulk acknowledgment
- Event details modal

### Rule Management
- Visual rule configuration
- Enable/disable rules
- Path validation
- Target agent selection

## 🔐 Security Considerations

- File hash verification using SHA256
- Configurable critical path monitoring
- Immediate alerting for high-risk changes
- Event acknowledgment tracking
- Audit trail maintenance

## 📞 Support

For technical support or feature requests, refer to the main AthalaSIEM documentation or create an issue in the project repository.

---

**Note**: This enhanced FIM system is designed for production use with enterprise-grade performance and security features. Always test configurations in a development environment before deployment. 