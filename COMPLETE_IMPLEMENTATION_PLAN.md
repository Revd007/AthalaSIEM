# AthalaSIEM Complete Implementation Plan

##  Completed Components

### Backend - Core SIEM Logic

#### 1. Enhanced ECS Normalization (`EnhancedECSLogNormalizer.cs`)
-  **REQUIRED Fields**: Every log has `timestamp`, `source_ip`, `event_type`, `severity`
-  **Event Categorization**: Authentication, Process, Network, File, Security, General
-  **Severity Mapping**: Critical=10, Error=7, Warning=4, Info=2, Debug=1
-  **IP Extraction**: From properties, message, or agent ID fallback
-  **Windows Event Log Support**: Event ID mapping (4624, 4625, 4688, etc.)
-  **Sysmon Support**: Event ID mapping (1, 3, 7, 11)

#### 2. Simple Rule Engine (`SimpleRuleEngine.cs`)
-  **Brute Force Detection**: 5+ failed logins from same IP in 5 minutes
-  **Credential Stuffing**: 10+ failed logins for different users from same IP
-  **Privilege Escalation**: Successful login after multiple failures
-  **Port Scanning**: 20+ connection attempts to different ports
-  **Suspicious Process Execution**: 10+ process creations in 2 minutes
-  **Configurable Rules**: Time windows, thresholds, grouping keys

#### 3. Correlation Worker (`CorrelationWorker.cs`)
-  **Background Processing**: Processes normalized logs automatically
-  **Alert Generation**: Creates alerts when correlation rules trigger
-  **Database Integration**: Queries related logs by correlation key

#### 4. New Controllers

**NormalizationController** (`/api/normalization`)
- `GET /statistics` - Normalization statistics and metrics
- `GET /normalized` - Query normalized logs with ECS fields

**CorrelationController** (`/api/correlation`)
- `GET /statistics` - Correlation statistics and rule breakdown
- `GET /rules` - List active correlation rules
- `POST /trigger/{logEntryId}` - Manually trigger correlation (Admin only)

### Frontend - Missing Components to Implement

#### 1. Normalization Dashboard (`/dashboard/normalization`)
**Features:**
- Real-time normalization statistics
- Event type distribution chart
- Severity distribution chart
- Normalization rate over time
- Search/filter normalized logs by ECS fields

**Components Needed:**
- `NormalizationStats.tsx` - Statistics cards
- `EventTypeChart.tsx` - Pie/bar chart for event types
- `SeverityChart.tsx` - Severity distribution
- `NormalizedLogsTable.tsx` - Table with ECS fields
- `NormalizationFilters.tsx` - Filter by event_type, source_ip, severity

#### 2. Correlation Dashboard (`/dashboard/correlation`)
**Features:**
- Active correlation rules list
- Correlation statistics
- Recent correlation alerts
- Rule performance metrics
- Manual correlation trigger

**Components Needed:**
- `CorrelationRulesList.tsx` - List of active rules
- `CorrelationStats.tsx` - Statistics cards
- `CorrelationAlerts.tsx` - Recent alerts from correlation
- `RulePerformanceChart.tsx` - Rule trigger frequency
- `TriggerCorrelationDialog.tsx` - Manual trigger UI

#### 3. Enhanced Log Viewer (`/dashboard/logs`)
**Features:**
- View normalized logs with ECS fields
- Filter by event_type, source_ip, severity
- Search across ECS fields
- Export normalized logs
- Link to correlation results

**Components Needed:**
- `NormalizedLogViewer.tsx` - Enhanced log viewer
- `ECSFilters.tsx` - Filter by ECS fields
- `LogDetailsModal.tsx` - Show full ECS fields
- `CorrelationLink.tsx` - Link to correlation results

#### 4. Alert Management Enhancements (`/dashboard/alerts`)
**Features:**
- Filter alerts by correlation source
- View correlation metadata
- Link to correlated logs
- Correlation rule information

**Components Needed:**
- `CorrelationAlertBadge.tsx` - Badge for correlation alerts
- `CorrelationMetadata.tsx` - Show correlation metadata
- `CorrelatedLogsLink.tsx` - Link to correlated logs

##  Implementation Checklist

### Backend ( Complete)
- [x] Enhanced ECS Normalizer
- [x] Simple Rule Engine
- [x] Correlation Worker
- [x] NormalizationController
- [x] CorrelationController
- [x] Integration with AlertService
- [x] Database queries for normalized logs

### Frontend (🔄 To Implement)

#### Phase 1: Normalization Dashboard
- [ ] Create `/dashboard/normalization` page
- [ ] Implement `NormalizationStats` component
- [ ] Implement `EventTypeChart` component
- [ ] Implement `SeverityChart` component
- [ ] Implement `NormalizedLogsTable` component
- [ ] Implement `NormalizationFilters` component
- [ ] Add API integration for `/api/normalization/*`

#### Phase 2: Correlation Dashboard
- [ ] Create `/dashboard/correlation` page
- [ ] Implement `CorrelationRulesList` component
- [ ] Implement `CorrelationStats` component
- [ ] Implement `CorrelationAlerts` component
- [ ] Implement `RulePerformanceChart` component
- [ ] Implement `TriggerCorrelationDialog` component
- [ ] Add API integration for `/api/correlation/*`

#### Phase 3: Enhanced Log Viewer
- [ ] Enhance existing log viewer with ECS fields
- [ ] Add ECS field filters
- [ ] Add correlation result links
- [ ] Add export functionality for normalized logs

#### Phase 4: Alert Enhancements
- [ ] Add correlation badge to alerts
- [ ] Show correlation metadata in alert details
- [ ] Add link to correlated logs
- [ ] Filter alerts by correlation source

## 🔧 API Endpoints Summary

### Normalization
```
GET  /api/normalization/statistics?startDate=&endDate=
GET  /api/normalization/normalized?page=&pageSize=&eventType=&sourceIp=&minSeverity=
```

### Correlation
```
GET  /api/correlation/statistics?startDate=&endDate=
GET  /api/correlation/rules
POST /api/correlation/trigger/{logEntryId}  [Admin only]
```

## 📊 Data Flow

```
1. Agent → Logs → Backend
   ↓
2. LogNormalizationWorker → EnhancedECSLogNormalizer
   - Ensures: timestamp, source_ip, event_type, severity
   ↓
3. Normalized Log → Database (NormalizedLogs table)
   ↓
4. CorrelationWorker → SimpleRuleEngine
   - Checks rules (Brute Force, Port Scan, etc.)
   - Queries related logs
   - Generates alerts
   ↓
5. Alert → Database (Alerts table)
   ↓
6. Frontend → Displays alerts, logs, statistics
```

## 🎯 Next Steps

1. **Fix Compilation Errors** (In Progress)
   - Add LogEntryIds and Metadata to CreateAlertDto 
   - Fix EventId type conversion 

2. **Implement Frontend Components** (Next)
   - Start with Normalization Dashboard
   - Then Correlation Dashboard
   - Then Enhanced Log Viewer
   - Finally Alert Enhancements

3. **Testing**
   - Test normalization with various log types
   - Test correlation rules with sample data
   - Test frontend components with real data

4. **Documentation**
   - API documentation
   - Component usage guide
   - Deployment guide
