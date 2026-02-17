# AthalaSIEM Backend Architecture Analysis
## Comprehensive Assessment from Software Architecture & Security Engineering Perspectives

**Document Version:** 1.0  
**Date:** 2025-01-27  
**Authors:** Revian Ravil Athala 
**Project:** AthalaSIEM - Security Information and Event Management System

---

## Executive Summary

This document provides a dual-perspective analysis of the AthalaSIEM backend architecture:
1. **Software Architecture Perspective**: Clean Architecture, scalability, maintainability, and production-readiness
2. **Security Engineering Perspective**: Threat detection pipeline, MITRE ATT&CK alignment, and SIEM-specific security concerns

**Critical Finding**: The backend architecture is significantly immature compared to the frontend, with fundamental gaps that prevent it from handling real-world SIEM workloads. Immediate refactoring is required before production deployment.

---

## Table of Contents

1. [High-Level Assessment of Current Backend Architecture](#1-high-level-assessment-of-current-backend-architecture)
2. [Critical Gaps for a Real SIEM System](#2-critical-gaps-for-a-real-siem-system)
3. [Backend vs Frontend Maturity Comparison](#3-backend-vs-frontend-maturity-comparison)
4. [Threat Model Summary for AthalaSIEM](#4-threat-model-summary-for-athalasiem)
5. [Target Detection Pipeline Architecture](#5-target-detection-pipeline-architecture)
6. [Recommended Target Architecture](#6-recommended-target-architecture)
7. [Proposed Improved Backend Folder Structure](#7-proposed-improved-backend-folder-structure)
8. [Technology Decisions Justification](#8-technology-decisions-justification)
9. [Refactoring and Evolution Roadmap](#9-refactoring-and-evolution-roadmap)
10. [Key Risks if Backend is Not Restructured](#10-key-risks-if-backend-is-not-restructured)

---

## 1. High-Level Assessment of Current Backend Architecture

### 1.1 Current Structure Analysis

**Current Architecture Pattern**: Layered Architecture (not Clean Architecture)
- **Controllers** → **Services** → **Repositories** → **EF Core** → **PostgreSQL**
- Basic separation of concerns, but lacks domain-driven design principles
- No clear boundaries between layers (Services directly access DbContext)

**Strengths:**
-  Uses ASP.NET Core 8 with modern patterns (async/await, DI)
-  PostgreSQL with EF Core provides solid data persistence
-  gRPC support for high-performance agent communication
-  JWT authentication implemented
-  Background services for monitoring and cleanup
-  Basic repository pattern for data access abstraction
-  Serilog for structured logging

**Critical Weaknesses:**
- **No log normalization layer** - logs stored as-is without ECS-like schema
- **No correlation engine** - basic time-window correlation only
- **No rule engine** - alert generation is ad-hoc, not rule-based
- **No event streaming** - synchronous processing blocks ingestion
- **No message queue** - cannot handle log bursts
- **No detection pipeline** - logs go directly to database
- **No MITRE ATT&CK mapping** - techniques stored but not actively used
- **No enrichment pipeline** - threat intelligence not integrated into detection
- **No alert deduplication** - will create duplicate alerts
- **No severity scoring model** - hardcoded severity logic
- **No explainable detection** - alerts lack "why this fired" metadata

### 1.2 Architectural Smells Identified

#### Overengineering:
- **Multiple overlapping services**: `AlertService`, `AlertProcessingService`, `AgentService`, `AgentManagementService`
- **Inconsistent patterns**: Some services use repositories, others access DbContext directly
- **Unnecessary abstraction layers**: Repository pattern without clear benefit

#### Underengineering:
- **No domain models**: Business logic scattered across services
- **No event sourcing**: Cannot replay detection logic
- **No CQRS**: Read/write operations mixed in same services
- **No caching strategy**: Every query hits database
- **No rate limiting**: API endpoints vulnerable to DoS
- **No circuit breakers**: External service failures will cascade

#### Missing Critical Components:
- **Log Parser/Normalizer**: No unified log schema (ECS-like)
- **Detection Engine**: No rule-based or ML-based detection
- **Correlation Engine**: Basic time-window only, no behavioral correlation
- **Enrichment Service**: Threat intelligence not integrated
- **Alert Deduplication**: Will flood SOC with duplicates
- **Event Streaming**: Cannot handle high-throughput ingestion
- **Message Queue**: No buffering for log bursts

---

## 2. Critical Gaps for a Real SIEM System

### 2.1 Detection Pipeline Gaps

**Current State:**
```
Agent → gRPC/REST → LogService → Database → (AlertService checks logs)
```

**Required State:**
```
Agent → Ingestion → Parser → Normalizer → Enricher → Correlation → Detection → Alert → Response
```

**Missing Components:**

1. **Log Ingestion Layer**
   - No buffering/queuing mechanism
   - Synchronous processing blocks under load
   - No backpressure handling
   - No batch optimization

2. **Log Parser**
   - No structured parsing (Windows Event Log, Sysmon, JSON, etc.)
   - No schema validation
   - No parsing error handling

3. **Log Normalizer**
   - No ECS (Elastic Common Schema) or custom normalization
   - Inconsistent field names across sources
   - No field mapping/transformation

4. **Enrichment Pipeline**
   - Threat intelligence exists but not integrated into detection
   - No GeoIP enrichment
   - No asset/identity enrichment
   - No external API integration (VirusTotal, AbuseIPDB, etc.)

5. **Correlation Engine**
   - Basic time-window correlation only
   - No behavioral correlation
   - No cross-agent correlation
   - No attack chain detection
   - No MITRE ATT&CK technique correlation

6. **Detection Engine**
   - No rule engine (Sigma-like rules)
   - No threshold-based detection
   - No statistical anomaly detection
   - No ML-based detection (Python backend exists but not integrated)
   - No MITRE ATT&CK technique mapping

7. **Alert Processing**
   - No deduplication
   - No severity scoring
   - No alert enrichment
   - No alert correlation
   - No explainable detection metadata

### 2.2 Data Model Gaps

**Current LogEntryModels:**
- Basic fields (Message, Level, Source, Timestamp)
- Properties stored as JSON string (not queryable)
- No normalized fields (user, process, network, etc.)
- No ECS-compatible schema

**Required:**
- Normalized fields for common SIEM queries
- Separate tables for different log types (Windows Events, Sysmon, Network, etc.)
- Indexed fields for fast correlation
- Metadata table for enrichment data

**Current AlertModels:**
- Basic alert structure
- No MITRE ATT&CK technique mapping
- No detection rule reference
- No correlation metadata
- No explainable detection fields

**Required:**
- Link to detection rules
- MITRE ATT&CK technique IDs
- Correlation group ID
- Detection confidence score
- Explainable detection metadata (why this fired)

### 2.3 Performance & Scalability Gaps

**Current Limitations:**
- Synchronous log processing (blocks under load)
- No message queue (cannot buffer bursts)
- No horizontal scaling support
- No read replicas for queries
- No partitioning strategy for large tables
- No materialized views for dashboards

**Required:**
- Event streaming (Kafka, RabbitMQ, or Azure Service Bus)
- Async processing with workers
- Horizontal scaling support
- Read replicas for analytics
- Table partitioning by time
- Materialized views for common queries

---

## 3. Backend vs Frontend Maturity Comparison

### 3.1 Frontend Maturity Assessment

**Frontend Stack:**
- Next.js 14+ (App Router) - Modern, production-ready
- TypeScript - Type safety
- TailwindCSS + shadcn/ui - Professional UI components
- React Query - Sophisticated data fetching with caching
- Zustand - Clean state management
- Comprehensive component library (232 files)

**Frontend Strengths:**
-  Modern architecture with App Router
-  Type-safe API clients
-  Comprehensive UI components
-  Proper state management
-  Error handling and loading states
-  Responsive design
-  Professional UI/UX

**Frontend Maturity Score: 8/10** (Production-ready with minor polish needed)

### 3.2 Backend Maturity Assessment

**Backend Stack:**
- ASP.NET Core 8 - Modern framework
- PostgreSQL + EF Core - Solid foundation
- gRPC + REST - Good communication options
- Basic services architecture - Functional but immature

**Backend Weaknesses:**
- No detection pipeline
- No correlation engine
- No rule engine
- No event streaming
- No proper log normalization
- No MITRE ATT&CK integration
- No alert deduplication
- No explainable detection

**Backend Maturity Score: 3/10** (Not production-ready for SIEM workloads)

### 3.3 Risk Assessment

**Critical Risk: Frontend-Driven Development**
- Frontend is ready for production, backend is not
- Frontend may drive requirements that backend cannot support
- Risk of building features that backend cannot deliver

**Technical Debt:**
- Backend architecture needs complete refactoring
- Detection logic needs to be built from scratch
- Data models need normalization
- Performance optimizations required

**Business Risk:**
- Cannot handle real-world SIEM workloads
- Will fail under log bursts
- Detection accuracy will be poor
- SOC analysts will be overwhelmed with false positives

---

## 4. Threat Model Summary for AthalaSIEM

### 4.1 Attack Surface

**Endpoints (Windows/Linux Agents):**
- Agent compromise → Log tampering
- Agent uninstallation → Visibility loss
- Agent configuration tampering → Detection bypass

**Network Services:**
- gRPC endpoint → Unauthorized log injection
- REST API → DoS attacks
- Database → SQL injection (mitigated by EF Core, but still a risk)

**Authentication & Identity:**
- JWT token compromise → Unauthorized access
- API key theft → Agent impersonation
- Role-based access control → Privilege escalation

**Lateral Movement:**
- Cross-agent correlation gaps → Attack chain missed
- No network log analysis → Lateral movement undetected

**Persistence Mechanisms:**
- No detection of scheduled tasks
- No detection of service installation
- No detection of registry modifications

**Command and Control:**
- No DNS log analysis
- No network connection correlation
- No beacon detection

**Log Tampering & Evasion:**
- No log integrity verification
- No agent heartbeat validation
- No detection of log gaps

### 4.2 Adversary Models

**Insider Threat:**
- Current: Basic audit logging
- Required: Behavioral anomaly detection, privilege escalation monitoring

**Commodity Malware:**
- Current: Basic signature matching (if implemented)
- Required: Behavioral detection, MITRE ATT&CK technique mapping

**Advanced Persistent Threat (APT):**
- Current: No APT-specific detection
- Required: Attack chain correlation, long-term behavioral analysis

**Misconfiguration Abuse:**
- Current: No configuration monitoring
- Required: Baseline comparison, drift detection

**Brute Force & Credential Stuffing:**
- Current: No authentication log analysis
- Required: Failed login correlation, account lockout detection

### 4.3 Detection Coverage Gaps

**Windows Event Logs:**
-  Basic ingestion (if agent supports)
- No specific event ID analysis (4624, 4625, 4672, etc.)
- No authentication failure correlation
- No privilege escalation detection

**Sysmon:**
- No Sysmon-specific parsing
- No process creation correlation
- No network connection analysis
- No file creation monitoring

**Authentication Logs:**
- No failed login correlation
- No account lockout detection
- No suspicious login pattern detection

**Network Logs:**
- No network log ingestion
- No connection correlation
- No beacon detection
- No C2 communication detection

**Application Logs:**
-  Basic ingestion
- No application-specific parsing
- No SQL injection detection
- No XSS detection

---

## 5. Target Detection Pipeline Architecture

### 5.1 Logical Detection Pipeline

```
┌─────────────┐
│   Agents    │ (Windows/Linux/Network/Cloud)
└──────┬──────┘
       │ gRPC/REST
       ▼
┌─────────────────────────────────┐
│   Ingestion Layer                │
│   - Message Queue (RabbitMQ)     │
│   - Backpressure Handling        │
│   - Batch Optimization           │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Parser Layer                   │
│   - Windows Event Log Parser      │
│   - Sysmon Parser                │
│   - JSON Parser                   │
│   - Custom Format Parsers         │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Normalizer Layer               │
│   - ECS Schema Mapping           │
│   - Field Standardization         │
│   - Schema Validation             │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Enrichment Layer               │
│   - Threat Intelligence          │
│   - GeoIP                        │
│   - Asset/Identity DB            │
│   - External APIs (VirusTotal)   │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Correlation Engine             │
│   - Temporal Correlation          │
│   - Cross-Agent Correlation      │
│   - Behavioral Correlation        │
│   - Attack Chain Detection        │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Detection Engine               │
│   - Rule Engine (Sigma-like)      │
│   - Threshold Detection           │
│   - Statistical Anomaly           │
│   - ML Detection (Python)         │
│   - MITRE ATT&CK Mapping          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Alert Processing               │
│   - Deduplication                │
│   - Severity Scoring             │
│   - Alert Enrichment             │
│   - Explainable Metadata         │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Response Layer                 │
│   - Notification (Email/Slack)   │
│   - SOAR Integration             │
│   - Incident Creation            │
└─────────────────────────────────┘
```

### 5.2 Log Normalization Strategy

**Recommended: ECS (Elastic Common Schema) with SIEM Extensions**

**Core Fields:**
- `@timestamp` - Event timestamp
- `agent.id` - Agent identifier
- `agent.name` - Agent name
- `host.name` - Hostname
- `user.name` - Username
- `process.name` - Process name
- `process.pid` - Process ID
- `source.ip` - Source IP
- `destination.ip` - Destination IP
- `event.action` - Action type
- `event.category` - Event category
- `event.type` - Event type
- `event.outcome` - Success/Failure

**SIEM Extensions:**
- `siem.rule_id` - Detection rule ID
- `siem.technique_id` - MITRE ATT&CK technique
- `siem.confidence` - Detection confidence
- `siem.severity` - Calculated severity
- `siem.correlation_id` - Correlation group ID

**Implementation:**
- Create `LogNormalizer` service
- Map source-specific fields to ECS schema
- Store normalized logs in separate table
- Keep raw logs for forensics

### 5.3 Rule Engine and Detection Logic Design

**Rule Format (Sigma-like):**
```yaml
title: Suspicious Process Execution
id: rule-001
description: Detects execution of suspicious processes
logsource:
  product: windows
  service: sysmon
detection:
  selection:
    EventID: 1
    Image|endswith:
      - '\cmd.exe'
      - '\powershell.exe'
  condition: selection
falsepositives:
  - Legitimate administrative tasks
level: medium
tags:
  - attack.execution
  - technique.T1059
```

**Rule Engine Components:**
1. **Rule Parser** - Parse YAML/JSON rules
2. **Rule Compiler** - Convert to executable queries
3. **Rule Executor** - Execute against normalized logs
4. **Rule Manager** - CRUD operations for rules

**Detection Types:**
- **Pattern Matching** - Exact match, regex, wildcard
- **Threshold** - Count-based (e.g., 5 failed logins in 5 minutes)
- **Statistical** - Anomaly detection (mean, stddev)
- **Correlation** - Multi-event patterns
- **ML-Based** - Python backend integration

### 5.4 Correlation and Enrichment Strategy

**Correlation Types:**

1. **Temporal Correlation:**
   - Time-window based (e.g., 15 minutes)
   - Sequence-based (event A → event B)
   - Frequency-based (N events in time window)

2. **Cross-Agent Correlation:**
   - Same user across multiple agents
   - Same IP across multiple agents
   - Same process hash across agents

3. **Behavioral Correlation:**
   - Baseline deviation
   - User behavior anomaly
   - Process behavior anomaly

4. **Attack Chain Correlation:**
   - MITRE ATT&CK technique sequences
   - Kill chain progression
   - Campaign detection

**Enrichment Sources:**
- Threat Intelligence Feeds (already implemented, needs integration)
- GeoIP databases
- Asset management database
- Identity management system
- External APIs (VirusTotal, AbuseIPDB, etc.)

### 5.5 Alert Lifecycle and Severity Scoring Model

**Alert States:**
- `New` → `Acknowledged` → `InProgress` → `Resolved` / `FalsePositive` / `Closed`

**Severity Scoring:**
```csharp
SeverityScore = BaseSeverity 
    + ThreatIntelligenceScore 
    + CorrelationScore 
    + AnomalyScore 
    + TechniqueScore
```

**Base Severity:**
- Critical: 10
- High: 7
- Medium: 4
- Low: 2
- Info: 1

**Threat Intelligence Score:**
- Known malicious IP: +3
- Known malicious hash: +5
- Known C2 domain: +4

**Correlation Score:**
- Single event: 0
- 2-5 correlated events: +2
- 6+ correlated events: +4
- Attack chain detected: +5

**Anomaly Score:**
- Statistical anomaly: +2
- Behavioral anomaly: +3

**Technique Score:**
- MITRE ATT&CK technique: +1 to +3 (based on technique severity)

**Final Severity Mapping:**
- 15+ → Critical
- 10-14 → High
- 5-9 → Medium
- 2-4 → Low
- <2 → Info

### 5.6 Mapping Detections to MITRE ATT&CK

**Current State:**
- `AttackTechnique` table exists
- Techniques stored but not actively used
- No automatic technique mapping

**Required:**
1. **Technique Mapping Rules:**
   - Map detection rules to MITRE techniques
   - Map log events to techniques
   - Map correlation patterns to techniques

2. **Technique Detection:**
   - When alert fires, identify techniques
   - Store technique IDs in alert metadata
   - Display techniques in alert details

3. **Technique Correlation:**
   - Detect technique sequences (attack chains)
   - Map to MITRE tactics (Initial Access → Execution → Persistence)
   - Generate technique-based alerts

**Example Mapping:**
```csharp
// Failed login attempts → T1078 (Valid Accounts)
if (event.action == "authentication_failure" && count > 5)
    technique = "T1078";

// Process creation from suspicious location → T1055 (Process Injection)
if (process.parent.name == "suspicious" && process.name == "cmd.exe")
    technique = "T1055";
```

### 5.7 False Positive Reduction Strategies

1. **Whitelisting:**
   - IP whitelist
   - Process whitelist
   - User whitelist
   - Rule-specific whitelists

2. **Threshold Tuning:**
   - Adjust thresholds based on false positive rate
   - Machine learning for threshold optimization

3. **Context Enrichment:**
   - Enrich with asset data (is this a test server?)
   - Enrich with user data (is this an admin?)
   - Enrich with time data (is this during maintenance window?)

4. **Feedback Loop:**
   - Analyst feedback (false positive marking)
   - Automatic rule tuning
   - Rule disabling for known false positives

5. **Confidence Scoring:**
   - Low confidence alerts → Review queue
   - High confidence alerts → Immediate notification

---

## 6. Recommended Target Architecture

### 6.1 Clean Architecture Principles

**Layers:**
1. **Domain Layer** - Business logic, entities, value objects
2. **Application Layer** - Use cases, DTOs, interfaces
3. **Infrastructure Layer** - Data access, external services
4. **Presentation Layer** - Controllers, gRPC services

**Dependencies:**
- Domain ← Application ← Infrastructure
- Presentation → Application (via interfaces)

### 6.2 Domain-Driven Design

**Bounded Contexts:**
1. **Log Ingestion Context** - Receiving and parsing logs
2. **Detection Context** - Rule execution and correlation
3. **Alert Management Context** - Alert lifecycle
4. **Threat Intelligence Context** - TI feed management
5. **Agent Management Context** - Agent registration and monitoring

**Aggregates:**
- `LogEntry` - Root aggregate for logs
- `Alert` - Root aggregate for alerts
- `DetectionRule` - Root aggregate for rules
- `Agent` - Root aggregate for agents

### 6.3 Event-Driven Architecture

**Event Types:**
- `LogIngested` - Log received and parsed
- `LogNormalized` - Log normalized to ECS
- `LogEnriched` - Log enriched with TI
- `CorrelationDetected` - Correlation pattern found
- `DetectionFired` - Detection rule matched
- `AlertCreated` - Alert created from detection
- `AlertUpdated` - Alert status changed

**Event Bus:**
- Use MediatR for in-process events
- Use message queue (RabbitMQ) for cross-service events
- Event sourcing for audit trail

### 6.4 CQRS Pattern

**Commands (Write):**
- `IngestLogCommand`
- `CreateAlertCommand`
- `UpdateAlertStatusCommand`
- `CreateDetectionRuleCommand`

**Queries (Read):**
- `GetLogsQuery`
- `GetAlertsQuery`
- `GetDetectionRulesQuery`
- `GetCorrelationQuery`

**Benefits:**
- Separate read/write models
- Optimize queries independently
- Scale reads and writes separately

---

## 7. Proposed Improved Backend Folder Structure

```
backend/
├── src/
│   ├── AthalaSIEM.Domain/                    # Domain Layer
│   │   ├── Entities/
│   │   │   ├── LogEntry.cs
│   │   │   ├── Alert.cs
│   │   │   ├── DetectionRule.cs
│   │   │   ├── Agent.cs
│   │   │   └── ThreatIndicator.cs
│   │   ├── ValueObjects/
│   │   │   ├── ECSLogFields.cs
│   │   │   ├── SeverityScore.cs
│   │   │   └── MITRETechnique.cs
│   │   ├── Enums/
│   │   │   ├── LogLevel.cs
│   │   │   ├── AlertStatus.cs
│   │   │   └── DetectionType.cs
│   │   └── Interfaces/
│   │       ├── ILogRepository.cs
│   │       └── IAlertRepository.cs
│   │
│   ├── AthalaSIEM.Application/                # Application Layer
│   │   ├── UseCases/
│   │   │   ├── LogIngestion/
│   │   │   │   ├── IngestLogCommand.cs
│   │   │   │   ├── IngestLogHandler.cs
│   │   │   │   └── IngestLogValidator.cs
│   │   │   ├── Detection/
│   │   │   │   ├── ExecuteDetectionRuleCommand.cs
│   │   │   │   ├── ExecuteDetectionRuleHandler.cs
│   │   │   │   └── DetectionResult.cs
│   │   │   ├── Correlation/
│   │   │   │   ├── CorrelateLogsCommand.cs
│   │   │   │   └── CorrelateLogsHandler.cs
│   │   │   └── Alerts/
│   │   │       ├── CreateAlertCommand.cs
│   │   │       └── CreateAlertHandler.cs
│   │   ├── Queries/
│   │   │   ├── Logs/
│   │   │   │   ├── GetLogsQuery.cs
│   │   │   │   └── GetLogsHandler.cs
│   │   │   └── Alerts/
│   │   │       ├── GetAlertsQuery.cs
│   │   │       └── GetAlertsHandler.cs
│   │   ├── DTOs/
│   │   │   ├── LogDto.cs
│   │   │   ├── AlertDto.cs
│   │   │   └── DetectionRuleDto.cs
│   │   └── Mappings/
│   │       └── AutoMapperProfiles.cs
│   │
│   ├── AthalaSIEM.Infrastructure/            # Infrastructure Layer
│   │   ├── Data/
│   │   │   ├── ApplicationDbContext.cs
│   │   │   ├── Repositories/
│   │   │   │   ├── LogRepository.cs
│   │   │   │   └── AlertRepository.cs
│   │   │   └── Migrations/
│   │   ├── Messaging/
│   │   │   ├── RabbitMQ/
│   │   │   │   ├── RabbitMQService.cs
│   │   │   │   └── LogIngestionConsumer.cs
│   │   │   └── EventBus/
│   │   │       └── MediatREventBus.cs
│   │   ├── Parsers/
│   │   │   ├── ILogParser.cs
│   │   │   ├── WindowsEventLogParser.cs
│   │   │   ├── SysmonParser.cs
│   │   │   └── JsonLogParser.cs
│   │   ├── Normalizers/
│   │   │   ├── ILogNormalizer.cs
│   │   │   └── ECSLogNormalizer.cs
│   │   ├── Enrichers/
│   │   │   ├── ILogEnricher.cs
│   │   │   ├── ThreatIntelligenceEnricher.cs
│   │   │   ├── GeoIPEnricher.cs
│   │   │   └── AssetEnricher.cs
│   │   ├── Correlation/
│   │   │   ├── ICorrelationEngine.cs
│   │   │   ├── TemporalCorrelator.cs
│   │   │   ├── BehavioralCorrelator.cs
│   │   │   └── AttackChainCorrelator.cs
│   │   ├── Detection/
│   │   │   ├── IDetectionEngine.cs
│   │   │   ├── RuleEngine/
│   │   │   │   ├── RuleParser.cs
│   │   │   │   ├── RuleCompiler.cs
│   │   │   │   └── RuleExecutor.cs
│   │   │   ├── ThresholdDetector.cs
│   │   │   ├── AnomalyDetector.cs
│   │   │   └── MLDetector.cs (calls Python backend)
│   │   ├── AlertProcessing/
│   │   │   ├── AlertDeduplicator.cs
│   │   │   ├── AlertSeverityScorer.cs
│   │   │   └── AlertEnricher.cs
│   │   └── ExternalServices/
│   │       ├── VirusTotalService.cs
│   │       └── AbuseIPDBService.cs
│   │
│   ├── AthalaSIEM.API/                       # Presentation Layer
│   │   ├── Controllers/
│   │   │   ├── LogsController.cs
│   │   │   ├── AlertsController.cs
│   │   │   └── DetectionRulesController.cs
│   │   ├── gRPC/
│   │   │   ├── LogIngestionService.cs
│   │   │   └── AgentService.cs
│   │   └── Middleware/
│   │       ├── ErrorHandlingMiddleware.cs
│   │       └── RateLimitingMiddleware.cs
│   │
│   └── AthalaSIEM.Workers/                   # Background Workers
│       ├── LogIngestionWorker.cs
│       ├── CorrelationWorker.cs
│       ├── DetectionWorker.cs
│       └── AlertProcessingWorker.cs
│
├── tests/
│   ├── AthalaSIEM.Domain.Tests/
│   ├── AthalaSIEM.Application.Tests/
│   └── AthalaSIEM.Infrastructure.Tests/
│
└── docker-compose.yml
```

### 7.1 Key Architectural Decisions

**1. Separate Projects by Layer:**
- Clear separation of concerns
- Independent versioning
- Easier testing

**2. Use Cases in Application Layer:**
- Business logic in one place
- Easy to test
- Clear dependencies

**3. Infrastructure Implementations:**
- All external dependencies in Infrastructure
- Easy to swap implementations
- Testable with mocks

**4. Background Workers:**
- Separate processes for heavy workloads
- Independent scaling
- Fault isolation

---

## 8. Technology Decisions Justification

### 8.1 .NET 8 for Core Backend

**Justification:**
-  High performance (async/await, Span<T>)
-  Strong typing and null safety
-  Excellent tooling (Visual Studio, Rider)
-  Mature ecosystem (NuGet packages)
-  Cross-platform (Linux, Windows)
-  Good for long-running services
-  Strong concurrency support

**Keep in .NET 8:**
- Log ingestion and parsing
- Detection rule engine
- Correlation engine
- Alert processing
- API layer
- Background workers

### 8.2 PostgreSQL for Data Storage

**Justification:**
-  ACID compliance
-  JSON support (for flexible schemas)
-  Full-text search
-  Partitioning support
-  Read replicas
-  Self-hosted friendly
-  Cost-effective

**Optimizations Needed:**
- Table partitioning by time (monthly partitions)
- Read replicas for analytics
- Materialized views for dashboards
- Proper indexing strategy

### 8.3 Message Queue (RabbitMQ Recommended)

**Justification:**
-  Handles log bursts
-  Decouples ingestion from processing
-  Backpressure handling
-  Self-hosted friendly
-  .NET client support

**Alternatives:**
- Azure Service Bus (if using Azure)
- Apache Kafka (for very high throughput)
- Redis Streams (lighter weight)

### 8.4 Python Backend for ML

**Justification:**
-  Better ML ecosystem (TensorFlow, PyTorch)
-  Easier data science workflows
-  Existing Python backend in codebase

**Integration:**
- Call Python backend via HTTP/gRPC
- Send normalized logs for ML analysis
- Receive ML detection results
- Async processing to avoid blocking

### 8.5 Go for Agents (Future Consideration)

**Justification:**
-  Lower resource usage
-  Better for system-level operations
-  Single binary deployment
-  Fast startup time

**Current State:**
- .NET agents exist and work
- Consider Go for new agents if resource usage is a concern

---

## 9. Refactoring and Evolution Roadmap

### 9.1 Short-Term (0-3 Months) - Critical Foundation

**Phase 1: Detection Pipeline Foundation (Month 1)**
1. **Week 1-2: Log Normalization**
   - Create `LogNormalizer` service
   - Implement ECS schema mapping
   - Create normalized log table
   - Migrate existing logs

2. **Week 3-4: Message Queue Integration**
   - Set up RabbitMQ
   - Create `LogIngestionWorker`
   - Implement backpressure handling
   - Test with log bursts

**Phase 2: Basic Detection Engine (Month 2)**
1. **Week 1-2: Rule Engine**
   - Create rule parser (YAML/JSON)
   - Implement rule compiler
   - Create rule executor
   - Build rule management API

2. **Week 3-4: Basic Correlation**
   - Implement temporal correlation
   - Create correlation worker
   - Test correlation accuracy

**Phase 3: Alert Processing (Month 3)**
1. **Week 1-2: Alert Deduplication**
   - Implement deduplication logic
   - Create deduplication service
   - Test with duplicate scenarios

2. **Week 3-4: Severity Scoring**
   - Implement severity scoring model
   - Create scoring service
   - Integrate with alert creation

### 9.2 Mid-Term (3-6 Months) - Enhanced Detection

**Phase 4: Advanced Detection (Month 4)**
1. **MITRE ATT&CK Integration**
   - Map detection rules to techniques
   - Create technique detection service
   - Display techniques in alerts

2. **Enrichment Pipeline**
   - Integrate threat intelligence
   - Add GeoIP enrichment
   - Add asset enrichment

**Phase 5: Advanced Correlation (Month 5)**
1. **Behavioral Correlation**
   - Implement baseline detection
   - Create behavioral correlator
   - Test with real data

2. **Attack Chain Detection**
   - Implement technique sequence detection
   - Create attack chain correlator
   - Generate attack chain alerts

**Phase 6: ML Integration (Month 6)**
1. **Python Backend Integration**
   - Create ML detection service
   - Integrate with Python backend
   - Test ML detection accuracy

2. **Anomaly Detection**
   - Implement statistical anomaly detection
   - Create anomaly detector
   - Test false positive rate

### 9.3 Long-Term (6-12 Months) - Production Hardening

**Phase 7: Performance Optimization (Month 7-8)**
1. **Database Optimization**
   - Implement table partitioning
   - Create read replicas
   - Optimize queries
   - Add materialized views

2. **Horizontal Scaling**
   - Implement worker scaling
   - Add load balancing
   - Test under high load

**Phase 8: Advanced Features (Month 9-10)**
1. **Explainable Detection**
   - Add detection metadata
   - Create explanation service
   - Display in alert details

2. **False Positive Reduction**
   - Implement whitelisting
   - Add feedback loop
   - Create rule tuning service

**Phase 9: Production Hardening (Month 11-12)**
1. **Security Hardening**
   - Implement rate limiting
   - Add circuit breakers
   - Enhance authentication
   - Add audit logging

2. **Monitoring & Observability**
   - Add metrics (Prometheus)
   - Add distributed tracing
   - Create health checks
   - Add alerting for system health

---

## 10. Key Risks if Backend is Not Restructured

### 10.1 Technical Risks

**1. Performance Failure Under Load**
- **Risk**: System will fail during log bursts
- **Impact**: Lost logs, missed detections
- **Mitigation**: Implement message queue and workers

**2. Detection Accuracy Issues**
- **Risk**: High false positive rate, missed true positives
- **Impact**: SOC analyst fatigue, security gaps
- **Mitigation**: Implement proper detection pipeline

**3. Scalability Limitations**
- **Risk**: Cannot scale horizontally
- **Impact**: Limited growth, performance degradation
- **Mitigation**: Implement worker-based architecture

**4. Data Model Limitations**
- **Risk**: Cannot query logs efficiently
- **Impact**: Slow dashboards, poor search performance
- **Mitigation**: Normalize logs, optimize database

### 10.2 Business Risks

**1. Customer Dissatisfaction**
- **Risk**: System does not meet SIEM requirements
- **Impact**: Customer churn, negative reviews
- **Mitigation**: Complete backend refactoring

**2. Compliance Failures**
- **Risk**: Cannot meet compliance requirements (SOC 2, ISO 27001)
- **Impact**: Loss of enterprise customers
- **Mitigation**: Implement proper audit logging and security controls

**3. Competitive Disadvantage**
- **Risk**: Competitors have better detection capabilities
- **Impact**: Market share loss
- **Mitigation**: Implement advanced detection features

### 10.3 Security Risks

**1. Detection Gaps**
- **Risk**: Missed security incidents
- **Impact**: Security breaches, data loss
- **Mitigation**: Implement comprehensive detection pipeline

**2. Alert Fatigue**
- **Risk**: Too many false positives
- **Impact**: Real alerts ignored
- **Mitigation**: Implement deduplication and severity scoring

**3. Log Tampering**
- **Risk**: Agents compromised, logs tampered
- **Impact**: Loss of visibility
- **Mitigation**: Implement log integrity verification

---

## Conclusion

The AthalaSIEM backend requires **significant refactoring** before it can handle real-world SIEM workloads. The current architecture is functional for basic log storage and retrieval but lacks the critical components required for production-grade threat detection.

**Priority Actions:**
1. **Immediate**: Implement log normalization and message queue
2. **Short-term**: Build detection pipeline and rule engine
3. **Mid-term**: Add correlation, enrichment, and ML integration
4. **Long-term**: Optimize for scale and harden for production

**Key Success Metrics:**
- Log ingestion rate: 10,000+ logs/second
- Detection accuracy: <5% false positive rate
- Alert processing time: <1 second
- Query performance: <100ms for dashboard queries

**Estimated Effort:**
- Short-term (0-3 months): 2-3 senior developers
- Mid-term (3-6 months): 3-4 developers
- Long-term (6-12 months): 2-3 developers + DevOps

The frontend is production-ready, but the backend must catch up to avoid creating a critical gap in the product offering.

---

**Document End**
