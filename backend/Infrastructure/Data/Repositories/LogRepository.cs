using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;

namespace Backend.Infrastructure.Data.Repositories;

public class LogRepository : ILogRepository
{
    private readonly ApplicationDbContext _context;
    private readonly ILogger<LogRepository> _logger;

    public LogRepository(ApplicationDbContext context, ILogger<LogRepository> logger)
    {
        _context = context;
        _logger = logger;
    }

    public async Task<LogEntry?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
    {
        // Map to existing LogEntryModels table
        var model = await _context.LogEntries.FindAsync(new object[] { id }, cancellationToken);
        return model != null ? MapToDomain(model) : null;
    }

    public async Task<IEnumerable<LogEntry>> GetByAgentIdAsync(string agentId, DateTime? startTime = null, DateTime? endTime = null, CancellationToken cancellationToken = default)
    {
        var query = _context.LogEntries.Where(l => l.AgentId == agentId);
        
        if (startTime.HasValue)
            query = query.Where(l => l.Timestamp >= startTime.Value);
        if (endTime.HasValue)
            query = query.Where(l => l.Timestamp <= endTime.Value);

        var models = await query.ToListAsync(cancellationToken);
        return models.Select(MapToDomain);
    }

    public async Task<IEnumerable<LogEntry>> GetUnprocessedAsync(int limit = 1000, CancellationToken cancellationToken = default)
    {
        var models = await _context.LogEntries
            .Where(l => !l.Processed)
            .OrderBy(l => l.Timestamp)
            .Take(limit)
            .ToListAsync(cancellationToken);
        
        return models.Select(MapToDomain);
    }

    public async Task<IEnumerable<LogEntry>> GetUnnormalizedAsync(int limit = 1000, CancellationToken cancellationToken = default)
    {
        // Check if log entry has been normalized by checking if it has normalized fields in Properties
        // For now, we'll return unprocessed logs as unnormalized
        var models = await _context.LogEntries
            .Where(l => !l.Processed)
            .OrderBy(l => l.Timestamp)
            .Take(limit)
            .ToListAsync(cancellationToken);
        
        return models.Select(MapToDomain);
    }

    public async Task AddAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        var model = MapToModel(logEntry);
        await _context.LogEntries.AddAsync(model, cancellationToken);
        await _context.SaveChangesAsync(cancellationToken);
    }

    public async Task AddRangeAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default)
    {
        var models = logEntries.Select(MapToModel);
        await _context.LogEntries.AddRangeAsync(models, cancellationToken);
        await _context.SaveChangesAsync(cancellationToken);
    }

    public async Task UpdateAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        var model = await _context.LogEntries.FindAsync(new object[] { logEntry.Id }, cancellationToken);
        if (model != null)
        {
            UpdateModel(model, logEntry);
            await _context.SaveChangesAsync(cancellationToken);
        }
    }

    public async Task<LogEntry?> GetByCorrelationIdAsync(string correlationId, CancellationToken cancellationToken = default)
    {
        // This will need a CorrelationId column in log_entries table
        // For now, return empty
        await Task.CompletedTask;
        return null;
    }

    public async Task UpdateRangeAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default)
    {
        foreach (var logEntry in logEntries)
        {
            var model = await _context.LogEntries.FindAsync(new object[] { logEntry.Id }, cancellationToken);
            if (model != null)
            {
                UpdateModel(model, logEntry);
            }
        }
        await _context.SaveChangesAsync(cancellationToken);
    }

    public async Task<IEnumerable<LogEntry>> GetNormalizedLogsByFieldAsync(
        string fieldName, 
        string fieldValue, 
        DateTime startTime, 
        DateTime endTime, 
        CancellationToken cancellationToken = default)
    {
        // Query normalized logs from NormalizedLogs table
        var normalizedLogs = await _context.NormalizedLogs
            .Where(nl => nl.Timestamp >= startTime && nl.Timestamp <= endTime)
            .ToListAsync(cancellationToken);

        // Filter by field name and value
        var filtered = normalizedLogs.Where(nl =>
        {
            return fieldName switch
            {
                "SourceIp" => nl.SourceIp == fieldValue,
                "UserName" => nl.UserName == fieldValue,
                "DestinationIp" => nl.DestinationIp == fieldValue,
                "ProcessName" => nl.ProcessName == fieldValue,
                _ => false
            };
        }).ToList();

        // Get corresponding LogEntries and map to domain entities
        var logEntryIds = filtered.Select(nl => nl.LogEntryId).ToList();
        var logEntryModels = await _context.LogEntries
            .Where(le => logEntryIds.Contains(le.Id))
            .ToListAsync(cancellationToken);

        var logEntries = logEntryModels.Select(MapToDomain).ToList();

        // Attach normalized fields from NormalizedLogs
        foreach (var logEntry in logEntries)
        {
            var normalizedLog = filtered.FirstOrDefault(nl => nl.LogEntryId == logEntry.Id);
            if (normalizedLog != null)
            {
                // Map NormalizedLog to ECSLogFields
                logEntry.NormalizedFields = new Backend.Domain.ValueObjects.ECSLogFields
                {
                    Timestamp = normalizedLog.Timestamp,
                    AgentId = normalizedLog.AgentId,
                    AgentName = normalizedLog.AgentName,
                    HostName = normalizedLog.HostName,
                    HostIp = normalizedLog.HostIp,
                    UserName = normalizedLog.UserName,
                    UserId = normalizedLog.UserId,
                    UserDomain = normalizedLog.UserDomain,
                    ProcessName = normalizedLog.ProcessName,
                    ProcessId = normalizedLog.ProcessId,
                    ProcessPath = normalizedLog.ProcessPath,
                    ProcessCommandLine = normalizedLog.ProcessCommandLine,
                    ProcessHash = normalizedLog.ProcessHash,
                    ParentProcessName = normalizedLog.ParentProcessName,
                    ParentProcessId = normalizedLog.ParentProcessId,
                    SourceIp = normalizedLog.SourceIp,
                    SourcePort = normalizedLog.SourcePort,
                    DestinationIp = normalizedLog.DestinationIp,
                    DestinationPort = normalizedLog.DestinationPort,
                    Protocol = normalizedLog.Protocol,
                    EventAction = normalizedLog.EventAction,
                    EventCategory = normalizedLog.EventCategory,
                    EventType = normalizedLog.EventType,
                    EventOutcome = normalizedLog.EventOutcome,
                    EventCode = normalizedLog.EventCode,
                    FilePath = normalizedLog.FilePath,
                    FileName = normalizedLog.FileName,
                    FileHash = normalizedLog.FileHash,
                    FileSize = normalizedLog.FileSize,
                    SiemRuleId = normalizedLog.SiemRuleId,
                    SiemTechniqueId = normalizedLog.SiemTechniqueId,
                    SiemConfidence = normalizedLog.SiemConfidence,
                    SiemSeverity = normalizedLog.SiemSeverity,
                    SiemCorrelationId = normalizedLog.SiemCorrelationId,
                    Metadata = !string.IsNullOrEmpty(normalizedLog.MetadataJson)
                        ? System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(normalizedLog.MetadataJson)
                        : null
                };
                logEntry.IsNormalized = true;
                logEntry.NormalizedAt = normalizedLog.CreatedAt;
            }
        }

        return logEntries;
    }

    private LogEntry MapToDomain(Models.LogEntryModels model)
    {
        var entry = new LogEntry
        {
            Id = model.Id,
            AgentId = model.AgentId,
            Timestamp = model.Timestamp,
            ReceivedAt = model.ReceivedAt,
            RawMessage = model.Message,
            Source = model.Source,
            Level = model.Level ?? "Information",
            Category = model.Category,
            EventId = model.EventId,
            MachineName = model.MachineName ?? string.Empty,
            IPAddress = model.IPAddress ?? string.Empty,
            RawProperties = model.Properties,
            IsNormalized = false,
            Processed = model.Processed,
            ProcessedAt = model.ProcessedAt
        };

        return entry;
    }

    private Models.LogEntryModels MapToModel(LogEntry entry)
    {
        // Ensure all DateTime values are UTC (PostgreSQL requirement)
        DateTime EnsureUtc(DateTime dt) => dt.Kind == DateTimeKind.Utc ? dt : dt.ToUniversalTime();
        
        return new Models.LogEntryModels
        {
            Id = entry.Id,
            AgentId = entry.AgentId,
            Timestamp = EnsureUtc(entry.Timestamp),
            ReceivedAt = EnsureUtc(entry.ReceivedAt),
            Level = !string.IsNullOrEmpty(entry.Level) ? entry.Level : "Information",
            Message = !string.IsNullOrEmpty(entry.RawMessage) ? entry.RawMessage : "(no message)",
            Source = !string.IsNullOrEmpty(entry.Source) ? entry.Source : "Unknown",
            Category = entry.Category,
            EventId = entry.EventId ?? 0,
            MachineName = entry.MachineName ?? string.Empty,
            IPAddress = entry.IPAddress ?? string.Empty,
            Properties = entry.RawProperties,
            Processed = entry.Processed,
            ProcessedAt = entry.ProcessedAt.HasValue ? EnsureUtc(entry.ProcessedAt.Value) : null,
            CreatedAt = EnsureUtc(entry.ReceivedAt)
        };
    }

    private void UpdateModel(Models.LogEntryModels model, LogEntry entry)
    {
        model.Processed = entry.Processed;
        model.ProcessedAt = entry.ProcessedAt;

        // Write enriched properties (MITRE, IPs, etc.) back to the Properties JSON column
        // This is what the frontend reads via LogEntryDto.Properties
        if (!string.IsNullOrEmpty(entry.RawProperties))
        {
            model.Properties = entry.RawProperties;
        }

        // Update IsNormalized flag
        if (entry.IsNormalized)
        {
            model.Processed = true;
        }
    }
}
