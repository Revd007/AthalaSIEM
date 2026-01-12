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
            .Where(l => !l.Processed && l.IsNormalized)
            .OrderBy(l => l.Timestamp)
            .Take(limit)
            .ToListAsync(cancellationToken);
        
        return models.Select(MapToDomain);
    }

    public async Task<IEnumerable<LogEntry>> GetUnnormalizedAsync(int limit = 1000, CancellationToken cancellationToken = default)
    {
        var models = await _context.LogEntries
            .Where(l => !l.IsNormalized)
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
            Category = model.Category,
            EventId = model.EventId,
            RawProperties = model.Properties,
            IsNormalized = false, // Will be set when normalized
            Processed = model.Processed,
            ProcessedAt = model.ProcessedAt
        };

        // Parse normalized fields from Properties if available
        // In production, you'd have a separate NormalizedLogs table

        return entry;
    }

    private Models.LogEntryModels MapToModel(LogEntry entry)
    {
        return new Models.LogEntryModels
        {
            Id = entry.Id,
            AgentId = entry.AgentId,
            Timestamp = entry.Timestamp,
            ReceivedAt = entry.ReceivedAt,
            Message = entry.RawMessage,
            Source = entry.Source,
            Category = entry.Category,
            EventId = entry.EventId ?? 0,
            Properties = entry.RawProperties,
            Processed = entry.Processed,
            ProcessedAt = entry.ProcessedAt,
            CreatedAt = entry.ReceivedAt
        };
    }

    private void UpdateModel(Models.LogEntryModels model, LogEntry entry)
    {
        model.Processed = entry.Processed;
        model.ProcessedAt = entry.ProcessedAt;
        // Add normalized fields to Properties JSON in production
    }
}
