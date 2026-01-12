using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;

namespace Backend.Infrastructure.Data.Repositories;

public interface INormalizedLogRepository
{
    Task<NormalizedLog?> GetByIdAsync(string id, CancellationToken cancellationToken = default);
    Task<NormalizedLog?> GetByLogEntryIdAsync(string logEntryId, CancellationToken cancellationToken = default);
    Task<IEnumerable<NormalizedLog>> GetByCorrelationIdAsync(string correlationId, CancellationToken cancellationToken = default);
    Task<IEnumerable<NormalizedLog>> GetByTechniqueIdAsync(string techniqueId, DateTime? startTime = null, DateTime? endTime = null, CancellationToken cancellationToken = default);
    Task<IEnumerable<NormalizedLog>> SearchAsync(string? sourceIp = null, string? destinationIp = null, string? processName = null, string? userName = null, DateTime? startTime = null, DateTime? endTime = null, int limit = 1000, CancellationToken cancellationToken = default);
    Task AddAsync(NormalizedLog normalizedLog, CancellationToken cancellationToken = default);
    Task AddRangeAsync(IEnumerable<NormalizedLog> normalizedLogs, CancellationToken cancellationToken = default);
    Task UpdateAsync(NormalizedLog normalizedLog, CancellationToken cancellationToken = default);
}

public class NormalizedLogRepository : INormalizedLogRepository
{
    private readonly ApplicationDbContext _context;
    private readonly ILogger<NormalizedLogRepository> _logger;

    public NormalizedLogRepository(ApplicationDbContext context, ILogger<NormalizedLogRepository> logger)
    {
        _context = context;
        _logger = logger;
    }

    public async Task<NormalizedLog?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
    {
        var model = await _context.NormalizedLogs.FindAsync(new object[] { id }, cancellationToken);
        return model;
    }

    public async Task<NormalizedLog?> GetByLogEntryIdAsync(string logEntryId, CancellationToken cancellationToken = default)
    {
        var model = await _context.NormalizedLogs
            .FirstOrDefaultAsync(n => n.LogEntryId == logEntryId, cancellationToken);
        return model;
    }

    public async Task<IEnumerable<NormalizedLog>> GetByCorrelationIdAsync(string correlationId, CancellationToken cancellationToken = default)
    {
        var models = await _context.NormalizedLogs
            .Where(n => n.SiemCorrelationId == correlationId)
            .OrderBy(n => n.Timestamp)
            .ToListAsync(cancellationToken);
        return models;
    }

    public async Task<IEnumerable<NormalizedLog>> GetByTechniqueIdAsync(string techniqueId, DateTime? startTime = null, DateTime? endTime = null, CancellationToken cancellationToken = default)
    {
        var query = _context.NormalizedLogs.Where(n => n.SiemTechniqueId == techniqueId);
        
        if (startTime.HasValue)
            query = query.Where(n => n.Timestamp >= startTime.Value);
        if (endTime.HasValue)
            query = query.Where(n => n.Timestamp <= endTime.Value);

        var models = await query.OrderBy(n => n.Timestamp).ToListAsync(cancellationToken);
        return models;
    }

    public async Task<IEnumerable<NormalizedLog>> SearchAsync(string? sourceIp = null, string? destinationIp = null, string? processName = null, string? userName = null, DateTime? startTime = null, DateTime? endTime = null, int limit = 1000, CancellationToken cancellationToken = default)
    {
        var query = _context.NormalizedLogs.AsQueryable();

        if (!string.IsNullOrEmpty(sourceIp))
            query = query.Where(n => n.SourceIp == sourceIp);
        if (!string.IsNullOrEmpty(destinationIp))
            query = query.Where(n => n.DestinationIp == destinationIp);
        if (!string.IsNullOrEmpty(processName))
            query = query.Where(n => n.ProcessName == processName);
        if (!string.IsNullOrEmpty(userName))
            query = query.Where(n => n.UserName == userName);
        if (startTime.HasValue)
            query = query.Where(n => n.Timestamp >= startTime.Value);
        if (endTime.HasValue)
            query = query.Where(n => n.Timestamp <= endTime.Value);

        var models = await query
            .OrderByDescending(n => n.Timestamp)
            .Take(limit)
            .ToListAsync(cancellationToken);
        
        return models;
    }

    public async Task AddAsync(NormalizedLog normalizedLog, CancellationToken cancellationToken = default)
    {
        await _context.NormalizedLogs.AddAsync(normalizedLog, cancellationToken);
        await _context.SaveChangesAsync(cancellationToken);
    }

    public async Task AddRangeAsync(IEnumerable<NormalizedLog> normalizedLogs, CancellationToken cancellationToken = default)
    {
        await _context.NormalizedLogs.AddRangeAsync(normalizedLogs, cancellationToken);
        await _context.SaveChangesAsync(cancellationToken);
    }

    public async Task UpdateAsync(NormalizedLog normalizedLog, CancellationToken cancellationToken = default)
    {
        _context.NormalizedLogs.Update(normalizedLog);
        await _context.SaveChangesAsync(cancellationToken);
    }
}
