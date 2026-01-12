using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;
using System.Text.Json;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Data.Repositories;

public class AlertRepository : IAlertRepository
{
    private readonly ApplicationDbContext _context;
    private readonly ILogger<AlertRepository> _logger;

    public AlertRepository(ApplicationDbContext context, ILogger<AlertRepository> logger)
    {
        _context = context;
        _logger = logger;
    }

    public async Task<Alert?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
    {
        var model = await _context.Alert.FindAsync(new object[] { id }, cancellationToken);
        if (model == null)
            return null;
            
        var alert = MapToDomain(model);
        await LoadExtendedFieldsAsync(alert, cancellationToken);
        return alert;
    }

    public async Task<Alert?> GetByDeduplicationKeyAsync(string key, CancellationToken cancellationToken = default)
    {
        // Query by deduplication_key column (added in migration)
        // Use raw SQL for now since column may not be mapped yet
        try
        {
            var sql = "SELECT * FROM alerts WHERE deduplication_key = {0} LIMIT 1";
            var model = await _context.Alert
                .FromSqlRaw(sql, key)
                .AsNoTracking()
                .FirstOrDefaultAsync(cancellationToken);
            
            if (model == null)
                return null;
                
            var alert = MapToDomain(model);
            await LoadExtendedFieldsAsync(alert, cancellationToken);
            return alert;
        }
        catch
        {
            // Column may not exist yet - fallback to ResolutionNotes
            var model = await _context.Alert
                .FirstOrDefaultAsync(a => a.ResolutionNotes == key, cancellationToken);
            return model != null ? MapToDomain(model) : null;
        }
    }

    public async Task<IEnumerable<Alert>> GetByStatusAsync(AlertStatus status, CancellationToken cancellationToken = default)
    {
        var statusModel = MapStatus(status);
        var models = await _context.Alert
            .Where(a => a.Status == statusModel)
            .ToListAsync(cancellationToken);
        return models.Select(MapToDomain);
    }

    public async Task<IEnumerable<Alert>> GetByAgentIdAsync(string agentId, CancellationToken cancellationToken = default)
    {
        var models = await _context.Alert
            .Where(a => a.AgentId == agentId)
            .ToListAsync(cancellationToken);
        return models.Select(MapToDomain);
    }

    public async Task AddAsync(Alert alert, CancellationToken cancellationToken = default)
    {
        var model = MapToModel(alert);
        await _context.Alert.AddAsync(model, cancellationToken);
        await _context.SaveChangesAsync(cancellationToken);
        
        // Set extended fields after save
        await SetExtendedFieldsAsync(model, alert);
    }

    public async Task UpdateAsync(Alert alert, CancellationToken cancellationToken = default)
    {
        var model = await _context.Alert.FindAsync(new object[] { alert.Id }, cancellationToken);
        if (model != null)
        {
            await UpdateModelAsync(model, alert, cancellationToken);
            await _context.SaveChangesAsync(cancellationToken);
        }
    }

    public async Task DeleteAsync(string id, CancellationToken cancellationToken = default)
    {
        var model = await _context.Alert.FindAsync(new object[] { id }, cancellationToken);
        if (model != null)
        {
            _context.Alert.Remove(model);
            await _context.SaveChangesAsync(cancellationToken);
        }
    }

    private Alert MapToDomain(Models.AlertModels model)
    {
        var alert = new Alert
        {
            Id = model.Id,
            AgentId = model.AgentId,
            Title = model.Title,
            Description = model.Description,
            Message = model.Message,
            Severity = MapSeverity(model.Severity),
            Status = MapStatus(model.Status),
            Timestamp = model.Timestamp,
            Source = model.Source,
            AcknowledgedBy = model.AcknowledgedBy,
            AcknowledgedAt = model.AcknowledgedAt,
            ResolvedBy = model.ResolvedBy,
            ResolvedAt = model.ResolvedAt,
            ResolutionNotes = model.ResolutionNotes,
            CreatedAt = model.CreatedAt,
            UpdatedAt = model.UpdatedAt
        };

        // Try to read extended fields - will be populated after migration
        // For now, these will be null until migration runs

        return alert;
    }
    
    private async Task LoadExtendedFieldsAsync(Alert alert, CancellationToken cancellationToken)
    {
        // Load extended fields using raw SQL
        var sql = @"
            SELECT deduplication_key, rule_id, correlation_id, confidence, 
                   detection_reason, occurrence_count, first_occurrence, last_occurrence,
                   technique_ids_json, related_log_ids_json, detection_metadata_json
            FROM alerts 
            WHERE id = {0}";
        
        try
        {
            var result = await _context.Database
                .SqlQueryRaw<ExtendedAlertFields>(sql, alert.Id)
                .FirstOrDefaultAsync(cancellationToken);
            
            if (result != null)
            {
                alert.DeduplicationKey = result.DeduplicationKey;
                alert.RuleId = result.RuleId;
                alert.CorrelationId = result.CorrelationId;
                alert.Confidence = result.Confidence;
                alert.DetectionReason = result.DetectionReason;
                alert.OccurrenceCount = result.OccurrenceCount;
                alert.FirstOccurrence = result.FirstOccurrence;
                alert.LastOccurrence = result.LastOccurrence;
                
                if (!string.IsNullOrEmpty(result.TechniqueIdsJson))
                    alert.TechniqueIds = JsonSerializer.Deserialize<List<string>>(result.TechniqueIdsJson) ?? new();
                if (!string.IsNullOrEmpty(result.RelatedLogIdsJson))
                    alert.RelatedLogIds = JsonSerializer.Deserialize<List<string>>(result.RelatedLogIdsJson) ?? new();
                if (!string.IsNullOrEmpty(result.DetectionMetadataJson))
                    alert.DetectionMetadata = JsonSerializer.Deserialize<Dictionary<string, object>>(result.DetectionMetadataJson);
            }
        }
        catch
        {
            // Columns may not exist yet
        }
    }
    
    private class ExtendedAlertFields
    {
        public string? DeduplicationKey { get; set; }
        public string? RuleId { get; set; }
        public string? CorrelationId { get; set; }
        public double Confidence { get; set; }
        public string? DetectionReason { get; set; }
        public int OccurrenceCount { get; set; }
        public DateTime? FirstOccurrence { get; set; }
        public DateTime? LastOccurrence { get; set; }
        public string? TechniqueIdsJson { get; set; }
        public string? RelatedLogIdsJson { get; set; }
        public string? DetectionMetadataJson { get; set; }
    }

    private Models.AlertModels MapToModel(Alert alert)
    {
        var model = new Models.AlertModels
        {
            Id = alert.Id,
            AgentId = alert.AgentId,
            Title = alert.Title,
            Description = alert.Description,
            Message = alert.Message,
            Severity = MapSeverity(alert.Severity),
            Status = MapStatus(alert.Status),
            Timestamp = alert.Timestamp,
            Source = alert.Source,
            AcknowledgedBy = alert.AcknowledgedBy,
            AcknowledgedAt = alert.AcknowledgedAt,
            ResolvedBy = alert.ResolvedBy,
            ResolvedAt = alert.ResolvedAt,
            ResolutionNotes = alert.ResolutionNotes,
            CreatedAt = alert.CreatedAt,
            UpdatedAt = alert.UpdatedAt
        };

        return model;
    }
    
    private async Task SetExtendedFieldsAsync(Models.AlertModels model, Alert alert)
    {
        // Use raw SQL to update extended fields until migration is applied
        if (!string.IsNullOrEmpty(alert.DeduplicationKey) || 
            !string.IsNullOrEmpty(alert.RuleId) || 
            alert.TechniqueIds.Any())
        {
            var sql = @"
                UPDATE alerts 
                SET deduplication_key = {0},
                    rule_id = {1},
                    correlation_id = {2},
                    confidence = {3},
                    detection_reason = {4},
                    occurrence_count = {5},
                    first_occurrence = {6},
                    last_occurrence = {7},
                    technique_ids_json = {8},
                    related_log_ids_json = {9},
                    detection_metadata_json = {10}
                WHERE id = {11}";
            
            await _context.Database.ExecuteSqlRawAsync(sql,
                alert.DeduplicationKey ?? (object)DBNull.Value,
                alert.RuleId ?? (object)DBNull.Value,
                alert.CorrelationId ?? (object)DBNull.Value,
                alert.Confidence,
                alert.DetectionReason ?? (object)DBNull.Value,
                alert.OccurrenceCount,
                alert.FirstOccurrence ?? (object)DBNull.Value,
                alert.LastOccurrence ?? (object)DBNull.Value,
                alert.TechniqueIds.Any() ? JsonSerializer.Serialize(alert.TechniqueIds) : (object)DBNull.Value,
                alert.RelatedLogIds.Any() ? JsonSerializer.Serialize(alert.RelatedLogIds) : (object)DBNull.Value,
                alert.DetectionMetadata != null ? JsonSerializer.Serialize(alert.DetectionMetadata) : (object)DBNull.Value,
                alert.Id);
        }
    }

    private async Task UpdateModelAsync(Models.AlertModels model, Alert alert, CancellationToken cancellationToken)
    {
        model.Status = MapStatus(alert.Status);
        model.Severity = MapSeverity(alert.Severity);
        model.AcknowledgedBy = alert.AcknowledgedBy;
        model.AcknowledgedAt = alert.AcknowledgedAt;
        model.ResolvedBy = alert.ResolvedBy;
        model.ResolvedAt = alert.ResolvedAt;
        model.ResolutionNotes = alert.ResolutionNotes;
        model.UpdatedAt = alert.UpdatedAt;
        
        await SetExtendedFieldsAsync(model, alert);
    }

    private AlertSeverityLevel MapSeverity(Models.AlertSeverityModels severity)
    {
        return severity switch
        {
            Models.AlertSeverityModels.Critical => AlertSeverityLevel.Critical,
            Models.AlertSeverityModels.High => AlertSeverityLevel.High,
            Models.AlertSeverityModels.Medium => AlertSeverityLevel.Medium,
            Models.AlertSeverityModels.Low => AlertSeverityLevel.Low,
            _ => AlertSeverityLevel.Info
        };
    }

    private Models.AlertSeverityModels MapSeverity(AlertSeverityLevel severity)
    {
        return severity switch
        {
            AlertSeverityLevel.Critical => Models.AlertSeverityModels.Critical,
            AlertSeverityLevel.High => Models.AlertSeverityModels.High,
            AlertSeverityLevel.Medium => Models.AlertSeverityModels.Medium,
            AlertSeverityLevel.Low => Models.AlertSeverityModels.Low,
            _ => Models.AlertSeverityModels.Info
        };
    }

    private AlertStatus MapStatus(Models.AlertStatusModels status)
    {
        return status switch
        {
            Models.AlertStatusModels.New => AlertStatus.New,
            Models.AlertStatusModels.Acknowledged => AlertStatus.Acknowledged,
            Models.AlertStatusModels.InProgress => AlertStatus.InProgress,
            Models.AlertStatusModels.Resolved => AlertStatus.Resolved,
            Models.AlertStatusModels.FalsePositive => AlertStatus.FalsePositive,
            Models.AlertStatusModels.Closed => AlertStatus.Closed,
            _ => AlertStatus.New
        };
    }

    private Models.AlertStatusModels MapStatus(AlertStatus status)
    {
        return status switch
        {
            AlertStatus.New => Models.AlertStatusModels.New,
            AlertStatus.Acknowledged => Models.AlertStatusModels.Acknowledged,
            AlertStatus.InProgress => Models.AlertStatusModels.InProgress,
            AlertStatus.Resolved => Models.AlertStatusModels.Resolved,
            AlertStatus.FalsePositive => Models.AlertStatusModels.FalsePositive,
            AlertStatus.Closed => Models.AlertStatusModels.Closed,
            _ => Models.AlertStatusModels.New
        };
    }
}
