using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Hubs;

namespace Backend.Workers;

/// <summary>
/// Background service that aggregates dashboard metrics and pushes them
/// to connected frontend clients via SignalR every 5 seconds.
/// This replaces the raw log push pattern with pre-computed chart data,
/// so the frontend doesn't have to re-query and re-aggregate on every render.
/// </summary>
public class DashboardAggregatorWorker : BackgroundService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly IHubContext<SiemHub> _hubContext;
    private readonly ILogger<DashboardAggregatorWorker> _logger;

    private static readonly TimeSpan PushInterval = TimeSpan.FromSeconds(5);

    // In-memory counters updated by the gRPC ingestion path
    private static readonly ConcurrentQueue<LogIngestionEvent> _recentIngestions = new();

    public DashboardAggregatorWorker(
        IServiceScopeFactory scopeFactory,
        IHubContext<SiemHub> hubContext,
        ILogger<DashboardAggregatorWorker> logger)
    {
        _scopeFactory = scopeFactory;
        _hubContext = hubContext;
        _logger = logger;
    }

    /// <summary>
    /// Called by the gRPC SiemService when logs are ingested. 
    /// This feeds the real-time counters without querying the database.
    /// </summary>
    public static void RecordIngestion(string agentId, int count, string source, string level)
    {
        _recentIngestions.Enqueue(new LogIngestionEvent
        {
            Timestamp = DateTime.UtcNow,
            AgentId = agentId,
            Count = count,
            Source = source,
            Level = level
        });

        // Trim events older than 5 minutes to prevent memory growth
        while (_recentIngestions.TryPeek(out var oldest) && 
               oldest.Timestamp < DateTime.UtcNow.AddMinutes(-5))
        {
            _recentIngestions.TryDequeue(out _);
        }
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("DashboardAggregatorWorker started. Pushing metrics every {Interval}s", PushInterval.TotalSeconds);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(PushInterval, stoppingToken);
                await PushDashboardStateAsync(stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in DashboardAggregatorWorker");
            }
        }

        _logger.LogInformation("DashboardAggregatorWorker stopped");
    }

    private async Task PushDashboardStateAsync(CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();

        var now = DateTime.UtcNow;
        var oneHourAgo = now.AddHours(-1);
        var twentyFourHoursAgo = now.AddHours(-24);

        // 1. Logs per second (from in-memory recent ingestions, last 60s)
        var recentEvents = _recentIngestions.ToArray();
        var last60s = recentEvents.Where(e => e.Timestamp >= now.AddSeconds(-60)).ToArray();
        var logsPerSecond = last60s.Sum(e => e.Count) / 60.0;

        // 2. Event distribution by category (last 1 hour, from DB)
        var eventDistribution = await db.LogEntries
            .Where(l => l.Timestamp >= oneHourAgo)
            .GroupBy(l => l.Category ?? l.Source ?? "Other")
            .Select(g => new { Category = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .Take(8)
            .ToDictionaryAsync(x => x.Category, x => x.Count, ct);

        // 3. Severity distribution (last 1 hour)
        var severityDistribution = await db.LogEntries
            .Where(l => l.Timestamp >= oneHourAgo)
            .GroupBy(l => l.Level ?? "Unknown")
            .Select(g => new { Level = g.Key, Count = g.Count() })
            .ToDictionaryAsync(x => x.Level, x => x.Count, ct);

        // 4. Hourly event counts for the timeline chart (last 24h)
        var hourlyEvents = await db.LogEntries
            .Where(l => l.Timestamp >= twentyFourHoursAgo)
            .GroupBy(l => new { l.Timestamp.Year, l.Timestamp.Month, l.Timestamp.Day, l.Timestamp.Hour })
            .Select(g => new
            {
                Hour = g.Key.Hour,
                Day = g.Key.Day,
                Count = g.Count(),
                Errors = g.Count(x => x.Level == "Error" || x.Level == "Critical" || x.Level == "Warning")
            })
            .ToListAsync(ct);

        // Build the 24-hour timeline
        var timeline = new List<object>();
        for (int i = 23; i >= 0; i--)
        {
            var targetHour = now.AddHours(-i);
            var match = hourlyEvents.FirstOrDefault(h => h.Day == targetHour.Day && h.Hour == targetHour.Hour);
            timeline.Add(new
            {
                time = targetHour.ToString("HH:00"),
                events = match?.Count ?? 0,
                anomalies = match?.Errors ?? 0
            });
        }

        // 5. Total counts
        var totalLogs1h = await db.LogEntries.CountAsync(l => l.Timestamp >= oneHourAgo, ct);
        var totalLogs24h = await db.LogEntries.CountAsync(l => l.Timestamp >= twentyFourHoursAgo, ct);

        // 6. Top source IPs (from metadata if available)
        var topSourceIps = await db.LogEntries
            .Where(l => l.Timestamp >= oneHourAgo && l.IPAddress != null && l.IPAddress != "")
            .GroupBy(l => l.IPAddress)
            .Select(g => new { Ip = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .Take(5)
            .ToListAsync(ct);

        var dashboardState = new
        {
            timestamp = now.ToString("o"),
            logsPerSecond = Math.Round(logsPerSecond, 1),
            totalLogs1h,
            totalLogs24h,
            eventDistribution,
            severityDistribution,
            timeline,
            topSourceIps = topSourceIps.Select(x => new { ip = x.Ip, count = x.Count }),
            networkLoad = $"{Math.Round(logsPerSecond, 0)}/s"
        };

        await _hubContext.Clients.All.SendAsync("ReceiveDashboardState", dashboardState, ct);
    }

    private class LogIngestionEvent
    {
        public DateTime Timestamp { get; set; }
        public string AgentId { get; set; } = "";
        public int Count { get; set; }
        public string Source { get; set; } = "";
        public string Level { get; set; } = "";
    }
}
