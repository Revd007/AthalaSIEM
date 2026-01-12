using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Collectors;

public class WindowsEventLogCollector : ICollector
{
    private readonly ILogger<WindowsEventLogCollector> _logger;
    private readonly List<string> _logNames;
    private readonly bool _enabled;
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    private List<Task>? _watcherTasks;

    public WindowsEventLogCollector(
        ILogger<WindowsEventLogCollector> logger,
        IEnumerable<string> logNames,
        bool enabled = true)
    {
        _logger = logger;
        _logNames = logNames.ToList();
        _enabled = enabled;
    }

    public string Name => "WindowsEventLogCollector";
    public string SourceType => "WindowsEventLog";
    public bool IsEnabled => _enabled && OperatingSystem.IsWindows();

    public event EventHandler<IRawEvent>? EventCollected;

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (!IsEnabled)
            return Task.CompletedTask;

        _watcherTasks = new List<Task>();

        foreach (var logName in _logNames)
        {
            var task = Task.Run(() => WatchEventLog(logName, _cancellationTokenSource.Token), cancellationToken);
            _watcherTasks.Add(task);
        }

        _logger.LogInformation("Started Windows Event Log collector for logs: {LogNames}", string.Join(", ", _logNames));
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _cancellationTokenSource.Cancel();
        _watcherTasks?.ForEach(t => t.Wait(TimeSpan.FromSeconds(5)));
        return Task.CompletedTask;
    }

    private void WatchEventLog(string logName, CancellationToken cancellationToken)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                var query = new EventLogQuery(logName, PathType.LogName);
                using var reader = new EventLogReader(query);

                EventRecord? lastEvent = null;
                while (!cancellationToken.IsCancellationRequested)
                {
                    try
                    {
                        var evt = reader.ReadEvent();
                        if (evt == null)
                        {
                            await Task.Delay(1000, cancellationToken);
                            continue;
                        }

                        if (lastEvent != null && evt.RecordId <= lastEvent.RecordId)
                        {
                            evt.Dispose();
                            await Task.Delay(1000, cancellationToken);
                            continue;
                        }

                        lastEvent = evt;

                        var rawEvent = new RawEvent
                        {
                            Id = Guid.NewGuid().ToString(),
                            Timestamp = evt.TimeCreated ?? DateTime.UtcNow,
                            CollectorName = Name,
                            SourceType = SourceType,
                            RawData = System.Text.Encoding.UTF8.GetBytes(evt.ToXml()),
                            Metadata = new Dictionary<string, string>
                            {
                                ["log_name"] = logName,
                                ["event_id"] = evt.Id.ToString(),
                                ["level"] = evt.LevelDisplayName ?? evt.Level.ToString() ?? "Unknown",
                                ["machine_name"] = evt.MachineName ?? string.Empty
                            }
                        };

                        EventCollected?.Invoke(this, rawEvent);
                        evt.Dispose();
                        
                        await Task.Delay(10, cancellationToken);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error reading from event log {LogName}", logName);
                        await Task.Delay(5000, cancellationToken);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to watch event log {LogName}", logName);
            }
        }, cancellationToken);
    }
}
