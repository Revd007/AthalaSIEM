using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Xml;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Collectors;

public class WindowsEventLogCollector : ICollector
{
    private readonly ILogger<WindowsEventLogCollector> _logger;
    private readonly List<string> _logNames;
    private readonly bool _enabled;
    private readonly CancellationTokenSource _cts = new();
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

        // Link the external cancellation token so both external stop and our own _cts work
        _watcherTasks = new List<Task>();
        var linked = CancellationTokenSource.CreateLinkedTokenSource(_cts.Token, cancellationToken);

        foreach (var logName in _logNames)
        {
            _watcherTasks.Add(Task.Run(() => WatchEventLogAsync(logName, linked.Token)));
        }

        _logger.LogInformation("Started Windows Event Log collector for logs: {LogNames}", string.Join(", ", _logNames));
        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _cts.Cancel();

        if (_watcherTasks is { Count: > 0 })
        {
            // Wait for all watchers to finish gracefully, with a timeout
            var allDone = Task.WhenAll(_watcherTasks);
            await Task.WhenAny(allDone, Task.Delay(TimeSpan.FromSeconds(5), CancellationToken.None));
        }
    }

    private static string GetLevelDisplayName(EventRecord evt)
    {
        var level = evt.Level;
        return level switch
        {
            1 => "Critical",
            2 => "Error",
            3 => "Warning",
            4 => "Information",
            5 => "Verbose",
            0 or null => "Information",
            _ => level.ToString() ?? "Unknown"
        };
    }

    /// <summary>
    /// Safely get TaskDisplayName - can throw EventLogNotFoundException
    /// when the event provider metadata is not registered on this machine.
    /// </summary>
    private static string GetTaskDisplayNameSafe(EventRecord evt)
    {
        try
        {
            return evt.TaskDisplayName ?? string.Empty;
        }
        catch (EventLogNotFoundException)
        {
            return evt.Task?.ToString() ?? string.Empty;
        }
        catch
        {
            return string.Empty;
        }
    }

    /// <summary>
    /// Extracts human-readable message from Windows Event Record.
    /// Uses FormatDescription() first, then falls back to XML parsing or descriptive text.
    /// This is CRITICAL - without this, logs show "(no message)" in the dashboard.
    /// </summary>
    private string ExtractHumanReadableMessage(EventRecord evt)
    {
        // Method 1: Use FormatDescription() - this is the standard way to get human-readable text
        try
        {
            var formatted = evt.FormatDescription();
            if (!string.IsNullOrWhiteSpace(formatted))
            {
                return formatted;
            }
        }
        catch (Exception ex)
        {
            // FormatDescription() can fail if:
            // - Event provider metadata is not installed
            // - Event uses parameterized messages without proper DLL
            _logger.LogDebug(ex, "FormatDescription() failed for event {EventId}, trying fallback methods", evt.Id);
        }

        // Method 2: Try to extract from XML EventData
        try
        {
            var xml = evt.ToXml();
            var xmlDoc = new XmlDocument();
            xmlDoc.LoadXml(xml);
            var nsManager = new XmlNamespaceManager(xmlDoc.NameTable);
            nsManager.AddNamespace("evt", "http://schemas.microsoft.com/win/2004/08/events/event");
            
            // Try to find EventData/Data elements and construct a readable message
            var eventDataNode = xmlDoc.SelectSingleNode("//evt:EventData", nsManager);
            if (eventDataNode != null)
            {
                var dataNodes = eventDataNode.SelectNodes("evt:Data", nsManager);
                if (dataNodes != null && dataNodes.Count > 0)
                {
                    var dataValues = new List<string>();
                    foreach (XmlNode dataNode in dataNodes)
                    {
                        var name = dataNode.Attributes?["Name"]?.Value;
                        var value = dataNode.InnerText;
                        if (!string.IsNullOrEmpty(value))
                        {
                            if (!string.IsNullOrEmpty(name))
                            {
                                dataValues.Add($"{name}={value}");
                            }
                            else
                            {
                                dataValues.Add(value);
                            }
                        }
                    }
                    if (dataValues.Count > 0)
                    {
                        return $"Event {evt.Id} ({evt.ProviderName ?? "Unknown"}): {string.Join(", ", dataValues.Take(5))}";
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "XML parsing fallback failed for event {EventId}", evt.Id);
        }

        // Method 3: Construct descriptive message from available fields
        var taskName = GetTaskDisplayNameSafe(evt);
        var providerName = evt.ProviderName ?? "Unknown";
        var levelName = GetLevelDisplayName(evt);
        
        return $"Event ID {evt.Id} ({levelName}) from {providerName}" + 
               (!string.IsNullOrEmpty(taskName) ? $": {taskName}" : string.Empty);
    }

    private async Task WatchEventLogAsync(string logName, CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                // Build query with XPath filter directly in constructor (.NET 8 correct API)
                var queryString = "*[System[(Level=0 or Level=1 or Level=2 or Level=3 or Level=4 or Level=5)]]";
                var query = new EventLogQuery(logName, PathType.LogName, queryString);
                query.ReverseDirection = true; // Start reading from newest events

                using var reader = new EventLogReader(query);
                long lastRecordId = 0;

                // Read the most recent event to establish our bookmark position
                var newestEvt = reader.ReadEvent();
                if (newestEvt != null)
                {
                    lastRecordId = newestEvt.RecordId ?? 0;
                    newestEvt.Dispose();
                }

                _logger.LogDebug("Event log {LogName}: starting from record ID {RecordId}", logName, lastRecordId);

                // Now switch to forward-reading for new events
                while (!ct.IsCancellationRequested)
                {
                    try
                    {
                        // Build a query that only reads events newer than our last seen record
                        var forwardQuery = lastRecordId > 0
                            ? new EventLogQuery(logName, PathType.LogName,
                                $"*[System[EventRecordID>{lastRecordId}]]")
                            : new EventLogQuery(logName, PathType.LogName);

                        using var forwardReader = new EventLogReader(forwardQuery);
                        EventRecord? evt;
                        int batchCount = 0;

                        while ((evt = forwardReader.ReadEvent()) != null && !ct.IsCancellationRequested)
                        {
                            using (evt)
                            {
                                if (evt.RecordId.HasValue && evt.RecordId.Value > lastRecordId)
                                {
                                    lastRecordId = evt.RecordId.Value;
                                }

                                // Extract human-readable message using FormatDescription()
                                // This is CRITICAL - without this, logs show "(no message)" in the dashboard
                                string humanReadableMessage = ExtractHumanReadableMessage(evt);

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
                                        ["level"] = GetLevelDisplayName(evt),
                                        ["task"] = GetTaskDisplayNameSafe(evt),
                                        ["machine_name"] = evt.MachineName ?? string.Empty,
                                        ["message"] = humanReadableMessage, // CRITICAL: Human-readable message
                                        ["provider_name"] = evt.ProviderName ?? string.Empty,
                                        ["record_id"] = evt.RecordId?.ToString() ?? "0"
                                    }
                                };

                                EventCollected?.Invoke(this, rawEvent);
                                batchCount++;
                            }
                        }

                        if (batchCount > 0)
                        {
                            _logger.LogDebug("Event log {LogName}: processed {Count} new events", logName, batchCount);
                        }

                        // Poll interval - wait before checking for new events again
                        await Task.Delay(2000, ct);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                    catch (InvalidOperationException)
                    {
                        _logger.LogDebug("Event log reader state invalid for {LogName}, reconnecting", logName);
                        break;
                    }
                    catch (EventLogException ex)
                    {
                        _logger.LogWarning(ex, "Event log query error for {LogName}, will retry", logName);
                        await SafeDelay(5000, ct);
                        break; // Break inner loop to recreate reader
                    }
                    catch (Exception ex) when (!ct.IsCancellationRequested)
                    {
                        _logger.LogError(ex, "Error reading from event log {LogName}", logName);
                        await SafeDelay(5000, ct);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Normal shutdown - exit silently
                break;
            }
            catch (EventLogNotFoundException)
            {
                _logger.LogWarning("Event log '{LogName}' not found on this system, skipping", logName);
                break;
            }
            catch (UnauthorizedAccessException)
            {
                _logger.LogWarning("Access denied to event log '{LogName}'. Agent must run as Administrator for Security log. Skipping.", logName);
                break;
            }
            catch (Exception ex) when (!ct.IsCancellationRequested)
            {
                _logger.LogError(ex, "Failed to watch event log {LogName}, will retry in 10s", logName);
                await SafeDelay(10000, ct);
            }
        }

        _logger.LogInformation("Event log watcher for '{LogName}' stopped", logName);
    }

    /// <summary>
    /// Delay that swallows OperationCanceledException on shutdown.
    /// </summary>
    private static async Task SafeDelay(int ms, CancellationToken ct)
    {
        try
        {
            await Task.Delay(ms, ct);
        }
        catch (OperationCanceledException)
        {
            // Expected during shutdown
        }
    }
}
