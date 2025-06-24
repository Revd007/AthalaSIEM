using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.Runtime.Versioning;
using AthalaSIEM.Agent.Core;
using AthalaSIEM.UniversalAgent.Models;
using Core = AthalaSIEM.Agent.Core;
using Microsoft.Extensions.Logging;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Windows Event Log Collector following ManageEngine EventLog Analyzer patterns
    /// Supports both agentless (WMI/DCOM/RPC) and agent-based collection methods
    /// Implements filtering, parsing, and enrichment following enterprise patterns
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class WindowsEventLogCollector : ILogCollector
    {
        private readonly ILogger<WindowsEventLogCollector> _logger;

        public string CollectorName => "Windows Event Log";
        public Core.OperatingSystem SupportedOS => Core.OperatingSystem.Windows;
        public bool IsActive { get; private set; }
        public long LogsCollected { get; private set; }

        private readonly List<LogEntry> _collectedLogs = new List<LogEntry>();
        private readonly Dictionary<string, EventLogQuery> _logQueries = new Dictionary<string, EventLogQuery>();
        private readonly List<EventLogFilter> _securityFilters = new List<EventLogFilter>();
        private readonly CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();

        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        public WindowsEventLogCollector(ILogger<WindowsEventLogCollector> logger)
        {
            _logger = logger;
        }

        public Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("Initializing Windows Event Log Collector...");
                
                // Check if running with administrator privileges
                var isAdmin = IsRunningAsAdministrator();
                if (!isAdmin)
                {
                    _logger.LogWarning("⚠️ NOT running as Administrator - Security Event Log will be unavailable!");
                    _logger.LogWarning("For full SIEM functionality, run as Administrator. Continuing with available logs...");
                    
                    // Continue initialization but exclude Security log
                    CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                    { 
                        Exception = new UnauthorizedAccessException("Administrator privileges required for Security logs"),
                        Message = "⚠️ Security Event Log unavailable - not running as Administrator",
                        Source = CollectorName
                    });
                }
                else
                {
                    _logger.LogInformation("✅ Running as Administrator - Full SIEM functionality available");
                }

                // Initialize security-focused filters (ManageEngine pattern)
                try
                {
                    InitializeSecurityFilters();
                    _logger.LogDebug("Security filters initialized successfully");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to initialize security filters");
                    throw;
                }
                
                // Setup event log queries for different log sources
                try
                {
                    InitializeEventLogQueries(config, isAdmin);
                    _logger.LogDebug("Event log queries initialized successfully");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to initialize event log queries");
                    throw;
                }
                
                _logger.LogInformation("Windows Event Log Collector initialized successfully with {QueryCount} queries", _logQueries.Count);
                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize Windows Event Log Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex, 
                    Message = $"Failed to initialize Windows Event Log Collector: {ex.Message}",
                    Source = CollectorName
                });
                return Task.FromResult(false);
            }
        }

        public async Task StartCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = true;
            
            // Start multiple collection tasks for different log sources
            var collectionTasks = new List<Task>();
            
            foreach (var queryPair in _logQueries)
            {
                collectionTasks.Add(Task.Run(() => CollectEventsFromSource(queryPair.Key, queryPair.Value), cancellationToken));
            }
            
            await Task.WhenAll(collectionTasks);
        }

        public Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = false;
            _cancellationTokenSource.Cancel();
            
            return Task.CompletedTask;
        }

        public Task<IEnumerable<LogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default)
        {
            var logs = _collectedLogs.Take(batchSize).ToList();
            _collectedLogs.RemoveRange(0, logs.Count);
            return Task.FromResult<IEnumerable<LogEntry>>(logs);
        }

        public Task<CollectorHealth> GetHealthAsync()
        {
            return Task.FromResult(new CollectorHealth
            {
                IsHealthy = true,
                Status = IsActive ? "Running" : "Stopped",
                LogsCollected = LogsCollected,
                LastCollection = DateTime.UtcNow,
                Metrics = new Dictionary<string, object>
                {
                    ["ActiveQueries"] = _logQueries.Count,
                    ["SecurityFilters"] = _securityFilters.Count,
                    ["BufferedLogs"] = _collectedLogs.Count
                }
            });
        }

        /// <summary>
        /// Initialize security-focused filters following ManageEngine patterns
        /// Filters events that provide security information vs routine activities
        /// </summary>
        private void InitializeSecurityFilters()
        {
            _securityFilters.AddRange(new[]
            {
                // Authentication Events
                new EventLogFilter { EventId = 4624, Description = "Successful Logon", SecurityRelevance = "High" },
                new EventLogFilter { EventId = 4625, Description = "Failed Logon", SecurityRelevance = "High" },
                new EventLogFilter { EventId = 4648, Description = "Logon using explicit credentials", SecurityRelevance = "Medium" },
                
                // Account Management
                new EventLogFilter { EventId = 4720, Description = "User account created", SecurityRelevance = "High" },
                new EventLogFilter { EventId = 4726, Description = "User account deleted", SecurityRelevance = "High" },
                new EventLogFilter { EventId = 4740, Description = "User account locked", SecurityRelevance = "Medium" },
                
                // Privilege Use
                new EventLogFilter { EventId = 4672, Description = "Special privileges assigned", SecurityRelevance = "High" },
                new EventLogFilter { EventId = 4673, Description = "Privileged service called", SecurityRelevance = "Medium" },
                
                // System Events
                new EventLogFilter { EventId = 1102, Description = "Audit log cleared", SecurityRelevance = "Critical" },
                new EventLogFilter { EventId = 4608, Description = "Windows starting up", SecurityRelevance = "Medium" },
                new EventLogFilter { EventId = 4609, Description = "Windows shutting down", SecurityRelevance = "Medium" }
            });
        }

        /// <summary>
        /// Initialize event log queries for different Windows log sources
        /// Following SIEM standard pattern - collect ALL events by default
        /// </summary>
        private void InitializeEventLogQueries(Dictionary<string, object> config, bool isAdmin)
        {
            string[] logSources;
            
            // Handle different configuration object types
            if (config.ContainsKey("LogSources"))
            {
                var logSourcesObj = config["LogSources"];
                logSources = ParseStringArrayFromConfig(logSourcesObj, new[] { "Security", "System", "Application" });
            }
            else
            {
                logSources = new[] { "Security", "System", "Application" };
            }

            // If not running as admin, exclude Security log
            if (!isAdmin)
            {
                logSources = logSources.Where(s => !s.Equals("Security", StringComparison.OrdinalIgnoreCase)).ToArray();
                _logger.LogWarning("Excluding Security log due to insufficient privileges. Available logs: {LogSources}", 
                    string.Join(", ", logSources));
            }

            var collectAllEvents = config.ContainsKey("CollectAllEvents") 
                ? ParseBooleanFromConfig(config["CollectAllEvents"], true)
                : true;

            var enableSecurityFiltering = config.ContainsKey("EnableSecurityFiltering") 
                ? ParseBooleanFromConfig(config["EnableSecurityFiltering"], false)
                : false;

            foreach (var logSource in logSources)
            {
                EventLogQuery query;
                
                if (collectAllEvents && !enableSecurityFiltering)
                {
                    // Collect ALL events like Splunk/Wazuh/ELK
                    query = new EventLogQuery(logSource, PathType.LogName, "*");
                }
                else if (enableSecurityFiltering && _securityFilters.Any())
                {
                    // Use security filtering (legacy mode)
                    var securityEventIds = string.Join(" or ", _securityFilters.Select(f => $"EventID={f.EventId}"));
                    query = new EventLogQuery(logSource, PathType.LogName, $"*[System[{securityEventIds}]]");
                }
                else
                {
                    // Default: collect all events
                    query = new EventLogQuery(logSource, PathType.LogName, "*");
                }
                
                _logQueries[logSource] = query;
                _logger.LogInformation("Configured query for {LogSource} log", logSource);
            }

            _logger.LogInformation("Initialized {QueryCount} event log queries", _logQueries.Count);
        }

        /// <summary>
        /// Collect events from specific log source with filtering and parsing
        /// Implements ManageEngine's parsing pattern: normalize, parse, index
        /// </summary>
        private void CollectEventsFromSource(string sourceName, EventLogQuery query)
        {
            try
            {
                using var reader = new EventLogReader(query);
                
                EventRecord eventRecord;
                while ((eventRecord = reader.ReadEvent()) != null && IsActive)
                {
                    // Check if we should filter events
                    var config = new Dictionary<string, object>(); // This should come from initialization
                    var enableSecurityFiltering = false; // Default to collect all
                    var relevantFilter = _securityFilters.FirstOrDefault(f => f.EventId == eventRecord.Id);
                    
                    // If security filtering is enabled and no relevant filter found, skip
                    if (enableSecurityFiltering && relevantFilter == null) continue;

                    // Parse and normalize event (SIEM standard pattern)
                    var logEntry = ParseAndNormalizeEvent(eventRecord, sourceName, relevantFilter ?? new EventLogFilter());
                    
                    // Enrich with context (SIEM enrichment pattern)
                    EnrichLogEntry(logEntry, eventRecord);
                    
                    _collectedLogs.Add(logEntry);
                    LogsCollected++;
                    
                    LogCollected?.Invoke(this, new LogCollectedEventArgs 
                    { 
                        Logs = new[] { logEntry },
                        Source = sourceName,
                        CollectionTime = DateTime.UtcNow
                    });

                    // Prevent memory overflow
                    if (_collectedLogs.Count > 10000)
                    {
                        _collectedLogs.RemoveRange(0, 5000);
                    }
                }
            }
            catch (UnauthorizedAccessException ex)
            {
                if (sourceName.Equals("Security", StringComparison.OrdinalIgnoreCase))
                {
                    // Critical SIEM issue - Security log access denied
                    CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                    { 
                        Exception = ex,
                        Source = sourceName,
                        Message = "🚨 CRITICAL: Cannot access Security Event Log! SIEM functionality compromised. " +
                                 "Agent must run with Administrator privileges to collect security events. " +
                                 "Without Security logs, this is NOT a functional SIEM agent!"
                    });
                }
                else
                {
                    CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                    { 
                        Exception = ex,
                        Source = sourceName,
                        Message = $"Access denied to {sourceName} Event Log. Administrator privileges required."
                    });
                }
            }
            catch (Exception ex)
            {
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex,
                    Source = sourceName,
                    Message = $"Error collecting from {sourceName}"
                });
            }
        }

        /// <summary>
        /// Parse and normalize event following SIEM standard pattern
        /// Breaks down events into structured, searchable components
        /// </summary>
        private LogEntry ParseAndNormalizeEvent(EventRecord eventRecord, string sourceName, EventLogFilter filter)
        {
            var logEntry = new LogEntry
            {
                Timestamp = eventRecord.TimeCreated ?? DateTime.UtcNow,
                Source = sourceName,
                Level = MapEventLevel(eventRecord.Level),
                Message = eventRecord.FormatDescription() ?? "No description",
                EventId = eventRecord.Id.ToString(),
                Category = filter?.Description ?? eventRecord.LogName ?? sourceName,
                SecurityRelevance = filter?.SecurityRelevance ?? "Medium"
            };

            // Extract structured data (device name, username, etc.)
            if (eventRecord.Properties != null)
            {
                var properties = new Dictionary<string, object>();
                for (int i = 0; i < eventRecord.Properties.Count; i++)
                {
                    var property = eventRecord.Properties[i];
                    properties[$"Property_{i}"] = property?.Value?.ToString() ?? "";
                }
                logEntry.Properties = properties;
            }

            return logEntry;
        }

        /// <summary>
        /// Enrich log entry with additional context for analysis
        /// Following ManageEngine's enrichment pattern
        /// </summary>
        private void EnrichLogEntry(LogEntry logEntry, EventRecord eventRecord)
        {
            // Add computer name
            if (!string.IsNullOrEmpty(eventRecord.MachineName))
            {
                logEntry.Properties["ComputerName"] = eventRecord.MachineName;
            }

            // Add process information if available
            if (eventRecord.ProcessId.HasValue)
            {
                logEntry.Properties["ProcessId"] = eventRecord.ProcessId.Value;
            }

            // Add thread information if available
            if (eventRecord.ThreadId.HasValue)
            {
                logEntry.Properties["ThreadId"] = eventRecord.ThreadId.Value;
            }

            // Add user context if available
            if (eventRecord.UserId != null)
            {
                logEntry.Properties["UserId"] = eventRecord.UserId.ToString();
            }

            // Add correlation for attack chain detection
            logEntry.Properties["CollectionTime"] = DateTime.UtcNow;
            logEntry.Properties["CollectorVersion"] = "1.0.0";
        }

        private string MapEventLevel(byte? level)
        {
            return level switch
            {
                1 => "Critical",
                2 => "Error", 
                3 => "Warning",
                4 => "Information",
                5 => "Verbose",
                _ => "Information"
            };
        }

        /// <summary>
        /// Check if the current process is running with Administrator privileges
        /// </summary>
        private bool IsRunningAsAdministrator()
        {
            try
            {
                var identity = System.Security.Principal.WindowsIdentity.GetCurrent();
                var principal = new System.Security.Principal.WindowsPrincipal(identity);
                return principal.IsInRole(System.Security.Principal.WindowsBuiltInRole.Administrator);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Parses a string array from various configuration object types.
        /// Handles JsonElement arrays, object arrays, and comma-separated strings.
        /// </summary>
        /// <param name="configObj">The configuration object to parse.</param>
        /// <param name="defaultValue">Default value if parsing fails.</param>
        /// <returns>Parsed string array.</returns>
        private string[] ParseStringArrayFromConfig(object configObj, string[] defaultValue)
        {
            try
            {
                if (configObj is string[] stringArray)
                {
                    return stringArray;
                }
                else if (configObj is System.Text.Json.JsonElement jsonElement && jsonElement.ValueKind == System.Text.Json.JsonValueKind.Array)
                {
                    var result = new List<string>();
                    foreach (var element in jsonElement.EnumerateArray())
                    {
                        if (element.ValueKind == System.Text.Json.JsonValueKind.String)
                        {
                            var value = element.GetString();
                            if (!string.IsNullOrWhiteSpace(value))
                            {
                                result.Add(value);
                            }
                        }
                    }
                    return result.ToArray();
                }
                else if (configObj is object[] objectArray)
                {
                    return objectArray.Select(o => o?.ToString()).Where(s => !string.IsNullOrWhiteSpace(s)).ToArray()!;
                }
                else if (configObj is List<object> objectList)
                {
                    return objectList.Select(o => o?.ToString()).Where(s => !string.IsNullOrWhiteSpace(s)).ToArray()!;
                }
                else if (configObj is string configString)
                {
                    if (configString.Contains(','))
                    {
                        return configString.Split(',').Select(s => s.Trim()).Where(s => !string.IsNullOrWhiteSpace(s)).ToArray();
                    }
                    else
                    {
                        return new[] { configString.Trim() };
                    }
                }
                else if (configObj is System.Collections.IEnumerable enumerable && !(configObj is string))
                {
                    var result = new List<string>();
                    foreach (var item in enumerable)
                    {
                        if (item != null)
                        {
                            var itemString = item.ToString();
                            if (!string.IsNullOrWhiteSpace(itemString))
                            {
                                result.Add(itemString);
                            }
                        }
                    }
                    return result.ToArray();
                }

                _logger.LogWarning("Unable to parse configuration object of type {Type}, using default value", configObj?.GetType().Name);
                return defaultValue;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error parsing string array from configuration, using default value");
                return defaultValue;
            }
        }

        /// <summary>
        /// Parses a boolean value from various configuration object types.
        /// Handles JsonElement booleans, string representations, and direct boolean values.
        /// </summary>
        /// <param name="configObj">The configuration object to parse.</param>
        /// <param name="defaultValue">Default value if parsing fails.</param>
        /// <returns>Parsed boolean value.</returns>
        private bool ParseBooleanFromConfig(object configObj, bool defaultValue)
        {
            try
            {
                return configObj switch
                {
                    null => defaultValue,
                    bool boolValue => boolValue,
                    string stringValue => bool.TryParse(stringValue, out var result) ? result : defaultValue,
                    System.Text.Json.JsonElement jsonElement when jsonElement.ValueKind == System.Text.Json.JsonValueKind.True => true,
                    System.Text.Json.JsonElement jsonElement when jsonElement.ValueKind == System.Text.Json.JsonValueKind.False => false,
                    System.Text.Json.JsonElement jsonElement when jsonElement.ValueKind == System.Text.Json.JsonValueKind.String => 
                        bool.TryParse(jsonElement.GetString(), out var result) ? result : defaultValue,
                    _ => defaultValue
                };
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to parse boolean from config object {Type}: {Value}", 
                    configObj?.GetType().Name ?? "null", configObj?.ToString() ?? "null");
                return defaultValue;
            }
        }

        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            _cancellationTokenSource?.Dispose();
        }
    }

    /// <summary>
    /// Event filter for security-focused log collection
    /// Based on ManageEngine's filtering patterns
    /// </summary>
    public class EventLogFilter
    {
        public int EventId { get; set; }
        public string Description { get; set; } = "";
        public string SecurityRelevance { get; set; } = "Medium"; // Critical, High, Medium, Low
        public string Category { get; set; } = "";
        public bool Enabled { get; set; } = true;
    }
}
