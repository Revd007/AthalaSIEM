using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Models;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Collector for Windows Event Logs
    /// </summary>
    public class WindowsEventLogCollector : ILogCollector
    {
        private readonly ILogger<WindowsEventLogCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private Timer? _collectionTimer;
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings;
        private readonly List<string> _eventLogs = new();
        private readonly Dictionary<string, string> _lastEventIds = new();
        private bool _startFromEnd = true;
        private int _maxEvents = 1000;
        private bool _includeErrors = true;
        private bool _includeWarnings = true;
        private bool _includeInformation = true;

        /// <summary>
        /// Event raised when a log is collected
        /// </summary>
        public event EventHandler<NormalizedLogEntry>? LogCollected;

        /// <summary>
        /// Gets the type of the collector
        /// </summary>
        public string CollectorType => "WindowsEventLog";

        /// <summary>
        /// Gets the status of the collector
        /// </summary>
        public CollectorStatus Status
        {
            get
            {
                if (!string.IsNullOrEmpty(_errorMessage))
                    return CollectorStatus.Error;
                if (_isPaused)
                    return CollectorStatus.Paused;
                if (_isRunning)
                    return CollectorStatus.Running;
                return CollectorStatus.Stopped;
            }
        }

        /// <summary>
        /// Gets the error message if the collector is in an error state
        /// </summary>
        public string ErrorMessage => _errorMessage;

        /// <summary>
        /// Creates a new instance of the WindowsEventLogCollector
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="normalizer">Log normalizer</param>
        public WindowsEventLogCollector(ILogger<WindowsEventLogCollector> logger, ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
            _settings = new CollectorSettings { Type = "WindowsEventLog" };
        }

        /// <summary>
        /// Initializes the collector with the provided settings
        /// </summary>
        /// <param name="settings">Collector settings</param>
        /// <returns>True if initialization was successful, otherwise false</returns>
        public bool Initialize(CollectorSettings settings)
        {
            try
            {
                _settings = settings ?? throw new ArgumentNullException(nameof(settings));

                // Parse event logs
                if (settings.Properties.TryGetValue("EventLogs", out var eventLogsStr))
                {
                    _eventLogs.Clear();
                    _eventLogs.AddRange(eventLogsStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
                }

                // Parse start from end
                if (settings.Properties.TryGetValue("StartFromEnd", out var startFromEndStr))
                {
                    if (bool.TryParse(startFromEndStr, out var startFromEnd))
                    {
                        _startFromEnd = startFromEnd;
                    }
                }

                // Parse max events
                if (settings.Properties.TryGetValue("MaxEvents", out var maxEventsStr))
                {
                    if (int.TryParse(maxEventsStr, out var maxEvents))
                    {
                        _maxEvents = maxEvents;
                    }
                }

                // Parse include errors
                if (settings.Properties.TryGetValue("IncludeErrors", out var includeErrorsStr))
                {
                    if (bool.TryParse(includeErrorsStr, out var includeErrors))
                    {
                        _includeErrors = includeErrors;
                    }
                }

                // Parse include warnings
                if (settings.Properties.TryGetValue("IncludeWarnings", out var includeWarningsStr))
                {
                    if (bool.TryParse(includeWarningsStr, out var includeWarnings))
                    {
                        _includeWarnings = includeWarnings;
                    }
                }

                // Parse include information
                if (settings.Properties.TryGetValue("IncludeInformation", out var includeInformationStr))
                {
                    if (bool.TryParse(includeInformationStr, out var includeInformation))
                    {
                        _includeInformation = includeInformation;
                    }
                }

                _logger.LogInformation("Initialized Windows Event Log collector with {Count} event logs", _eventLogs.Count);
                return true;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error initializing Windows Event Log collector");
                return false;
            }
        }

        /// <summary>
        /// Starts the log collector
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        public Task StartAsync()
        {
            if (_isRunning)
            {
                _logger.LogWarning("Windows Event Log collector is already running");
                return Task.CompletedTask;
            }

            try
            {
                _logger.LogInformation("Starting Windows Event Log collector");
                _isRunning = true;
                _isPaused = false;
                _errorMessage = string.Empty;

                // Start collection timer
                var interval = TimeSpan.FromSeconds(_settings.IntervalSeconds);
                _collectionTimer = new Timer(CollectLogsCallback, null, TimeSpan.Zero, interval);

                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _isRunning = false;
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error starting Windows Event Log collector");
                return Task.FromException(ex);
            }
        }

        /// <summary>
        /// Stops the log collector
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        public Task StopAsync()
        {
            if (!_isRunning)
            {
                _logger.LogWarning("Windows Event Log collector is not running");
                return Task.CompletedTask;
            }

            try
            {
                _logger.LogInformation("Stopping Windows Event Log collector");
                _collectionTimer?.Dispose();
                _collectionTimer = null;
                _isRunning = false;
                _isPaused = false;
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error stopping Windows Event Log collector");
                return Task.FromException(ex);
            }
        }

        /// <summary>
        /// Pauses the log collector
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        public Task PauseAsync()
        {
            if (!_isRunning || _isPaused)
            {
                _logger.LogWarning("Windows Event Log collector is not running or already paused");
                return Task.CompletedTask;
            }

            try
            {
                _logger.LogInformation("Pausing Windows Event Log collector");
                _collectionTimer?.Change(Timeout.Infinite, Timeout.Infinite);
                _isPaused = true;
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error pausing Windows Event Log collector");
                return Task.FromException(ex);
            }
        }

        /// <summary>
        /// Resumes the log collector
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        public Task ResumeAsync()
        {
            if (!_isRunning || !_isPaused)
            {
                _logger.LogWarning("Windows Event Log collector is not running or not paused");
                return Task.CompletedTask;
            }

            try
            {
                _logger.LogInformation("Resuming Windows Event Log collector");
                var interval = TimeSpan.FromSeconds(_settings.IntervalSeconds);
                _collectionTimer?.Change(TimeSpan.Zero, interval);
                _isPaused = false;
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error resuming Windows Event Log collector");
                return Task.FromException(ex);
            }
        }

        /// <summary>
        /// Collects logs on demand
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The number of logs collected</returns>
        public async Task<int> CollectLogsAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogDebug("Collecting Windows Event Logs on demand");
                return await Task.Run(() => CollectLogs(), cancellationToken);
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error collecting Windows Event Logs on demand");
                return 0;
            }
        }

        /// <summary>
        /// Callback for the collection timer
        /// </summary>
        /// <param name="state">Timer state</param>
        private void CollectLogsCallback(object? state)
        {
            try
            {
                CollectLogs();
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error collecting Windows Event Logs");
            }
        }

        /// <summary>
        /// Collects logs from Windows Event Logs
        /// </summary>
        /// <returns>The number of logs collected</returns>
        private int CollectLogs()
        {
            // Skip collection if not on Windows
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _logger.LogWarning("Windows Event Log collector can only be used on Windows");
                return 0;
            }

            int totalCollected = 0;

            foreach (var eventLog in _eventLogs)
            {
                try
                {
                    var query = BuildEventLogQuery(eventLog);
                    using var reader = new EventLogReader(query);

                    int collected = 0;
                    EventRecord? eventRecord;
                    while ((eventRecord = reader.ReadEvent()) != null && collected < _maxEvents)
                    {
                        using (eventRecord)
                        {
                            // Process the event
                            var rawLog = ConvertToRawLogData(eventRecord, eventLog);
                            var normalizedLog = _normalizer.Normalize(rawLog);
                            
                            // Raise event
                            LogCollected?.Invoke(this, normalizedLog);
                            
                            // Update last event ID
                            if (eventRecord.RecordId.HasValue)
                            {
                                _lastEventIds[eventLog] = eventRecord.RecordId.Value.ToString();
                            }
                            
                            collected++;
                        }
                    }

                    totalCollected += collected;
                    _logger.LogDebug("Collected {Count} events from {EventLog}", collected, eventLog);
                }
                catch (Exception ex)
                {
                    _errorMessage = ex.Message;
                    _logger.LogError(ex, "Error collecting events from {EventLog}", eventLog);
                }
            }

            return totalCollected;
        }

        /// <summary>
        /// Builds an event log query for the specified event log
        /// </summary>
        /// <param name="eventLog">Event log name</param>
        /// <returns>Event log query</returns>
        private EventLogQuery BuildEventLogQuery(string eventLog)
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _logger.LogWarning("Windows Event Log query can only be built on Windows");
                // Create a default query on non-Windows platforms - this won't be used but prevents null reference
                // The PathType.LogName is Windows-specific, but we're already checking the platform
#pragma warning disable CA1416 // Validate platform compatibility
                return new EventLogQuery(eventLog, PathType.LogName);
#pragma warning restore CA1416
            }

            // Create a basic query for all events - Windows only code
#pragma warning disable CA1416 // Validate platform compatibility
            var query = new EventLogQuery(eventLog, PathType.LogName);
#pragma warning restore CA1416
            
            // Build filter parts
            var filterParts = new List<string>();
            
            // Add event types to include
            var eventTypesToInclude = new List<string>();
            if (_includeErrors) eventTypesToInclude.Add("Level=2");
            if (_includeWarnings) eventTypesToInclude.Add("Level=3");
            if (_includeInformation) eventTypesToInclude.Add("Level=4");
            
            // Add event types filter if needed
            if (eventTypesToInclude.Count > 0)
            {
                filterParts.Add($"({string.Join(" or ", eventTypesToInclude)})");
            }
            
            // Add filter for events after a specific time if needed
            if (_lastEventIds.TryGetValue(eventLog, out string? lastId) && !string.IsNullOrEmpty(lastId))
            {
                filterParts.Add($"System/EventRecordID>{lastId}");
            }
            
            // Apply the filter to the query
            if (filterParts.Count > 0)
            {
                string queryString = $"*[System[{string.Join(" and ", filterParts)}]]";
                _logger.LogDebug("Event log query: {Query}", queryString);
                
                // Set the query string via reflection since the property might not be directly accessible
                var queryType = query.GetType();
                var queryProperty = queryType.GetProperty("QueryString");
                if (queryProperty != null)
                {
                    queryProperty.SetValue(query, queryString);
                }
                else
                {
                    _logger.LogWarning("Could not find QueryString property on EventLogQuery type");
                }
            }

            return query;
        }

        /// <summary>
        /// Converts an EventRecord to RawLogData
        /// </summary>
        /// <param name="eventRecord">Event record</param>
        /// <param name="eventLog">Event log name</param>
        /// <returns>Raw log data</returns>
        private RawLogData ConvertToRawLogData(EventRecord eventRecord, string eventLog)
        {
            // Ensure we're on Windows
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _logger.LogWarning("Windows Event Log conversion can only be done on Windows");
                return new RawLogData
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    Source = "WindowsEventLog",
                    SourceType = "WindowsEventLog",
                    SourceIdentifier = eventLog,
                    CollectorType = "WindowsEventLog",
                    LogLevel = "Error",
                    Content = "Platform not supported",
                    Metadata = new Dictionary<string, string>
                    {
                        ["EventLog"] = eventLog,
                        ["Error"] = "Windows Event Log collector can only be used on Windows"
                    }
                };
            }

            // Extract event details
            string message = eventRecord.FormatDescription() ?? string.Empty;
            string source = eventRecord.ProviderName ?? string.Empty;
            string level = GetEventLevel(eventRecord.Level);
            
            // Get timestamp - handle DateTime conversion correctly
            DateTime timestamp = DateTime.UtcNow; // Default to current time if we can't get the original
            if (eventRecord.TimeCreated.HasValue)
            {
                timestamp = eventRecord.TimeCreated.Value;
            }
            
            // Create raw log data
            var rawLog = new RawLogData
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = timestamp,
                Source = source,
                SourceType = "WindowsEventLog",
                SourceIdentifier = eventLog,
                CollectorType = "WindowsEventLog",
                LogLevel = level,
                Content = message,
                Metadata = new Dictionary<string, string>
                {
                    ["EventID"] = eventRecord.Id.ToString(),
                    ["RecordID"] = eventRecord.RecordId?.ToString() ?? "0",
                    ["EventLog"] = eventLog,
                    ["Task"] = eventRecord.TaskDisplayName ?? string.Empty,
                    ["Computer"] = eventRecord.MachineName ?? Environment.MachineName
                }
            };

            return rawLog;
        }

        /// <summary>
        /// Gets the event level as a string
        /// </summary>
        /// <param name="level">Event level</param>
        /// <returns>Event level as a string</returns>
        private string GetEventLevel(byte? level)
        {
            return level switch
            {
                1 => "Error",
                2 => "Warning",
                3 => "Information",
                4 => "Information",
                5 => "Verbose",
                0 => "Information",
                _ => "Unknown"
            };
        }
    }
} 