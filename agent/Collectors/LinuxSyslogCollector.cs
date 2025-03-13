using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Collector for Linux syslog files
    /// </summary>
    public class LinuxSyslogCollector : ILogCollector
    {
        private readonly ILogger<LinuxSyslogCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private Timer? _collectionTimer;
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings;
        private readonly List<string> _logFiles = new();
        private readonly Dictionary<string, long> _filePositions = new();
        private int _maxLinesPerRead = 1000;
        private bool _startFromEnd = true;
        
        // Regex pattern for standard syslog format
        private static readonly Regex SyslogPattern = new(
            @"^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^:\s]+)(?:\[(\d+)\])?:?\s*(.*)$",
            RegexOptions.Compiled);

        /// <summary>
        /// Event raised when a log is collected
        /// </summary>
        public event EventHandler<NormalizedLogEntry>? LogCollected;

        /// <summary>
        /// Gets the type of the collector
        /// </summary>
        public string CollectorType => "LinuxSyslog";

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
        /// Creates a new instance of the LinuxSyslogCollector
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="normalizer">Log normalizer</param>
        public LinuxSyslogCollector(ILogger<LinuxSyslogCollector> logger, ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
            _settings = new CollectorSettings { Type = "LinuxSyslog" };
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
                if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    _errorMessage = "Linux syslog collector can only be used on Linux";
                    _logger.LogError(_errorMessage);
                    return false;
                }

                _settings = settings ?? throw new ArgumentNullException(nameof(settings));

                // Parse log files
                if (settings.Properties.TryGetValue("LogFiles", out var logFilesStr))
                {
                    _logFiles.Clear();
                    _logFiles.AddRange(logFilesStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
                }
                else
                {
                    // Default to common syslog files
                    _logFiles.AddRange(new[] { "/var/log/syslog", "/var/log/auth.log" });
                }

                // Parse max lines per read
                if (settings.Properties.TryGetValue("MaxLinesPerRead", out var maxLinesStr))
                {
                    if (int.TryParse(maxLinesStr, out var maxLines))
                    {
                        _maxLinesPerRead = maxLines;
                    }
                }

                // Parse start from end
                if (settings.Properties.TryGetValue("StartFromEnd", out var startFromEndStr))
                {
                    if (bool.TryParse(startFromEndStr, out var startFromEnd))
                    {
                        _startFromEnd = startFromEnd;
                    }
                }

                // Initialize file positions
                foreach (var logFile in _logFiles)
                {
                    if (File.Exists(logFile))
                    {
                        if (_startFromEnd)
                        {
                            // Start from the end of the file
                            var fileInfo = new FileInfo(logFile);
                            _filePositions[logFile] = fileInfo.Length;
                        }
                        else
                        {
                            // Start from the beginning of the file
                            _filePositions[logFile] = 0;
                        }
                    }
                    else
                    {
                        _logger.LogWarning("Log file does not exist: {LogFile}", logFile);
                    }
                }

                _logger.LogInformation("Initialized Linux syslog collector with {Count} log files", _logFiles.Count);
                return true;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error initializing Linux syslog collector");
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
                _logger.LogWarning("Linux syslog collector is already running");
                return Task.CompletedTask;
            }

            try
            {
                _logger.LogInformation("Starting Linux syslog collector");
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
                _logger.LogError(ex, "Error starting Linux syslog collector");
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
                _logger.LogWarning("Linux syslog collector is not running");
                return Task.CompletedTask;
            }

            try
            {
                _logger.LogInformation("Stopping Linux syslog collector");
                _collectionTimer?.Dispose();
                _collectionTimer = null;
                _isRunning = false;
                _isPaused = false;
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error stopping Linux syslog collector");
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
                _logger.LogWarning("Linux syslog collector is not running or already paused");
                return Task.CompletedTask;
            }

            try
            {
                _logger.LogInformation("Pausing Linux syslog collector");
                _collectionTimer?.Change(Timeout.Infinite, Timeout.Infinite);
                _isPaused = true;
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error pausing Linux syslog collector");
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
                _logger.LogWarning("Linux syslog collector is not running or not paused");
                return Task.CompletedTask;
            }

            try
            {
                _logger.LogInformation("Resuming Linux syslog collector");
                var interval = TimeSpan.FromSeconds(_settings.IntervalSeconds);
                _collectionTimer?.Change(TimeSpan.Zero, interval);
                _isPaused = false;
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error resuming Linux syslog collector");
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
                _logger.LogDebug("Collecting Linux syslog on demand");
                return await Task.Run(() => CollectLogs(), cancellationToken);
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Error collecting Linux syslog on demand");
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
                _logger.LogError(ex, "Error collecting Linux syslog");
            }
        }

        /// <summary>
        /// Collects logs from syslog files
        /// </summary>
        /// <returns>The number of logs collected</returns>
        private int CollectLogs()
        {
            int totalCollected = 0;

            foreach (var logFile in _logFiles)
            {
                try
                {
                    if (!File.Exists(logFile))
                    {
                        _logger.LogWarning("Log file does not exist: {LogFile}", logFile);
                        continue;
                    }

                    // Get the current file position
                    _filePositions.TryGetValue(logFile, out var position);

                    // Get the current file size
                    var fileInfo = new FileInfo(logFile);
                    var fileSize = fileInfo.Length;

                    // If the file has been truncated, start from the beginning
                    if (fileSize < position)
                    {
                        _logger.LogInformation("Log file has been truncated: {LogFile}", logFile);
                        position = 0;
                    }

                    // If there's nothing new to read, skip this file
                    if (position >= fileSize)
                    {
                        continue;
                    }

                    // Read the file
                    using var fileStream = new FileStream(logFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                    fileStream.Seek(position, SeekOrigin.Begin);
                    using var reader = new StreamReader(fileStream);

                    int linesRead = 0;
                    string? line;
                    while ((line = reader.ReadLine()) != null && linesRead < _maxLinesPerRead)
                    {
                        if (!string.IsNullOrWhiteSpace(line))
                        {
                            // Process the line
                            var rawLog = ParseSyslogLine(line, logFile);
                            var normalizedLog = _normalizer.Normalize(rawLog);
                            
                            // Raise event
                            LogCollected?.Invoke(this, normalizedLog);
                            
                            linesRead++;
                        }
                    }

                    // Update the file position
                    _filePositions[logFile] = fileStream.Position;
                    
                    totalCollected += linesRead;
                    _logger.LogDebug("Collected {Count} lines from {LogFile}", linesRead, logFile);
                }
                catch (Exception ex)
                {
                    _errorMessage = ex.Message;
                    _logger.LogError(ex, "Error collecting logs from {LogFile}", logFile);
                }
            }

            return totalCollected;
        }

        /// <summary>
        /// Parses a syslog line into raw log data
        /// </summary>
        /// <param name="line">Syslog line</param>
        /// <param name="logFile">Log file path</param>
        /// <returns>Raw log data</returns>
        private RawLogData ParseSyslogLine(string line, string logFile)
        {
            try
            {
                var match = SyslogPattern.Match(line);
                if (match.Success)
                {
                    var timestamp = match.Groups[1].Value;
                    var hostname = match.Groups[2].Value;
                    var program = match.Groups[3].Value;
                    var pid = match.Groups[4].Success ? match.Groups[4].Value : string.Empty;
                    var message = match.Groups[5].Value;

                    // Parse timestamp (assuming current year)
                    if (DateTimeOffset.TryParse($"{DateTime.Now.Year} {timestamp}", out var parsedTimestamp))
                    {
                        // If the parsed date is in the future, it's probably from the previous year
                        if (parsedTimestamp > DateTimeOffset.Now)
                        {
                            parsedTimestamp = parsedTimestamp.AddYears(-1);
                        }

                        var rawLog = new RawLogData
                        {
                            Id = Guid.NewGuid().ToString(),
                            Timestamp = parsedTimestamp.DateTime,
                            Source = program,
                            SourceType = "LinuxSyslog",
                            SourceIdentifier = logFile,
                            CollectorType = "LinuxSyslog",
                            LogLevel = DetermineSyslogLevel(message),
                            Content = message,
                            Metadata = new Dictionary<string, string>
                            {
                                ["Hostname"] = hostname,
                                ["Program"] = program,
                                ["PID"] = pid,
                                ["LogFile"] = Path.GetFileName(logFile)
                            }
                        };

                        return rawLog;
                    }
                }

                // If we couldn't parse the line, just return it as-is
                return new RawLogData
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTimeOffset.Now.DateTime,
                    Source = "LinuxSyslog",
                    SourceType = "LinuxSyslog",
                    SourceIdentifier = logFile,
                    CollectorType = "LinuxSyslog",
                    LogLevel = "Information",
                    Content = line,
                    Metadata = new Dictionary<string, string>
                    {
                        ["LogFile"] = Path.GetFileName(logFile)
                    }
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing syslog line: {Line}", line);
                
                // Return a fallback log entry
                return new RawLogData
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTimeOffset.Now.DateTime,
                    Source = "LinuxSyslog",
                    SourceType = "LinuxSyslog",
                    SourceIdentifier = logFile,
                    CollectorType = "LinuxSyslog",
                    LogLevel = "Error",
                    Content = line,
                    Metadata = new Dictionary<string, string>
                    {
                        ["LogFile"] = Path.GetFileName(logFile),
                        ["ParseError"] = ex.Message
                    }
                };
            }
        }

        /// <summary>
        /// Determines the syslog level based on the message content
        /// </summary>
        /// <param name="message">Syslog message</param>
        /// <returns>Log level</returns>
        private string DetermineSyslogLevel(string message)
        {
            // Check for common level indicators in the message
            var lowerMessage = message.ToLowerInvariant();
            
            if (lowerMessage.Contains("emerg") || lowerMessage.Contains("panic") || lowerMessage.Contains("fatal"))
                return "Critical";
            
            if (lowerMessage.Contains("alert"))
                return "Critical";
            
            if (lowerMessage.Contains("crit"))
                return "Critical";
            
            if (lowerMessage.Contains("error") || lowerMessage.Contains("err"))
                return "Error";
            
            if (lowerMessage.Contains("warn"))
                return "Warning";
            
            if (lowerMessage.Contains("notice"))
                return "Information";
            
            if (lowerMessage.Contains("info"))
                return "Information";
            
            if (lowerMessage.Contains("debug"))
                return "Debug";
            
            // Default to Information
            return "Information";
        }
    }
} 