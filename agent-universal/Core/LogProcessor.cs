using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.Agent.Core
{
    /// <summary>
    /// Log processor implementing ManageEngine EventLog Analyzer processing pipeline:
    /// Raw Logs → Security Filters → Parser → Enrichment → Indexing → Correlation
    /// </summary>
    public class LogProcessor : IAsyncDisposable
    {
        private readonly ILogger<LogProcessor> _logger;
        private readonly List<ILogFilter> _securityFilters = new();
        private readonly List<ILogEnricher> _enrichers = new();
        private readonly List<ILogCorrelator> _correlators = new();
        private readonly Dictionary<string, List<LogEntry>> _correlationBuffer = new();
        private readonly Timer _correlationTimer;
        private readonly object _processingLock = new();

        public bool IsProcessing { get; private set; }
        public long ProcessedLogs { get; private set; }
        public long FilteredLogs { get; private set; }

        public event EventHandler<LogProcessedEventArgs>? LogProcessed;
        public event EventHandler<CorrelationDetectedEventArgs>? CorrelationDetected;

        public LogProcessor(ILogger<LogProcessor> logger)
        {
            _logger = logger;
            
            // Initialize default security filters (ManageEngine pattern)
            InitializeSecurityFilters();
            
            // Initialize enrichers
            InitializeEnrichers();
            
            // Initialize correlators
            InitializeCorrelators();
            
            // Setup correlation timer (check every 30 seconds)
            _correlationTimer = new Timer(ProcessCorrelations, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
        }

        /// <summary>
        /// Process logs through the ManageEngine pipeline
        /// </summary>
        public async Task<ProcessedLogBatch> ProcessLogBatchAsync(IEnumerable<LogEntry> logs)
        {
            var processedBatch = new ProcessedLogBatch();
            
            try
            {
                lock (_processingLock)
                {
                    IsProcessing = true;
                }

                foreach (var log in logs)
                {
                    try
                    {
                        var processedLog = await ProcessSingleLogAsync(log);
                        if (processedLog != null)
                        {
                            processedBatch.ProcessedLogs.Add(processedLog);
                            ProcessedLogs++;
                        }
                        else
                        {
                            FilteredLogs++;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing log: {Message}", log.Message);
                        processedBatch.Errors.Add($"Error processing log: {ex.Message}");
                    }
                }

                // Add to correlation buffer
                AddToCorrelationBuffer(processedBatch.ProcessedLogs);

                processedBatch.ProcessingTime = DateTime.UtcNow;
                processedBatch.TotalProcessed = processedBatch.ProcessedLogs.Count;
                
                // Fire event
                LogProcessed?.Invoke(this, new LogProcessedEventArgs { ProcessedBatch = processedBatch });
                
                return processedBatch;
            }
            finally
            {
                lock (_processingLock)
                {
                    IsProcessing = false;
                }
            }
        }

        /// <summary>
        /// Process a single log through the complete pipeline
        /// </summary>
        private async Task<LogEntry?> ProcessSingleLogAsync(LogEntry log)
        {
            // Step 1: Apply security filters (ManageEngine pattern)
            if (!await ApplySecurityFiltersAsync(log))
            {
                return null; // Log filtered out
            }

            // Step 2: Parse and normalize (ManageEngine parsing pattern)
            await ParseAndNormalizeAsync(log);

            // Step 3: Enrich with context (ManageEngine enrichment pattern)
            await EnrichLogAsync(log);

            // Step 4: Create search index (ManageEngine indexing pattern)
            await CreateSearchIndexAsync(log);

            // Step 5: Generate integrity hash
            await GenerateLogHashAsync(log);

            return log;
        }

        /// <summary>
        /// Apply security-focused filters (ManageEngine pattern)
        /// Filters events that provide security information vs routine activities
        /// </summary>
        private async Task<bool> ApplySecurityFiltersAsync(LogEntry log)
        {
            foreach (var filter in _securityFilters)
            {
                if (!await filter.ShouldProcessAsync(log))
                {
                    _logger.LogDebug("Log filtered by {FilterName}: {LogMessage}", filter.Name, log.Message);
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Parse and normalize log following ManageEngine's parsing pattern
        /// Breaks down logs into structured, searchable components
        /// </summary>
        private async Task ParseAndNormalizeAsync(LogEntry log)
        {
            // Extract structured data based on log type
            if (log is WindowsLogEntry windowsLog)
            {
                await ParseWindowsEventAsync(windowsLog);
            }
            else if (log is SyslogEntry syslogEntry)
            {
                await ParseSyslogAsync(syslogEntry);
            }
            else if (log is IISLogEntry iisLog)
            {
                await ParseIISLogAsync(iisLog);
            }

            // Normalize timestamp to UTC
            if (log.Timestamp.Kind != DateTimeKind.Utc)
            {
                log.Timestamp = log.Timestamp.ToUniversalTime();
            }

            // Normalize level
            log.Level = NormalizeLogLevel(log.Level);
        }

        /// <summary>
        /// Enrich log with additional context (ManageEngine enrichment pattern)
        /// </summary>
        private async Task EnrichLogAsync(LogEntry log)
        {
            foreach (var enricher in _enrichers)
            {
                await enricher.EnrichAsync(log);
            }
        }

        /// <summary>
        /// Create search index for fast querying (ManageEngine search pattern)
        /// </summary>
        private async Task CreateSearchIndexAsync(LogEntry log)
        {
            var indexBuilder = new StringBuilder();
            
            // Add main fields to index
            indexBuilder.Append($"{log.Source} {log.Level} {log.Category} {log.Message} ");
            
            // Add properties to index
            foreach (var prop in log.Properties)
            {
                indexBuilder.Append($"{prop.Key}:{prop.Value} ");
            }
            
            // Add parsed fields
            if (!string.IsNullOrEmpty(log.ComputerName))
                indexBuilder.Append($"computer:{log.ComputerName} ");
            if (!string.IsNullOrEmpty(log.Username))
                indexBuilder.Append($"user:{log.Username} ");
            if (!string.IsNullOrEmpty(log.IpAddress))
                indexBuilder.Append($"ip:{log.IpAddress} ");

            log.SearchIndex = indexBuilder.ToString().ToLowerInvariant().Trim();
        }

        /// <summary>
        /// Generate integrity hash for log verification
        /// </summary>
        private async Task GenerateLogHashAsync(LogEntry log)
        {
            var logData = JsonSerializer.Serialize(new
            {
                log.Timestamp,
                log.Source,
                log.Level,
                log.Message,
                log.EventId,
                log.ComputerName,
                log.Username
            });

            using var sha256 = SHA256.Create();
            var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(logData));
            log.LogHash = Convert.ToBase64String(hashBytes);
        }

        /// <summary>
        /// Add logs to correlation buffer for attack chain detection
        /// </summary>
        private void AddToCorrelationBuffer(List<LogEntry> logs)
        {
            foreach (var log in logs)
            {
                var key = $"{log.ComputerName}_{log.Username}";
                if (!_correlationBuffer.ContainsKey(key))
                {
                    _correlationBuffer[key] = new List<LogEntry>();
                }
                
                _correlationBuffer[key].Add(log);
                
                // Keep only last 100 logs per key
                if (_correlationBuffer[key].Count > 100)
                {
                    _correlationBuffer[key].RemoveRange(0, 50);
                }
            }
        }

        /// <summary>
        /// Process correlations to detect attack chains (ManageEngine correlation pattern)
        /// </summary>
        private void ProcessCorrelations(object? state)
        {
            try
            {
                foreach (var correlator in _correlators)
                {
                    foreach (var bufferEntry in _correlationBuffer)
                    {
                        var correlations = correlator.DetectCorrelations(bufferEntry.Value);
                        foreach (var correlation in correlations)
                        {
                            CorrelationDetected?.Invoke(this, new CorrelationDetectedEventArgs
                            {
                                Correlation = correlation,
                                DetectedAt = DateTime.UtcNow
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing correlations");
            }
        }

        #region Initialization Methods

        private void InitializeSecurityFilters()
        {
            _securityFilters.AddRange(new ILogFilter[]
            {
                new SecurityRelevanceFilter(),
                new EventIdFilter(),
                new CriticalSystemFilter()
            });
        }

        private void InitializeEnrichers()
        {
            _enrichers.AddRange(new ILogEnricher[]
            {
                new GeoLocationEnricher(),
                new ThreatIntelligenceEnricher(),
                new AssetEnricher()
            });
        }

        private void InitializeCorrelators()
        {
            _correlators.AddRange(new ILogCorrelator[]
            {
                new AuthenticationCorrelator(),
                new PrivilegeEscalationCorrelator(),
                new LateralMovementCorrelator()
            });
        }

        #endregion

        #region Helper Methods

        private async Task ParseWindowsEventAsync(WindowsLogEntry log)
        {
            // Extract Windows-specific information
            if (log.Properties.ContainsKey("TargetUserName"))
            {
                log.TargetUserName = log.Properties["TargetUserName"]?.ToString();
                log.Username = log.TargetUserName;
            }

            if (log.Properties.ContainsKey("WorkstationName"))
            {
                log.WorkstationName = log.Properties["WorkstationName"]?.ToString();
                log.ComputerName = log.WorkstationName;
            }

            if (log.Properties.ContainsKey("IpAddress"))
            {
                log.IpAddress = log.Properties["IpAddress"]?.ToString();
            }
        }

        private async Task ParseSyslogAsync(SyslogEntry log)
        {
            // Parse syslog structured data
            log.ComputerName = log.Hostname;
            log.ProcessName = log.AppName;
        }

        private async Task ParseIISLogAsync(IISLogEntry log)
        {
            // Parse IIS-specific fields
            log.IpAddress = log.ClientIP;
            log.Username = log.Username;
            log.ComputerName = log.Properties.ContainsKey("ServerName") 
                ? log.Properties["ServerName"]?.ToString() 
                : Environment.MachineName;
        }

        private string NormalizeLogLevel(string level)
        {
            return level?.ToUpperInvariant() switch
            {
                "VERBOSE" or "TRACE" or "DEBUG" => "Debug",
                "INFO" or "INFORMATION" => "Information",
                "WARN" or "WARNING" => "Warning",
                "ERR" or "ERROR" => "Error",
                "CRIT" or "CRITICAL" or "FATAL" => "Critical",
                _ => "Information"
            };
        }

        #endregion

        public async ValueTask DisposeAsync()
        {
            _correlationTimer?.Dispose();
            _correlationBuffer.Clear();
        }
    }

    #region Supporting Classes and Interfaces

    public class ProcessedLogBatch
    {
        public List<LogEntry> ProcessedLogs { get; set; } = new();
        public List<string> Errors { get; set; } = new();
        public DateTime ProcessingTime { get; set; }
        public int TotalProcessed { get; set; }
    }

    public class LogProcessedEventArgs : EventArgs
    {
        public ProcessedLogBatch ProcessedBatch { get; set; } = new();
    }

    public class CorrelationDetectedEventArgs : EventArgs
    {
        public LogCorrelation Correlation { get; set; } = new();
        public DateTime DetectedAt { get; set; }
    }

    public interface ILogFilter
    {
        string Name { get; }
        Task<bool> ShouldProcessAsync(LogEntry log);
    }

    public interface ILogEnricher
    {
        string Name { get; }
        Task EnrichAsync(LogEntry log);
    }

    public interface ILogCorrelator
    {
        string Name { get; }
        IEnumerable<LogCorrelation> DetectCorrelations(IEnumerable<LogEntry> logs);
    }

    public class LogCorrelation
    {
        public string CorrelationId { get; set; } = Guid.NewGuid().ToString();
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public string Severity { get; set; } = "Medium";
        public List<LogEntry> RelatedLogs { get; set; } = new();
        public DateTime DetectedAt { get; set; } = DateTime.UtcNow;
        public Dictionary<string, object> Properties { get; set; } = new();
    }

    #region Default Filter Implementations

    public class SecurityRelevanceFilter : ILogFilter
    {
        public string Name => "Security Relevance Filter";

        public async Task<bool> ShouldProcessAsync(LogEntry log)
        {
            // Only process logs with High or Critical security relevance
            return log.SecurityRelevance == "High" || log.SecurityRelevance == "Critical" || log.SecurityRelevance == "Medium";
        }
    }

    public class EventIdFilter : ILogFilter
    {
        public string Name => "Event ID Filter";
        private readonly HashSet<string> _securityEventIds = new()
        {
            "4624", "4625", "4648", "4720", "4726", "4740", "4672", "4673", "1102", "4608", "4609"
        };

        public async Task<bool> ShouldProcessAsync(LogEntry log)
        {
            if (string.IsNullOrEmpty(log.EventId))
                return true; // Process non-Windows events

            return _securityEventIds.Contains(log.EventId);
        }
    }

    public class CriticalSystemFilter : ILogFilter
    {
        public string Name => "Critical System Filter";

        public async Task<bool> ShouldProcessAsync(LogEntry log)
        {
            // Always process Critical level logs
            return log.Level != "Debug" && log.Level != "Verbose";
        }
    }

    #endregion

    #region Default Enricher Implementations

    public class GeoLocationEnricher : ILogEnricher
    {
        public string Name => "GeoLocation Enricher";

        public async Task EnrichAsync(LogEntry log)
        {
            if (!string.IsNullOrEmpty(log.IpAddress) && !IsPrivateIP(log.IpAddress))
            {
                // In production, this would call a GeoIP service
                log.Properties["GeoLocation"] = "Unknown";
                log.Properties["Country"] = "Unknown";
            }
        }

        private bool IsPrivateIP(string ip)
        {
            // Simple private IP check
            return ip.StartsWith("192.168.") || ip.StartsWith("10.") || ip.StartsWith("172.");
        }
    }

    public class ThreatIntelligenceEnricher : ILogEnricher
    {
        public string Name => "Threat Intelligence Enricher";

        public async Task EnrichAsync(LogEntry log)
        {
            if (!string.IsNullOrEmpty(log.IpAddress))
            {
                // In production, this would check threat intelligence feeds
                log.Properties["ThreatIntelligence"] = "Clean";
            }
        }
    }

    public class AssetEnricher : ILogEnricher
    {
        public string Name => "Asset Enricher";

        public async Task EnrichAsync(LogEntry log)
        {
            if (!string.IsNullOrEmpty(log.ComputerName))
            {
                // In production, this would look up asset information
                log.Properties["AssetCriticality"] = "Medium";
                log.Properties["AssetOwner"] = "Unknown";
            }
        }
    }

    #endregion

    #region Default Correlator Implementations

    public class AuthenticationCorrelator : ILogCorrelator
    {
        public string Name => "Authentication Correlator";

        public IEnumerable<LogCorrelation> DetectCorrelations(IEnumerable<LogEntry> logs)
        {
            var sortedLogs = logs.OrderBy(l => l.Timestamp).ToList();
            var correlations = new List<LogCorrelation>();

            // Detect multiple failed logins followed by success
            for (int i = 0; i < sortedLogs.Count - 3; i++)
            {
                var failedLogins = sortedLogs.Skip(i).Take(3)
                    .Where(l => l.EventId == "4625").ToList();

                if (failedLogins.Count >= 2)
                {
                    var successLogin = sortedLogs.Skip(i + 3).FirstOrDefault(l => l.EventId == "4624");
                    if (successLogin != null)
                    {
                        correlations.Add(new LogCorrelation
                        {
                            Name = "Brute Force Attack",
                            Description = "Multiple failed logins followed by successful login",
                            Severity = "High",
                            RelatedLogs = failedLogins.Concat(new[] { successLogin }).ToList()
                        });
                    }
                }
            }

            return correlations;
        }
    }

    public class PrivilegeEscalationCorrelator : ILogCorrelator
    {
        public string Name => "Privilege Escalation Correlator";

        public IEnumerable<LogCorrelation> DetectCorrelations(IEnumerable<LogEntry> logs)
        {
            var correlations = new List<LogCorrelation>();
            
            // Detect privilege escalation patterns
            var privilegeLogs = logs.Where(l => l.EventId == "4672" || l.EventId == "4673").ToList();
            if (privilegeLogs.Count > 5) // Threshold for suspicious privilege use
            {
                correlations.Add(new LogCorrelation
                {
                    Name = "Privilege Escalation",
                    Description = "Excessive privilege use detected",
                    Severity = "High",
                    RelatedLogs = privilegeLogs
                });
            }

            return correlations;
        }
    }

    public class LateralMovementCorrelator : ILogCorrelator
    {
        public string Name => "Lateral Movement Correlator";

        public IEnumerable<LogCorrelation> DetectCorrelations(IEnumerable<LogEntry> logs)
        {
            var correlations = new List<LogCorrelation>();
            
            // Detect lateral movement patterns (same user, different computers)
            var logonLogs = logs.Where(l => l.EventId == "4624").ToList();
            var computerGroups = logonLogs.GroupBy(l => l.Username)
                .Where(g => g.Select(l => l.ComputerName).Distinct().Count() > 2);

            foreach (var group in computerGroups)
            {
                correlations.Add(new LogCorrelation
                {
                    Name = "Lateral Movement",
                    Description = $"User {group.Key} logged into multiple computers",
                    Severity = "Medium",
                    RelatedLogs = group.ToList()
                });
            }

            return correlations;
        }
    }

    #endregion

    #endregion
} 