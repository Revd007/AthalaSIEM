using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using Backend.Models;
using Backend.Data;
using Microsoft.EntityFrameworkCore;
using System.Text.RegularExpressions;

namespace Backend.Services
{
    /// <summary>
    /// Enhanced Threat Intelligence Service with multi-collector integration
    /// </summary>
    public class ThreatIntelligenceService : IThreatIntelligenceService
    {
        private readonly ILogger<ThreatIntelligenceService> _logger;
        private readonly ApplicationDbContext _context;
        private readonly IConfiguration _configuration;
        private readonly Dictionary<string, ThreatIndicator> _threatCache = new();
        private readonly Dictionary<string, CollectorThreatProfile> _collectorProfiles = new();
        private readonly object _cacheLock = new();

        public ThreatIntelligenceService(
            ILogger<ThreatIntelligenceService> logger,
            ApplicationDbContext context,
            IConfiguration configuration)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

            InitializeCollectorProfiles();
        }

        public async Task<ThreatAnalysisResult> AnalyzeLogEntryAsync(LogEntryModels logEntry)
        {
            try
            {
                _logger.LogInformation("Analyzing log entry {LogId} for threats", logEntry.Id);

                var result = new ThreatAnalysisResult
                {
                    LogEntryId = logEntry.Id,
                    Timestamp = logEntry.Timestamp,
                    ThreatLevel = ThreatLevel.None,
                    ThreatScore = 0.0,
                    Indicators = new List<ThreatIndicatorMatch>(),
                    CollectorSpecificAnalysis = new Dictionary<string, object>()
                };

                // Determine collector type from log source
                var collectorType = DetermineCollectorType(logEntry.Source);
                var profile = _collectorProfiles.GetValueOrDefault(collectorType, new CollectorThreatProfile());

                // Analyze based on collector type
                switch (collectorType)
                {
                    case "Container":
                        await AnalyzeContainerThreats(logEntry, result, profile);
                        break;
                    case "CloudServices":
                        await AnalyzeCloudServiceThreats(logEntry, result, profile);
                        break;
                    case "Database":
                        await AnalyzeDatabaseThreats(logEntry, result, profile);
                        break;
                    case "IoT":
                        await AnalyzeIoTThreats(logEntry, result, profile);
                        break;
                    case "FileIntegrity":
                        await AnalyzeFileIntegrityThreats(logEntry, result, profile);
                        break;
                    default:
                        await AnalyzeGeneralThreats(logEntry, result, profile);
                        break;
                }

                // Apply threat score multiplier
                result.ThreatScore *= profile.ThreatScoreMultiplier;

                // Determine final threat level
                result.ThreatLevel = DetermineThreatLevel(result);

                _logger.LogInformation("Threat analysis completed for log {LogId}: Level={ThreatLevel}, Score={ThreatScore}", 
                    logEntry.Id, result.ThreatLevel, result.ThreatScore);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing log entry {LogId}", logEntry.Id);
                throw;
            }
        }

        public async Task<CollectorThreatSummary> GetCollectorThreatSummaryAsync(string collectorType, DateTime? since = null)
        {
            try
            {
                var sinceDate = since ?? DateTime.UtcNow.AddDays(-7);
                
                var threats = await _context.LogEntries
                    .Where(l => l.Source.Contains(collectorType) && l.Timestamp >= sinceDate)
                    .ToListAsync();

                var summary = new CollectorThreatSummary
                {
                    CollectorType = collectorType,
                    AnalysisPeriod = sinceDate,
                    TotalLogs = threats.Count,
                    ThreatsByLevel = new Dictionary<ThreatLevel, int>(),
                    TopThreatIndicators = new List<string>(),
                    RecommendedActions = new List<string>()
                };

                // Analyze threats by collector type
                switch (collectorType.ToLowerInvariant())
                {
                    case "container":
                        await AnalyzeContainerThreatsForSummary(threats, summary);
                        break;
                    case "cloudservices":
                        await AnalyzeCloudThreats(threats, summary);
                        break;
                    case "database":
                        await AnalyzeDatabaseThreatsForSummary(threats, summary);
                        break;
                    case "iot":
                        await AnalyzeIoTThreatsForSummary(threats, summary);
                        break;
                    case "fileintegrity":
                        await AnalyzeFIMThreats(threats, summary);
                        break;
                    default:
                        await AnalyzeGeneralThreats(threats, summary);
                        break;
                }

                return summary;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting threat summary for collector {CollectorType}", collectorType);
                throw;
            }
        }

        public async Task<List<ThreatCorrelation>> FindThreatCorrelationsAsync(TimeSpan timeWindow, int minimumOccurrences = 3)
        {
            try
            {
                var correlations = new List<ThreatCorrelation>();
                var cutoffTime = DateTime.UtcNow - timeWindow;

                // Find patterns across different collectors
                var recentLogs = await _context.LogEntries
                    .Where(l => l.Timestamp >= cutoffTime)
                    .OrderBy(l => l.Timestamp)
                    .ToListAsync();

                // Group by source patterns
                var sourceGroups = recentLogs
                    .GroupBy(l => GetSourcePattern(l.Source))
                    .Where(g => g.Count() >= minimumOccurrences);

                foreach (var group in sourceGroups)
                {
                    var correlation = new ThreatCorrelation
                    {
                        Pattern = group.Key,
                        Occurrences = group.Count(),
                        TimeWindow = timeWindow,
                        FirstSeen = group.Min(l => l.Timestamp),
                        LastSeen = group.Max(l => l.Timestamp),
                        CollectorsInvolved = group.Select(l => DetermineCollectorType(l.Source)).Distinct().ToList(),
                        SeverityDistribution = group.GroupBy(l => l.Level).ToDictionary(g => g.Key, g => g.Count()),
                        RecommendedActions = GenerateCorrelationRecommendations(group.Key, group.ToList())
                    };

                    correlations.Add(correlation);
                }

                return correlations.OrderByDescending(c => c.Occurrences).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error finding threat correlations");
                throw;
            }
        }

        private void InitializeCollectorProfiles()
        {
            _collectorProfiles["Container"] = new CollectorThreatProfile
            {
                HighRiskPatterns = new[]
                {
                    "privileged container", "docker escape", "container breakout",
                    "kubectl exec", "docker exec", "privilege escalation",
                    "unauthorized image", "malicious container"
                },
                SuspiciousActivities = new[]
                {
                    "container created at unusual time",
                    "excessive resource usage",
                    "network anomaly in container",
                    "file system changes in container"
                },
                ThreatScoreMultiplier = 1.5
            };

            _collectorProfiles["CloudServices"] = new CollectorThreatProfile
            {
                HighRiskPatterns = new[]
                {
                    "credential exposure", "unauthorized api call", "data exfiltration",
                    "privilege escalation", "account compromise", "resource hijacking",
                    "configuration change", "security group modification"
                },
                SuspiciousActivities = new[]
                {
                    "unusual login location",
                    "multiple failed authentications",
                    "API calls from new IP",
                    "bulk data download"
                },
                ThreatScoreMultiplier = 2.0
            };

            _collectorProfiles["Database"] = new CollectorThreatProfile
            {
                HighRiskPatterns = new[]
                {
                    "sql injection", "data exfiltration", "unauthorized access",
                    "privilege escalation", "bulk data export", "schema modification",
                    "user creation", "permission granted"
                },
                SuspiciousActivities = new[]
                {
                    "unusual query patterns",
                    "after-hours database access",
                    "large data exports",
                    "failed authentication spikes"
                },
                ThreatScoreMultiplier = 2.5
            };

            _collectorProfiles["IoT"] = new CollectorThreatProfile
            {
                HighRiskPatterns = new[]
                {
                    "device compromise", "firmware modification", "unauthorized communication",
                    "network scanning", "botnet activity", "device hijacking",
                    "protocol anomaly", "authentication bypass"
                },
                SuspiciousActivities = new[]
                {
                    "device communicating with unknown servers",
                    "unusual traffic patterns",
                    "sensor data anomalies",
                    "configuration changes"
                },
                ThreatScoreMultiplier = 1.8
            };

            _collectorProfiles["FileIntegrity"] = new CollectorThreatProfile
            {
                HighRiskPatterns = new[]
                {
                    "malware detected", "ransomware activity", "unauthorized modification",
                    "system file change", "backdoor installation", "rootkit activity",
                    "persistence mechanism", "lateral movement"
                },
                SuspiciousActivities = new[]
                {
                    "system files modified",
                    "executables in temp directories",
                    "hidden file creation",
                    "startup modifications"
                },
                ThreatScoreMultiplier = 2.2
            };
        }

        private string DetermineCollectorType(string source)
        {
            if (source.Contains("Container") || source.Contains("Docker") || source.Contains("Kubernetes"))
                return "Container";
            if (source.Contains("AWS") || source.Contains("Azure") || source.Contains("GCP") || source.Contains("CloudServices"))
                return "CloudServices";
            if (source.Contains("Database") || source.Contains("SQL") || source.Contains("MySQL") || source.Contains("PostgreSQL") || source.Contains("MongoDB"))
                return "Database";
            if (source.Contains("IoT") || source.Contains("Sensor") || source.Contains("SCADA") || source.Contains("Modbus") || source.Contains("MQTT"))
                return "IoT";
            if (source.Contains("FIM") || source.Contains("FileIntegrity"))
                return "FileIntegrity";
            
            return "General";
        }

        private async Task PerformGeneralThreatAnalysis(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            // Check for known malicious IPs, domains, hashes
            await CheckThreatIntelligenceFeeds(logEntry, result);
            
            // Check for suspicious patterns in message
            CheckSuspiciousPatterns(logEntry, result);
            
            // Check for anomalous behavior
            await CheckAnomalousBehavior(logEntry, result);
        }

        private async Task PerformCollectorSpecificAnalysis(LogEntryModels logEntry, ThreatAnalysisResult result, string collectorType)
        {
            if (!_collectorProfiles.TryGetValue(collectorType, out var profile))
                return;

            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            var details = logEntry.Details ?? "";

            // Check high-risk patterns
            foreach (var pattern in profile.HighRiskPatterns)
            {
                if (message.Contains(pattern.ToLowerInvariant()) || details.Contains(pattern.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "HighRiskPattern",
                        Value = pattern,
                        Source = collectorType,
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }

            // Check suspicious activities
            foreach (var activity in profile.SuspiciousActivities)
            {
                if (message.Contains(activity.ToLowerInvariant()) || details.Contains(activity.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "SuspiciousActivity",
                        Value = activity,
                        Source = collectorType,
                        Confidence = 0.6,
                        Severity = "Medium"
                    });
                }
            }

            // Perform collector-specific deep analysis
            switch (collectorType)
            {
                case "Container":
                    await AnalyzeContainerSpecificThreats(logEntry, result);
                    break;
                case "CloudServices":
                    await AnalyzeCloudSpecificThreats(logEntry, result);
                    break;
                case "Database":
                    await AnalyzeDatabaseSpecificThreats(logEntry, result);
                    break;
                case "IoT":
                    await AnalyzeIoTSpecificThreats(logEntry, result);
                    break;
                case "FileIntegrity":
                    await AnalyzeFIMSpecificThreats(logEntry, result);
                    break;
            }

            result.CollectorSpecificAnalysis[collectorType] = new
            {
                profile_applied = true,
                risk_multiplier = profile.ThreatScoreMultiplier,
                patterns_checked = profile.HighRiskPatterns.Length,
                activities_checked = profile.SuspiciousActivities.Length
            };
        }

        private Task AnalyzeContainerSpecificThreats(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            try
            {
                var details = JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.Details ?? "{}") ?? new Dictionary<string, object>();
                
                // Check for privileged containers
                if (details.ContainsKey("privileged") && details["privileged"].ToString() == "true")
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "PrivilegedContainer",
                        Value = "Container running in privileged mode",
                        Source = "Container",
                        Confidence = 0.9,
                        Severity = "High"
                    });
                }

                // Check for suspicious image sources
                if (details.ContainsKey("image") && IsUnknownImageSource(details["image"]?.ToString()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "UnknownImageSource",
                        Value = details["image"]?.ToString() ?? "Unknown",
                        Source = "Container",
                        Confidence = 0.7,
                        Severity = "Medium"
                    });
                }

                // Check for excessive resource usage
                if (details.ContainsKey("cpu_usage") && double.TryParse(details["cpu_usage"]?.ToString(), out var cpuUsage) && cpuUsage > 90)
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "ResourceAbuse",
                        Value = $"High CPU usage: {cpuUsage}%",
                        Source = "Container",
                        Confidence = 0.6,
                        Severity = "Medium"
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in container-specific threat analysis");
            }
            
            return Task.CompletedTask;
        }

        private async Task AnalyzeCloudSpecificThreats(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            try
            {
                var details = JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.Details ?? "{}") ?? new Dictionary<string, object>();
                
                // Check for unusual login locations
                if (details.ContainsKey("source_ip") && await IsUnusualLocation(details["source_ip"].ToString()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "UnusualLoginLocation",
                        Value = details["source_ip"]?.ToString() ?? "Unknown",
                        Source = "CloudServices",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }

                // Check for bulk API calls
                if (details.ContainsKey("api_calls_count") && int.TryParse(details["api_calls_count"].ToString(), out var apiCalls) && apiCalls > 1000)
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "BulkAPIActivity",
                        Value = $"High API call volume: {apiCalls}",
                        Source = "CloudServices",
                        Confidence = 0.7,
                        Severity = "Medium"
                    });
                }

                // Check for privilege escalation
                if (logEntry.Message?.Contains("AttachUserPolicy") == true || logEntry.Message?.Contains("CreateRole") == true)
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "PrivilegeEscalation",
                        Value = "IAM privilege modification detected",
                        Source = "CloudServices",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in cloud-specific threat analysis");
            }
        }

        private Task AnalyzeDatabaseSpecificThreats(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            try
            {
                var message = logEntry.Message?.ToLowerInvariant() ?? "";
                
                // Check for SQL injection patterns
                var sqlInjectionPatterns = new[]
                {
                    "union select", "or 1=1", "'; drop", "exec xp_", "script>", "javascript:"
                };

                foreach (var pattern in sqlInjectionPatterns)
                {
                    if (message.Contains(pattern))
                    {
                        result.Indicators.Add(new ThreatIndicatorMatch
                        {
                            Type = "SQLInjection",
                            Value = pattern,
                            Source = "Database",
                            Confidence = 0.9,
                            Severity = "Critical"
                        });
                    }
                }

                // Check for bulk data operations
                if (message.Contains("select") && (message.Contains("*") || message.Contains("limit 1000")))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "BulkDataAccess",
                        Value = "Large data retrieval operation",
                        Source = "Database",
                        Confidence = 0.6,
                        Severity = "Medium"
                    });
                }

                // Check for after-hours access
                var currentHour = DateTime.Now.Hour;
                if (currentHour < 6 || currentHour > 22)
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "AfterHoursAccess",
                        Value = $"Database access at {DateTime.Now:HH:mm}",
                        Source = "Database",
                        Confidence = 0.5,
                        Severity = "Low"
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in database-specific threat analysis");
            }
            
            return Task.CompletedTask;
        }

        private async Task AnalyzeIoTSpecificThreats(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            try
            {
                var details = JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.Details ?? "{}") ?? new Dictionary<string, object>();
                
                // Check for device communication with unknown servers
                if (details.ContainsKey("destination_ip") && await IsUnknownServer(details["destination_ip"].ToString()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "UnknownServerCommunication",
                        Value = details["destination_ip"]?.ToString() ?? "Unknown",
                        Source = "IoT",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }

                // Check for sensor anomalies
                if (details.ContainsKey("sensor_value") && double.TryParse(details["sensor_value"].ToString(), out var sensorValue))
                {
                    if (await IsSensorValueAnomalous(details["sensor_type"]?.ToString(), sensorValue))
                    {
                        result.Indicators.Add(new ThreatIndicatorMatch
                        {
                            Type = "SensorAnomaly",
                            Value = $"Unusual sensor reading: {sensorValue}",
                            Source = "IoT",
                            Confidence = 0.7,
                            Severity = "Medium"
                        });
                    }
                }

                // Check for protocol violations
                if (logEntry.Message?.Contains("protocol violation") == true || logEntry.Message?.Contains("invalid packet") == true)
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "ProtocolViolation",
                        Value = "Invalid protocol communication detected",
                        Source = "IoT",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in IoT-specific threat analysis");
            }
        }

        private Task AnalyzeFIMSpecificThreats(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            try
            {
                var message = logEntry.Message?.ToLowerInvariant() ?? "";
                
                // Check for file integrity violations
                if (message.Contains("file modified") || message.Contains("checksum mismatch"))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "FileIntegrityViolation",
                        Value = "File integrity check failed",
                            Source = "FileIntegrity",
                            Confidence = 0.9,
                        Severity = "High"
                    });
                }

                // Check for unauthorized file changes
                if (message.Contains("unauthorized") && message.Contains("file"))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "UnauthorizedFileChange",
                        Value = "Unauthorized file modification detected",
                            Source = "FileIntegrity",
                        Confidence = 0.8,
                        Severity = "Medium"
                        });
                }

                // Check for system file changes
                var details = JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.Details ?? "{}") ?? new Dictionary<string, object>();
                if (details.ContainsKey("filePath"))
                {
                    var filePath = details["filePath"].ToString();
                    if (IsSystemFile(filePath))
                    {
                        result.Indicators.Add(new ThreatIndicatorMatch
                        {
                            Type = "SystemFileChange",
                            Value = $"System file modified: {filePath}",
                            Source = "FileIntegrity",
                            Confidence = 0.95,
                            Severity = "Critical"
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing FIM specific threats for log {LogId}", logEntry.Id);
            }
            
            return Task.CompletedTask;
        }

        private Task CheckThreatIntelligenceFeeds(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            try
            {
                // Extract potential IOCs from log entry
            var message = logEntry.Message ?? "";
            var details = logEntry.Details ?? "";
            
                // Check for IP addresses
                var ipPattern = @"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b";
                var ipMatches = System.Text.RegularExpressions.Regex.Matches(message + " " + details, ipPattern);
                
                foreach (System.Text.RegularExpressions.Match match in ipMatches)
                {
                    var ipAddress = match.Value;
                    // Check against threat intelligence cache
                    lock (_cacheLock)
                    {
                        var key = $"ip:{ipAddress}";
                        if (_threatCache.ContainsKey(key))
                        {
                            var threat = _threatCache[key];
                            result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "MaliciousIP",
                                Value = ipAddress,
                                Source = threat.Source ?? "ThreatIntel",
                                Confidence = 0.9,
                                Severity = threat.Severity
                    });
                }
            }
                }

                // Check for domain names
                var domainPattern = @"\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}\b";
                var domainMatches = System.Text.RegularExpressions.Regex.Matches(message + " " + details, domainPattern);
                
                foreach (System.Text.RegularExpressions.Match match in domainMatches)
                {
                    var domain = match.Value;
                    lock (_cacheLock)
                    {
                        var key = $"domain:{domain}";
                        if (_threatCache.ContainsKey(key))
                        {
                            var threat = _threatCache[key];
                            result.Indicators.Add(new ThreatIndicatorMatch
                            {
                                Type = "MaliciousDomain",
                                Value = domain,
                                Source = threat.Source ?? "ThreatIntel",
                                Confidence = 0.85,
                                Severity = threat.Severity
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking threat intelligence feeds for log {LogId}", logEntry.Id);
            }
            
            return Task.CompletedTask;
        }

        private void CheckSuspiciousPatterns(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            
            var suspiciousPatterns = new Dictionary<string, ThreatLevel>
            {
                { "failed login", ThreatLevel.Low },
                { "brute force", ThreatLevel.High },
                { "privilege escalation", ThreatLevel.Critical },
                { "unauthorized access", ThreatLevel.High },
                { "malware", ThreatLevel.Critical },
                { "ransomware", ThreatLevel.Critical },
                { "data exfiltration", ThreatLevel.Critical },
                { "suspicious activity", ThreatLevel.Medium }
            };

            foreach (var pattern in suspiciousPatterns)
            {
                if (message.Contains(pattern.Key))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "SuspiciousPattern",
                        Value = pattern.Key,
                        Source = "PatternAnalysis",
                        Confidence = 0.7,
                        Severity = pattern.Value.ToString().ToString().ToString()
                    });
                }
            }
        }

        private async Task CheckAnomalousBehavior(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            // Check for time-based anomalies
            var currentHour = logEntry.Timestamp.Hour;
            if (currentHour < 6 || currentHour > 22)
            {
                result.Indicators.Add(new ThreatIndicatorMatch
                {
                    Type = "TimeAnomaly",
                    Value = $"Activity at unusual time: {logEntry.Timestamp:HH:mm}",
                    Source = "BehaviorAnalysis",
                    Confidence = 0.4,
                    Severity = "Low"
                });
            }

            // Check for frequency anomalies (simplified)
            var recentLogs = await _context.LogEntries
                .Where(l => l.AgentId == logEntry.AgentId && 
                           l.Timestamp >= DateTime.UtcNow.AddMinutes(-10))
                .CountAsync();

            if (recentLogs > 100) // High frequency threshold
            {
                result.Indicators.Add(new ThreatIndicatorMatch
                {
                    Type = "FrequencyAnomaly",
                    Value = $"High log frequency: {recentLogs} logs in 10 minutes",
                    Source = "BehaviorAnalysis",
                    Confidence = 0.6,
                    Severity = "Medium"
                });
            }
        }

        private async Task ApplyCorrelationRules(LogEntryModels logEntry, ThreatAnalysisResult result)
        {
            // Look for related events within a time window
            var timeWindow = TimeSpan.FromMinutes(30);
            var relatedLogs = await _context.LogEntries
                .Where(l => l.AgentId == logEntry.AgentId &&
                           l.Timestamp >= logEntry.Timestamp - timeWindow &&
                           l.Timestamp <= logEntry.Timestamp + timeWindow)
                .ToListAsync();

            if (relatedLogs.Count > 1)
            {
                // Check for attack pattern sequences
                var hasFailedLogins = relatedLogs.Any(l => l.Message?.Contains("failed login") == true);
                var hasSuccessfulLogin = relatedLogs.Any(l => l.Message?.Contains("successful login") == true);
                var hasPrivilegeChange = relatedLogs.Any(l => l.Message?.Contains("privilege") == true);

                if (hasFailedLogins && hasSuccessfulLogin)
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "AttackSequence",
                        Value = "Failed logins followed by successful login",
                        Source = "CorrelationAnalysis",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }

                if (hasSuccessfulLogin && hasPrivilegeChange)
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "PrivilegeEscalationSequence",
                        Value = "Login followed by privilege escalation",
                        Source = "CorrelationAnalysis",
                        Confidence = 0.9,
                        Severity = "Critical"
                    });
                }
            }
        }

        private void CalculateFinalThreatScore(ThreatAnalysisResult result)
        {
            if (!result.Indicators.Any())
            {
                result.ThreatLevel = ThreatLevel.None;
                result.ThreatScore = 0;
                return;
            }

            // Convert string severities to numeric values for calculation
            var severityValues = result.Indicators.Select(i => i.Severity switch
            {
                "Critical" => 4,
                "High" => 3,
                "Medium" => 2,
                "Low" => 1,
                _ => 0
            });

            var maxSeverity = severityValues.Max();
            var averageConfidence = result.Indicators.Average(i => i.Confidence);
            var indicatorCount = result.Indicators.Count;

            // Calculate weighted score
            var score = maxSeverity * averageConfidence * Math.Min(indicatorCount / 5.0, 2.0);

            result.ThreatScore = Math.Min(score, 100);
            result.ThreatLevel = score switch
            {
                >= 80 => ThreatLevel.Critical,
                >= 60 => ThreatLevel.High,
                >= 40 => ThreatLevel.Medium,
                >= 20 => ThreatLevel.Low,
                _ => ThreatLevel.None
            };
        }

        // Helper methods for analysis
        private bool IsUnknownImageSource(string? imageName)
        {
            if (string.IsNullOrEmpty(imageName)) return false;
            
            // Check against known safe registries
            var safeRegistries = new[] { "docker.io", "gcr.io", "mcr.microsoft.com", "quay.io" };
            return !safeRegistries.Any(registry => imageName.StartsWith(registry, StringComparison.OrdinalIgnoreCase));
        }

        private Task<bool> IsUnusualLocation(string? ipAddress)
        {
            if (string.IsNullOrEmpty(ipAddress)) return Task.FromResult(false);
            
            // Simple check for private IP ranges (not unusual)
            if (ipAddress.StartsWith("192.168.") || ipAddress.StartsWith("10.") || ipAddress.StartsWith("172."))
                return Task.FromResult(false);
            
            // For demo purposes, consider any external IP as potentially unusual
            return Task.FromResult(true);
        }

        private Task<bool> IsUnknownServer(string? ipAddress)
        {
            if (string.IsNullOrEmpty(ipAddress)) return Task.FromResult(false);
            
            // Check against known IoT server ranges or whitelist
            var knownServers = new[] { "192.168.", "10.", "172." };
            return Task.FromResult(!knownServers.Any(server => ipAddress.StartsWith(server)));
        }

        private Task<bool> IsSensorValueAnomalous(string? sensorType, double value)
        {
            // Simple anomaly detection based on sensor type
            var result = sensorType?.ToLowerInvariant() switch
            {
                "temperature" => value < -50 || value > 100,
                "humidity" => value < 0 || value > 100,
                "pressure" => value < 0 || value > 2000,
                _ => false
            };
            return Task.FromResult(result);
        }

        private bool IsSystemFile(string? filePath)
        {
            if (string.IsNullOrEmpty(filePath)) return false;
            
            var systemPaths = new[]
            {
                "/etc/", "/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/",
                "C:\\Windows\\", "C:\\Program Files\\", "C:\\Program Files (x86)\\"
            };
            
            return systemPaths.Any(path => filePath.StartsWith(path, StringComparison.OrdinalIgnoreCase));
        }

        private string GetSourcePattern(string source)
        {
            // Extract pattern from source for correlation
            var patterns = new[]
            {
                @"(\w+)/\w+", // Type/Instance
                @"(\w+)", // Simple word
            };

            foreach (var pattern in patterns)
            {
                var match = Regex.Match(source, pattern);
                if (match.Success)
                {
                    return match.Groups[1].Value;
                }
            }

            return source;
        }

        private List<string> GenerateCorrelationRecommendations(string pattern, List<LogEntryModels> logs)
        {
            var recommendations = new List<string>();

            if (pattern.Contains("Failed") || pattern.Contains("Error"))
            {
                recommendations.Add("Investigate authentication failures");
                recommendations.Add("Check for brute force attacks");
            }

            if (pattern.Contains("Container") || pattern.Contains("Docker"))
            {
                recommendations.Add("Review container security policies");
                recommendations.Add("Check for container escape attempts");
            }

            if (pattern.Contains("Database") || pattern.Contains("SQL"))
            {
                recommendations.Add("Review database access patterns");
                recommendations.Add("Check for SQL injection attempts");
            }

            return recommendations;
        }

        // Analysis methods for different collectors
        private async Task AnalyzeContainerThreats(LogEntryModels logEntry, ThreatAnalysisResult result, CollectorThreatProfile profile)
        {
            // Container-specific threat analysis
            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            var details = logEntry.Details ?? "";

            // Check for high-risk patterns
            foreach (var pattern in profile.HighRiskPatterns)
            {
                if (message.Contains(pattern.ToLowerInvariant()) || details.Contains(pattern.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "HighRiskPattern",
                        Value = pattern,
                        Source = "Container",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }

            // Check for suspicious activities
            foreach (var activity in profile.SuspiciousActivities)
            {
                if (message.Contains(activity.ToLowerInvariant()) || details.Contains(activity.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "SuspiciousActivity",
                        Value = activity,
                        Source = "Container",
                        Confidence = 0.6,
                        Severity = "Medium"
                    });
                }
            }

            // Perform container-specific deep analysis
            await AnalyzeContainerSpecificThreats(logEntry, result);

            result.CollectorSpecificAnalysis["Container"] = new
            {
                profile_applied = true,
                risk_multiplier = profile.ThreatScoreMultiplier,
                patterns_checked = profile.HighRiskPatterns.Length,
                activities_checked = profile.SuspiciousActivities.Length
            };
        }

        private Task AnalyzeCloudThreats(List<LogEntryModels> logs, CollectorThreatSummary summary)
        {
            // Analyze cloud-specific threats
            foreach (var log in logs)
            {
                var message = log.Message?.ToLowerInvariant() ?? "";
                
                if (message.Contains("unauthorized access") || message.Contains("permission denied"))
                {
                    summary.ThreatsByLevel[ThreatLevel.High]++;
                }
                else if (message.Contains("suspicious activity") || message.Contains("anomaly"))
                {
                    summary.ThreatsByLevel[ThreatLevel.Medium]++;
                }
            }

            summary.TopThreatIndicators.AddRange(new[]
            {
                "Unauthorized cloud access attempts",
                "Suspicious API calls",
                "Unusual data access patterns",
                "Configuration changes"
            });

            summary.RecommendedActions.AddRange(new[]
            {
                "Review cloud access policies",
                "Monitor API usage patterns",
                "Implement cloud security monitoring",
                "Enable detailed audit logging"
            });
            
            return Task.CompletedTask;
        }

        private async Task AnalyzeDatabaseThreats(LogEntryModels logEntry, ThreatAnalysisResult result, CollectorThreatProfile profile)
        {
            // Database-specific threat analysis
            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            var details = logEntry.Details ?? "";

            // Check for high-risk patterns
            foreach (var pattern in profile.HighRiskPatterns)
            {
                if (message.Contains(pattern.ToLowerInvariant()) || details.Contains(pattern.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "HighRiskPattern",
                        Value = pattern,
                        Source = "Database",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }

            // Check for suspicious activities
            foreach (var activity in profile.SuspiciousActivities)
            {
                if (message.Contains(activity.ToLowerInvariant()) || details.Contains(activity.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "SuspiciousActivity",
                        Value = activity,
                        Source = "Database",
                        Confidence = 0.6,
                        Severity = "Medium"
                    });
                }
            }

            // Perform database-specific deep analysis
            await AnalyzeDatabaseSpecificThreats(logEntry, result);

            result.CollectorSpecificAnalysis["Database"] = new
            {
                profile_applied = true,
                risk_multiplier = profile.ThreatScoreMultiplier,
                patterns_checked = profile.HighRiskPatterns.Length,
                activities_checked = profile.SuspiciousActivities.Length
            };
        }

        private async Task AnalyzeIoTThreats(LogEntryModels logEntry, ThreatAnalysisResult result, CollectorThreatProfile profile)
        {
            // IoT-specific threat analysis
            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            var details = logEntry.Details ?? "";

            // Check for high-risk patterns
            foreach (var pattern in profile.HighRiskPatterns)
            {
                if (message.Contains(pattern.ToLowerInvariant()) || details.Contains(pattern.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "HighRiskPattern",
                        Value = pattern,
                        Source = "IoT",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }

            // Check for suspicious activities
            foreach (var activity in profile.SuspiciousActivities)
            {
                if (message.Contains(activity.ToLowerInvariant()) || details.Contains(activity.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "SuspiciousActivity",
                        Value = activity,
                        Source = "IoT",
                        Confidence = 0.6,
                        Severity = "Medium"
                    });
                }
            }

            // Perform IoT-specific deep analysis
            await AnalyzeIoTSpecificThreats(logEntry, result);

            result.CollectorSpecificAnalysis["IoT"] = new
            {
                profile_applied = true,
                risk_multiplier = profile.ThreatScoreMultiplier,
                patterns_checked = profile.HighRiskPatterns.Length,
                activities_checked = profile.SuspiciousActivities.Length
            };
        }

        private Task AnalyzeFIMThreats(List<LogEntryModels> logs, CollectorThreatSummary summary)
        {
            // Analyze file integrity threats
            foreach (var log in logs)
            {
                var message = log.Message?.ToLowerInvariant() ?? "";
                
                if (message.Contains("file modified") || message.Contains("checksum"))
                {
                    summary.ThreatsByLevel[ThreatLevel.Medium]++;
                }
                else if (message.Contains("system file") || message.Contains("critical"))
                {
                    summary.ThreatsByLevel[ThreatLevel.High]++;
                }
            }

            summary.TopThreatIndicators.AddRange(new[]
            {
                "Unauthorized file modifications",
                "System file changes",
                "Checksum mismatches",
                "Configuration file tampering"
            });

            summary.RecommendedActions.AddRange(new[]
            {
                "Implement file integrity monitoring",
                "Use application whitelisting",
                "Monitor system file changes",
                "Regular integrity baseline updates"
            });
            
            return Task.CompletedTask;
        }

        private Task AnalyzeGeneralThreats(List<LogEntryModels> logs, CollectorThreatSummary summary)
        {
            // Count threats by level
            foreach (var log in logs)
            {
                var threatLevel = DetermineThreatLevelFromLog(log);
                if (summary.ThreatsByLevel.ContainsKey(threatLevel))
                    summary.ThreatsByLevel[threatLevel]++;
                else
                    summary.ThreatsByLevel[threatLevel] = 1;
            }

            // Add general recommendations
            summary.RecommendedActions.Add("Review error patterns");
            summary.RecommendedActions.Add("Monitor for anomalous behavior");
            summary.RecommendedActions.Add("Update threat intelligence feeds");
            
            return Task.CompletedTask;
        }

        private ThreatLevel DetermineThreatLevelFromLog(LogEntryModels log)
        {
            // Simple threat level determination based on log content
            if (log.Level == "Error" || log.Level == "Critical")
                return ThreatLevel.High;
            if (log.Level == "Warning")
                return ThreatLevel.Medium;
            if (log.Message.Contains("failed", StringComparison.OrdinalIgnoreCase) ||
                log.Message.Contains("error", StringComparison.OrdinalIgnoreCase))
                return ThreatLevel.Medium;
            return ThreatLevel.Low;
        }

        // Missing interface method implementations
        public async Task UpdateFeedAsync(string feedId)
        {
            try
            {
                _logger.LogInformation("Updating threat intelligence feed: {FeedId}", feedId);
                // TODO: Implement threat feed update logic
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating threat feed {FeedId}", feedId);
                throw;
            }
        }

        public Task<object> EnrichIndicatorAsync(ThreatIndicator indicator)
        {
            try
            {
                _logger.LogInformation("Enriching threat indicator: {Type} - {Value}", indicator.Type, indicator.Value);
                
                var enrichmentData = new Dictionary<string, object>
                {
                    ["originalIndicator"] = indicator,
                    ["enrichedAt"] = DateTime.UtcNow,
                    ["source"] = "AthalaSIEM",
                    ["additionalContext"] = new Dictionary<string, object>()
                };

                // TODO: Add actual enrichment logic (external API calls, database lookups, etc.)
                
                return Task.FromResult<object>(enrichmentData);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error enriching indicator {Type} - {Value}", indicator.Type, indicator.Value);
                throw;
            }
        }

        public Task<IEnumerable<ThreatMatchDto>> SearchThreatsAsync(ThreatSearchRequest request)
        {
            try
            {
                _logger.LogInformation("Searching threats for: {SearchValue}", request.SearchValue);
                
                var matches = new List<ThreatMatchDto>();
                
                // Search in cached threats
                lock (_cacheLock)
                {
                    foreach (var threat in _threatCache.Values)
                    {
                        if (threat.Value.Contains(request.SearchValue, StringComparison.OrdinalIgnoreCase) ||
                            threat.Type.Contains(request.SearchValue, StringComparison.OrdinalIgnoreCase))
                        {
                            var match = new ThreatMatchDto
                            {
                                Id = Guid.NewGuid().ToString(),
                                IndicatorId = threat.Id,
                                LogEntryId = threat.LogEntryId ?? string.Empty,
                                MatchedValue = threat.Value,
                                MatchedField = "value",
                                Confidence = threat.Confidence.ToString(),
                                Severity = threat.Severity,
                                DetectedAt = DateTime.UtcNow,
                                Type = threat.Type,
                                Value = threat.Value,
                                Source = threat.Source ?? "Unknown"
                            };
                            matches.Add(match);
                        }
                    }
                }

                return Task.FromResult<IEnumerable<ThreatMatchDto>>(matches);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error searching threats for {SearchValue}", request.SearchValue);
                throw;
            }
        }

        public Task<bool> CheckIndicatorAsync(string value, string type)
        {
            try
            {
                _logger.LogDebug("Checking indicator: {Type} - {Value}", type, value);
                
                // Check in cache first
                lock (_cacheLock)
                {
                    var key = $"{type}:{value}";
                    if (_threatCache.ContainsKey(key))
                    {
                        return Task.FromResult(true);
                    }
                }

                // TODO: Check against external threat intelligence feeds
                
                return Task.FromResult(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking indicator {Type} - {Value}", type, value);
                return Task.FromResult(false);
            }
        }

        public async Task ProcessLogEntryAsync(LogEntryModels logEntry)
        {
            try
            {
                _logger.LogDebug("Processing log entry for threats: {LogId}", logEntry.Id);
                
                var analysisResult = await AnalyzeLogEntryAsync(logEntry);
                
                // Store analysis results if threats found
                if (analysisResult.ThreatLevel > ThreatLevel.None)
                {
                    _logger.LogWarning("Threat detected in log {LogId}: {ThreatLevel}", 
                        logEntry.Id, analysisResult.ThreatLevel);
                    
                    // TODO: Store threat analysis results, trigger alerts, etc.
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing log entry {LogId}", logEntry.Id);
                throw;
            }
        }

        private Task AnalyzeGeneralThreats(LogEntryModels logEntry, ThreatAnalysisResult result, CollectorThreatProfile profile)
        {
            // General threat analysis
            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            var details = logEntry.Details ?? "";

            // Check for high-risk patterns
            foreach (var pattern in profile.HighRiskPatterns)
            {
                if (message.Contains(pattern.ToLowerInvariant()) || details.Contains(pattern.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "HighRiskPattern",
                        Value = pattern,
                        Source = "General",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }

            // Check for suspicious activities
            foreach (var activity in profile.SuspiciousActivities)
            {
                if (message.Contains(activity.ToLowerInvariant()) || details.Contains(activity.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "SuspiciousActivity",
                        Value = activity,
                        Source = "General",
                        Confidence = 0.6,
                        Severity = "Medium"
            });
        }
    }

            result.CollectorSpecificAnalysis["General"] = new
            {
                profile_applied = true,
                risk_multiplier = profile.ThreatScoreMultiplier,
                patterns_checked = profile.HighRiskPatterns.Length,
                activities_checked = profile.SuspiciousActivities.Length
            };
            
            return Task.CompletedTask;
        }

        private async Task AnalyzeCloudServiceThreats(LogEntryModels logEntry, ThreatAnalysisResult result, CollectorThreatProfile profile)
        {
            // Cloud service-specific threat analysis
            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            var details = logEntry.Details ?? "";

            // Check for high-risk patterns
            foreach (var pattern in profile.HighRiskPatterns)
            {
                if (message.Contains(pattern.ToLowerInvariant()) || details.Contains(pattern.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "HighRiskPattern",
                        Value = pattern,
                        Source = "CloudServices",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }

            // Check for suspicious activities
            foreach (var activity in profile.SuspiciousActivities)
            {
                if (message.Contains(activity.ToLowerInvariant()) || details.Contains(activity.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "SuspiciousActivity",
                        Value = activity,
                        Source = "CloudServices",
                        Confidence = 0.6,
                        Severity = "Medium"
                    });
                }
            }

            // Perform cloud-specific deep analysis
            await AnalyzeCloudSpecificThreats(logEntry, result);

            result.CollectorSpecificAnalysis["CloudServices"] = new
            {
                profile_applied = true,
                risk_multiplier = profile.ThreatScoreMultiplier,
                patterns_checked = profile.HighRiskPatterns.Length,
                activities_checked = profile.SuspiciousActivities.Length
            };
        }

        private async Task AnalyzeFileIntegrityThreats(LogEntryModels logEntry, ThreatAnalysisResult result, CollectorThreatProfile profile)
        {
            // File integrity-specific threat analysis
            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            var details = logEntry.Details ?? "";

            // Check for high-risk patterns
            foreach (var pattern in profile.HighRiskPatterns)
            {
                if (message.Contains(pattern.ToLowerInvariant()) || details.Contains(pattern.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "HighRiskPattern",
                        Value = pattern,
                        Source = "FileIntegrity",
                        Confidence = 0.8,
                        Severity = "High"
                    });
                }
            }

            // Check for suspicious activities
            foreach (var activity in profile.SuspiciousActivities)
            {
                if (message.Contains(activity.ToLowerInvariant()) || details.Contains(activity.ToLowerInvariant()))
                {
                    result.Indicators.Add(new ThreatIndicatorMatch
                    {
                        Type = "SuspiciousActivity",
                        Value = activity,
                        Source = "FileIntegrity",
                        Confidence = 0.6,
                        Severity = "Medium"
                    });
                }
            }

            // Perform file integrity-specific deep analysis
            await AnalyzeFIMSpecificThreats(logEntry, result);

            result.CollectorSpecificAnalysis["FileIntegrity"] = new
            {
                profile_applied = true,
                risk_multiplier = profile.ThreatScoreMultiplier,
                patterns_checked = profile.HighRiskPatterns.Length,
                activities_checked = profile.SuspiciousActivities.Length
            };
        }

        private ThreatLevel DetermineThreatLevel(ThreatAnalysisResult result)
        {
            if (result.ThreatScore >= 0.8) return ThreatLevel.Critical;
            if (result.ThreatScore >= 0.6) return ThreatLevel.High;
            if (result.ThreatScore >= 0.4) return ThreatLevel.Medium;
            if (result.ThreatScore >= 0.2) return ThreatLevel.Low;
            return ThreatLevel.None;
        }


        private Task AnalyzeContainerThreatsForSummary(List<LogEntryModels> threats, CollectorThreatSummary summary)
        {
            summary.TopThreatIndicators.AddRange(new[]
            {
                "Privileged container execution",
                "Unknown image sources",
                "Container escape attempts",
                "Excessive resource usage"
            });

            summary.RecommendedActions.AddRange(new[]
            {
                "Implement container security policies",
                "Use trusted image registries only",
                "Monitor container runtime behavior",
                "Implement resource limits"
            });
            
            return Task.CompletedTask;
        }

        private Task AnalyzeDatabaseThreatsForSummary(List<LogEntryModels> threats, CollectorThreatSummary summary)
        {
            summary.TopThreatIndicators.AddRange(new[]
            {
                "SQL injection attempts",
                "Bulk data access",
                "After-hours database access",
                "Privilege escalation"
            });

            summary.RecommendedActions.AddRange(new[]
            {
                "Implement parameterized queries",
                "Enable database audit logging",
                "Restrict after-hours access",
                "Monitor bulk data operations"
            });
            
            return Task.CompletedTask;
        }

        private Task AnalyzeIoTThreatsForSummary(List<LogEntryModels> threats, CollectorThreatSummary summary)
        {
            summary.TopThreatIndicators.AddRange(new[]
            {
                "Unknown server communication",
                "Sensor anomalies",
                "Protocol violations",
                "Device hijacking attempts"
            });

            summary.RecommendedActions.AddRange(new[]
            {
                "Implement network segmentation",
                "Monitor device communications",
                "Update device firmware regularly",
                "Use encrypted communications"
            });
            
            return Task.CompletedTask;
        }
    }
}

