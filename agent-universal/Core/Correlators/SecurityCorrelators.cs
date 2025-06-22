using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.Agent.Core.Correlators
{
    /// <summary>
    /// Enterprise authentication correlator that detects advanced authentication-based attacks.
    /// Identifies brute force attacks, credential stuffing, pass-the-hash, and anomalous authentication patterns.
    /// </summary>
    public class EnterpriseAuthenticationCorrelator : ILogCorrelator
    {
        private readonly ILogger<EnterpriseAuthenticationCorrelator> _logger;
        private int _bruteForceThreshold = 5;
        private int _timeWindowMinutes = 15;
        private int _credentialStuffingThreshold = 10;
        private int _successAfterFailuresThreshold = 3;

        /// <inheritdoc />
        public string Name => "Enterprise Authentication Correlator";

        /// <inheritdoc />
        public string Description => "Detects authentication-based attacks including brute force, credential stuffing, and anomalous patterns";

        /// <inheritdoc />
        public List<string> DetectedTechniques => new()
        {
            "T1110.001", // Brute Force: Password Guessing
            "T1110.003", // Brute Force: Password Spraying
            "T1110.004", // Brute Force: Credential Stuffing
            "T1078",     // Valid Accounts
            "T1021"      // Remote Services
        };

        /// <inheritdoc />
        public double MinimumConfidence => 0.7;

        /// <summary>
        /// Initializes a new instance of the EnterpriseAuthenticationCorrelator.
        /// </summary>
        /// <param name="logger">Logger instance for this correlator.</param>
        public EnterpriseAuthenticationCorrelator(ILogger<EnterpriseAuthenticationCorrelator> logger)
        {
            _logger = logger;
        }

        /// <inheritdoc />
        public async Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            if (config.TryGetValue("BruteForceThreshold", out var threshold) && threshold is int bfThreshold)
            {
                _bruteForceThreshold = bfThreshold;
            }

            if (config.TryGetValue("TimeWindowMinutes", out var window) && window is int timeWindow)
            {
                _timeWindowMinutes = timeWindow;
            }

            if (config.TryGetValue("CredentialStuffingThreshold", out var csThreshold) && csThreshold is int credStuffThreshold)
            {
                _credentialStuffingThreshold = credStuffThreshold;
            }

            if (config.TryGetValue("SuccessAfterFailuresThreshold", out var safThreshold) && safThreshold is int successThreshold)
            {
                _successAfterFailuresThreshold = successThreshold;
            }

            _logger.LogInformation("Authentication correlator initialized with thresholds - BruteForce: {BF}, CredentialStuffing: {CS}, TimeWindow: {TW}min",
                _bruteForceThreshold, _credentialStuffingThreshold, _timeWindowMinutes);

            return await Task.FromResult(true);
        }

        /// <inheritdoc />
        public IEnumerable<LogCorrelation> DetectCorrelations(IEnumerable<LogEntry> logs)
        {
            var correlations = new List<LogCorrelation>();
            var authLogs = logs.Where(IsAuthenticationEvent).OrderBy(l => l.Timestamp).ToList();

            if (!authLogs.Any())
            {
                return correlations;
            }

            // Detect different attack patterns
            correlations.AddRange(DetectBruteForceAttacks(authLogs));
            correlations.AddRange(DetectCredentialStuffingAttacks(authLogs));
            correlations.AddRange(DetectSuccessAfterFailures(authLogs));
            correlations.AddRange(DetectAnomalousAuthenticationPatterns(authLogs));

            return correlations;
        }

        /// <summary>
        /// Determines if a log entry is an authentication event.
        /// </summary>
        /// <param name="log">The log entry to check.</param>
        /// <returns>True if it's an authentication event.</returns>
        private bool IsAuthenticationEvent(LogEntry log)
        {
            var authEventIds = new HashSet<string> { "4624", "4625", "4648", "4776", "4777" };
            return !string.IsNullOrEmpty(log.EventId) && authEventIds.Contains(log.EventId);
        }

        /// <summary>
        /// Detects brute force attacks against specific accounts.
        /// </summary>
        /// <param name="authLogs">Authentication logs to analyze.</param>
        /// <returns>Collection of brute force correlations.</returns>
        private IEnumerable<LogCorrelation> DetectBruteForceAttacks(List<LogEntry> authLogs)
        {
            var correlations = new List<LogCorrelation>();
            var timeWindow = TimeSpan.FromMinutes(_timeWindowMinutes);

            // Group by target username and source IP
            var groupedByUser = authLogs
                .Where(l => l.EventId == "4625" && !string.IsNullOrEmpty(l.Username))
                .GroupBy(l => new { l.Username, l.IpAddress });

            foreach (var userGroup in groupedByUser)
            {
                var userLogs = userGroup.OrderBy(l => l.Timestamp).ToList();
                
                for (int i = 0; i < userLogs.Count; i++)
                {
                    var windowStart = userLogs[i].Timestamp;
                    var windowEnd = windowStart.Add(timeWindow);
                    
                    var logsInWindow = userLogs
                        .Skip(i)
                        .TakeWhile(l => l.Timestamp <= windowEnd)
                        .ToList();

                    if (logsInWindow.Count >= _bruteForceThreshold)
                    {
                        var confidence = CalculateBruteForceConfidence(logsInWindow.Count, _bruteForceThreshold);
                        
                        correlations.Add(new LogCorrelation
                        {
                            Name = "Brute Force Attack",
                            Description = $"Multiple failed login attempts for user '{userGroup.Key.Username}' from IP '{userGroup.Key.IpAddress}'",
                            Severity = DetermineSeverity(logsInWindow.Count, _bruteForceThreshold),
                            ConfidenceScore = confidence,
                            MitreTechniques = new List<string> { "T1110.001" },
                            RelatedLogs = logsInWindow,
                            Properties = new Dictionary<string, object>
                            {
                                ["TargetUsername"] = userGroup.Key.Username ?? "Unknown",
                                ["SourceIP"] = userGroup.Key.IpAddress ?? "Unknown",
                                ["FailedAttempts"] = logsInWindow.Count,
                                ["TimeWindowMinutes"] = _timeWindowMinutes,
                                ["AttackType"] = "BruteForce"
                            }
                        });

                        // Skip processed logs
                        i += logsInWindow.Count - 1;
                    }
                }
            }

            return correlations;
        }

        /// <summary>
        /// Detects credential stuffing attacks (same source, multiple accounts).
        /// </summary>
        /// <param name="authLogs">Authentication logs to analyze.</param>
        /// <returns>Collection of credential stuffing correlations.</returns>
        private IEnumerable<LogCorrelation> DetectCredentialStuffingAttacks(List<LogEntry> authLogs)
        {
            var correlations = new List<LogCorrelation>();
            var timeWindow = TimeSpan.FromMinutes(_timeWindowMinutes);

            // Group by source IP
            var groupedByIp = authLogs
                .Where(l => l.EventId == "4625" && !string.IsNullOrEmpty(l.IpAddress))
                .GroupBy(l => l.IpAddress);

            foreach (var ipGroup in groupedByIp)
            {
                var ipLogs = ipGroup.OrderBy(l => l.Timestamp).ToList();
                var uniqueUsers = ipLogs.Select(l => l.Username).Distinct().Count();

                if (ipLogs.Count >= _credentialStuffingThreshold && uniqueUsers >= 3)
                {
                    var confidence = CalculateCredentialStuffingConfidence(ipLogs.Count, uniqueUsers);
                    
                    correlations.Add(new LogCorrelation
                    {
                        Name = "Credential Stuffing Attack",
                        Description = $"Multiple failed login attempts for different users from IP '{ipGroup.Key}'",
                        Severity = DetermineSeverity(ipLogs.Count, _credentialStuffingThreshold),
                        ConfidenceScore = confidence,
                        MitreTechniques = new List<string> { "T1110.004" },
                        RelatedLogs = ipLogs,
                        Properties = new Dictionary<string, object>
                        {
                            ["SourceIP"] = ipGroup.Key ?? "Unknown",
                            ["FailedAttempts"] = ipLogs.Count,
                            ["UniqueUsers"] = uniqueUsers,
                            ["AttackType"] = "CredentialStuffing"
                        }
                    });
                }
            }

            return correlations;
        }

        /// <summary>
        /// Detects successful logins after multiple failures (potential successful brute force).
        /// </summary>
        /// <param name="authLogs">Authentication logs to analyze.</param>
        /// <returns>Collection of success after failures correlations.</returns>
        private IEnumerable<LogCorrelation> DetectSuccessAfterFailures(List<LogEntry> authLogs)
        {
            var correlations = new List<LogCorrelation>();
            var timeWindow = TimeSpan.FromMinutes(_timeWindowMinutes);

            // Group by target username and source IP
            var groupedByUser = authLogs
                .Where(l => !string.IsNullOrEmpty(l.Username))
                .GroupBy(l => new { l.Username, l.IpAddress });

            foreach (var userGroup in groupedByUser)
            {
                var userLogs = userGroup.OrderBy(l => l.Timestamp).ToList();
                
                for (int i = 0; i < userLogs.Count; i++)
                {
                    if (userLogs[i].EventId == "4624") // Successful login
                    {
                        var successTime = userLogs[i].Timestamp;
                        var windowStart = successTime.Subtract(timeWindow);
                        
                        var precedingFailures = userLogs
                            .Take(i)
                            .Where(l => l.EventId == "4625" && l.Timestamp >= windowStart)
                            .ToList();

                        if (precedingFailures.Count >= _successAfterFailuresThreshold)
                        {
                            var allLogs = precedingFailures.Concat(new[] { userLogs[i] }).ToList();
                            var confidence = CalculateSuccessAfterFailuresConfidence(precedingFailures.Count);
                            
                            correlations.Add(new LogCorrelation
                            {
                                Name = "Successful Login After Failures",
                                Description = $"Successful login for '{userGroup.Key.Username}' after {precedingFailures.Count} failed attempts",
                                Severity = "High",
                                ConfidenceScore = confidence,
                                MitreTechniques = new List<string> { "T1110.001", "T1078" },
                                RelatedLogs = allLogs,
                                Properties = new Dictionary<string, object>
                                {
                                    ["TargetUsername"] = userGroup.Key.Username ?? "Unknown",
                                    ["SourceIP"] = userGroup.Key.IpAddress ?? "Unknown",
                                    ["PrecedingFailures"] = precedingFailures.Count,
                                    ["AttackType"] = "SuccessfulBruteForce"
                                }
                            });
                        }
                    }
                }
            }

            return correlations;
        }

        /// <summary>
        /// Detects anomalous authentication patterns.
        /// </summary>
        /// <param name="authLogs">Authentication logs to analyze.</param>
        /// <returns>Collection of anomalous pattern correlations.</returns>
        private IEnumerable<LogCorrelation> DetectAnomalousAuthenticationPatterns(List<LogEntry> authLogs)
        {
            var correlations = new List<LogCorrelation>();

            // Detect logins from multiple IPs for same user in short time
            var userGroups = authLogs
                .Where(l => l.EventId == "4624" && !string.IsNullOrEmpty(l.Username))
                .GroupBy(l => l.Username);

            foreach (var userGroup in userGroups)
            {
                var userLogs = userGroup.OrderBy(l => l.Timestamp).ToList();
                var timeWindow = TimeSpan.FromHours(1); // 1 hour window

                for (int i = 0; i < userLogs.Count - 1; i++)
                {
                    var windowStart = userLogs[i].Timestamp;
                    var windowEnd = windowStart.Add(timeWindow);
                    
                    var logsInWindow = userLogs
                        .Skip(i)
                        .TakeWhile(l => l.Timestamp <= windowEnd)
                        .ToList();

                    var uniqueIps = logsInWindow.Select(l => l.IpAddress).Distinct().Count();
                    
                    if (uniqueIps >= 3) // 3 or more different IPs
                    {
                        correlations.Add(new LogCorrelation
                        {
                            Name = "Anomalous Authentication Pattern",
                            Description = $"User '{userGroup.Key}' logged in from {uniqueIps} different IPs within 1 hour",
                            Severity = "Medium",
                            ConfidenceScore = 0.6,
                            MitreTechniques = new List<string> { "T1078" },
                            RelatedLogs = logsInWindow,
                            Properties = new Dictionary<string, object>
                            {
                                ["Username"] = userGroup.Key ?? "Unknown",
                                ["UniqueIPs"] = uniqueIps,
                                ["AttackType"] = "AnomalousPattern"
                            }
                        });

                        // Skip processed logs
                        i += logsInWindow.Count - 1;
                    }
                }
            }

            return correlations;
        }

        /// <summary>
        /// Calculates confidence score for brute force attacks.
        /// </summary>
        /// <param name="attemptCount">Number of failed attempts.</param>
        /// <param name="threshold">Configured threshold.</param>
        /// <returns>Confidence score between 0.0 and 1.0.</returns>
        private double CalculateBruteForceConfidence(int attemptCount, int threshold)
        {
            if (attemptCount < threshold) return 0.0;
            
            // Higher confidence for more attempts
            var baseConfidence = 0.7;
            var additionalConfidence = Math.Min(0.3, (attemptCount - threshold) * 0.05);
            
            return Math.Min(1.0, baseConfidence + additionalConfidence);
        }

        /// <summary>
        /// Calculates confidence score for credential stuffing attacks.
        /// </summary>
        /// <param name="attemptCount">Number of failed attempts.</param>
        /// <param name="uniqueUsers">Number of unique users targeted.</param>
        /// <returns>Confidence score between 0.0 and 1.0.</returns>
        private double CalculateCredentialStuffingConfidence(int attemptCount, int uniqueUsers)
        {
            var baseConfidence = 0.6;
            var userDiversityBonus = Math.Min(0.3, uniqueUsers * 0.05);
            var volumeBonus = Math.Min(0.1, attemptCount * 0.01);
            
            return Math.Min(1.0, baseConfidence + userDiversityBonus + volumeBonus);
        }

        /// <summary>
        /// Calculates confidence score for successful logins after failures.
        /// </summary>
        /// <param name="failureCount">Number of preceding failures.</param>
        /// <returns>Confidence score between 0.0 and 1.0.</returns>
        private double CalculateSuccessAfterFailuresConfidence(int failureCount)
        {
            var baseConfidence = 0.8;
            var volumeBonus = Math.Min(0.2, failureCount * 0.02);
            
            return Math.Min(1.0, baseConfidence + volumeBonus);
        }

        /// <summary>
        /// Determines severity based on attack volume.
        /// </summary>
        /// <param name="count">Number of events.</param>
        /// <param name="threshold">Base threshold.</param>
        /// <returns>Severity level string.</returns>
        private string DetermineSeverity(int count, int threshold)
        {
            if (count >= threshold * 3) return "Critical";
            if (count >= threshold * 2) return "High";
            return "Medium";
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["BruteForceThreshold"] = _bruteForceThreshold,
                ["CredentialStuffingThreshold"] = _credentialStuffingThreshold,
                ["TimeWindowMinutes"] = _timeWindowMinutes,
                ["SupportedEventIds"] = new[] { "4624", "4625", "4648", "4776", "4777" }
            };
        }
    }

    /// <summary>
    /// Enterprise privilege escalation correlator that detects unauthorized privilege elevation.
    /// Identifies suspicious privilege use, token manipulation, and elevation of privilege attacks.
    /// </summary>
    public class EnterprisePrivilegeEscalationCorrelator : ILogCorrelator
    {
        private readonly ILogger<EnterprisePrivilegeEscalationCorrelator> _logger;
        private int _privilegeUseThreshold = 3;
        private int _timeWindowMinutes = 30;

        /// <inheritdoc />
        public string Name => "Enterprise Privilege Escalation Correlator";

        /// <inheritdoc />
        public string Description => "Detects privilege escalation attacks including token manipulation and unauthorized elevation";

        /// <inheritdoc />
        public List<string> DetectedTechniques => new()
        {
            "T1548.002", // Abuse Elevation Control Mechanism: Bypass User Account Control
            "T1134",     // Access Token Manipulation
            "T1055",     // Process Injection
            "T1543.003"  // Create or Modify System Process: Windows Service
        };

        /// <inheritdoc />
        public double MinimumConfidence => 0.6;

        /// <summary>
        /// Initializes a new instance of the EnterprisePrivilegeEscalationCorrelator.
        /// </summary>
        /// <param name="logger">Logger instance for this correlator.</param>
        public EnterprisePrivilegeEscalationCorrelator(ILogger<EnterprisePrivilegeEscalationCorrelator> logger)
        {
            _logger = logger;
        }

        /// <inheritdoc />
        public async Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            if (config.TryGetValue("PrivilegeUseThreshold", out var threshold) && threshold is int privThreshold)
            {
                _privilegeUseThreshold = privThreshold;
            }

            if (config.TryGetValue("TimeWindowMinutes", out var window) && window is int timeWindow)
            {
                _timeWindowMinutes = timeWindow;
            }

            _logger.LogInformation("Privilege escalation correlator initialized with threshold: {Threshold}, window: {Window}min",
                _privilegeUseThreshold, _timeWindowMinutes);

            return await Task.FromResult(true);
        }

        /// <inheritdoc />
        public IEnumerable<LogCorrelation> DetectCorrelations(IEnumerable<LogEntry> logs)
        {
            var correlations = new List<LogCorrelation>();
            var privilegeLogs = logs.Where(IsPrivilegeEvent).OrderBy(l => l.Timestamp).ToList();

            if (!privilegeLogs.Any())
            {
                return correlations;
            }

            // Detect different privilege escalation patterns
            correlations.AddRange(DetectExcessivePrivilegeUse(privilegeLogs));
            correlations.AddRange(DetectSuspiciousPrivilegeChains(privilegeLogs));
            correlations.AddRange(DetectUnauthorizedSystemPrivileges(privilegeLogs));

            return correlations;
        }

        /// <summary>
        /// Determines if a log entry is a privilege-related event.
        /// </summary>
        /// <param name="log">The log entry to check.</param>
        /// <returns>True if it's a privilege event.</returns>
        private bool IsPrivilegeEvent(LogEntry log)
        {
            var privilegeEventIds = new HashSet<string> { "4672", "4673", "4674", "4697", "4698", "4699", "4700", "4701", "4702" };
            return !string.IsNullOrEmpty(log.EventId) && privilegeEventIds.Contains(log.EventId);
        }

        /// <summary>
        /// Detects excessive privilege use within a time window.
        /// </summary>
        /// <param name="privilegeLogs">Privilege-related logs to analyze.</param>
        /// <returns>Collection of excessive privilege use correlations.</returns>
        private IEnumerable<LogCorrelation> DetectExcessivePrivilegeUse(List<LogEntry> privilegeLogs)
        {
            var correlations = new List<LogCorrelation>();
            var timeWindow = TimeSpan.FromMinutes(_timeWindowMinutes);

            // Group by user and computer
            var groupedByUser = privilegeLogs
                .Where(l => !string.IsNullOrEmpty(l.Username))
                .GroupBy(l => new { l.Username, l.ComputerName });

            foreach (var userGroup in groupedByUser)
            {
                var userLogs = userGroup.OrderBy(l => l.Timestamp).ToList();
                
                for (int i = 0; i < userLogs.Count; i++)
                {
                    var windowStart = userLogs[i].Timestamp;
                    var windowEnd = windowStart.Add(timeWindow);
                    
                    var logsInWindow = userLogs
                        .Skip(i)
                        .TakeWhile(l => l.Timestamp <= windowEnd)
                        .ToList();

                    if (logsInWindow.Count >= _privilegeUseThreshold)
                    {
                        var confidence = CalculatePrivilegeEscalationConfidence(logsInWindow);
                        
                        correlations.Add(new LogCorrelation
                        {
                            Name = "Excessive Privilege Use",
                            Description = $"Excessive privilege use by '{userGroup.Key.Username}' on '{userGroup.Key.ComputerName}'",
                            Severity = DetermineSeverityByCount(logsInWindow.Count, _privilegeUseThreshold),
                            ConfidenceScore = confidence,
                            MitreTechniques = new List<string> { "T1134" },
                            RelatedLogs = logsInWindow,
                            Properties = new Dictionary<string, object>
                            {
                                ["Username"] = userGroup.Key.Username ?? "Unknown",
                                ["ComputerName"] = userGroup.Key.ComputerName ?? "Unknown",
                                ["PrivilegeUseCount"] = logsInWindow.Count,
                                ["TimeWindowMinutes"] = _timeWindowMinutes,
                                ["AttackType"] = "ExcessivePrivilegeUse"
                            }
                        });

                        // Skip processed logs
                        i += logsInWindow.Count - 1;
                    }
                }
            }

            return correlations;
        }

        /// <summary>
        /// Detects suspicious privilege escalation chains.
        /// </summary>
        /// <param name="privilegeLogs">Privilege-related logs to analyze.</param>
        /// <returns>Collection of privilege chain correlations.</returns>
        private IEnumerable<LogCorrelation> DetectSuspiciousPrivilegeChains(List<LogEntry> privilegeLogs)
        {
            var correlations = new List<LogCorrelation>();
            
            // Look for specific privilege escalation patterns
            var chainPatterns = new[]
            {
                new[] { "4672", "4697" }, // Special privileges + service installation
                new[] { "4673", "4698" }, // Privileged service called + service started
                new[] { "4672", "4688" }  // Special privileges + process creation
            };

            foreach (var pattern in chainPatterns)
            {
                var patternCorrelations = DetectPrivilegeChainPattern(privilegeLogs, pattern);
                correlations.AddRange(patternCorrelations);
            }

            return correlations;
        }

        /// <summary>
        /// Detects specific privilege escalation chain patterns.
        /// </summary>
        /// <param name="privilegeLogs">Privilege-related logs to analyze.</param>
        /// <param name="pattern">Event ID pattern to detect.</param>
        /// <returns>Collection of pattern correlations.</returns>
        private IEnumerable<LogCorrelation> DetectPrivilegeChainPattern(List<LogEntry> privilegeLogs, string[] pattern)
        {
            var correlations = new List<LogCorrelation>();
            var timeWindow = TimeSpan.FromMinutes(5); // Short window for chained events

            for (int i = 0; i < privilegeLogs.Count - pattern.Length + 1; i++)
            {
                var startLog = privilegeLogs[i];
                if (startLog.EventId != pattern[0]) continue;

                var chainLogs = new List<LogEntry> { startLog };
                var currentTime = startLog.Timestamp;
                var currentIndex = i + 1;
                var patternIndex = 1;

                while (patternIndex < pattern.Length && currentIndex < privilegeLogs.Count)
                {
                    var candidateLog = privilegeLogs[currentIndex];
                    
                    if (candidateLog.Timestamp - currentTime > timeWindow)
                    {
                        break; // Time window exceeded
                    }

                    if (candidateLog.EventId == pattern[patternIndex] &&
                        candidateLog.Username == startLog.Username &&
                        candidateLog.ComputerName == startLog.ComputerName)
                    {
                        chainLogs.Add(candidateLog);
                        currentTime = candidateLog.Timestamp;
                        patternIndex++;
                    }

                    currentIndex++;
                }

                if (patternIndex == pattern.Length) // Complete pattern found
                {
                    correlations.Add(new LogCorrelation
                    {
                        Name = "Privilege Escalation Chain",
                        Description = $"Detected privilege escalation chain pattern for '{startLog.Username}'",
                        Severity = "High",
                        ConfidenceScore = 0.8,
                        MitreTechniques = new List<string> { "T1548.002", "T1134" },
                        RelatedLogs = chainLogs,
                        Properties = new Dictionary<string, object>
                        {
                            ["Username"] = startLog.Username ?? "Unknown",
                            ["ComputerName"] = startLog.ComputerName ?? "Unknown",
                            ["Pattern"] = string.Join("->", pattern),
                            ["ChainDuration"] = (chainLogs.Last().Timestamp - chainLogs.First().Timestamp).TotalMinutes,
                            ["AttackType"] = "PrivilegeChain"
                        }
                    });
                }
            }

            return correlations;
        }

        /// <summary>
        /// Detects unauthorized system privilege use.
        /// </summary>
        /// <param name="privilegeLogs">Privilege-related logs to analyze.</param>
        /// <returns>Collection of unauthorized privilege correlations.</returns>
        private IEnumerable<LogCorrelation> DetectUnauthorizedSystemPrivileges(List<LogEntry> privilegeLogs)
        {
            var correlations = new List<LogCorrelation>();
            
            // Focus on critical system privileges
            var systemPrivilegeLogs = privilegeLogs
                .Where(l => l.EventId == "4672") // Special privileges assigned
                .Where(l => ContainsCriticalPrivileges(l))
                .ToList();

            // Group by user to identify unusual privilege assignments
            var userGroups = systemPrivilegeLogs
                .GroupBy(l => l.Username)
                .Where(g => !string.IsNullOrEmpty(g.Key));

            foreach (var userGroup in userGroups)
            {
                var userLogs = userGroup.ToList();
                
                // Check if user is getting system-level privileges
                var criticalPrivilegeCount = userLogs.Count;
                
                if (criticalPrivilegeCount >= 2) // Threshold for investigation
                {
                    correlations.Add(new LogCorrelation
                    {
                        Name = "Unauthorized System Privileges",
                        Description = $"User '{userGroup.Key}' assigned critical system privileges",
                        Severity = "Critical",
                        ConfidenceScore = 0.7,
                        MitreTechniques = new List<string> { "T1134", "T1543.003" },
                        RelatedLogs = userLogs,
                        Properties = new Dictionary<string, object>
                        {
                            ["Username"] = userGroup.Key ?? "Unknown",
                            ["CriticalPrivilegeAssignments"] = criticalPrivilegeCount,
                            ["AttackType"] = "UnauthorizedSystemPrivileges"
                        }
                    });
                }
            }

            return correlations;
        }

        /// <summary>
        /// Checks if a log contains critical system privileges.
        /// </summary>
        /// <param name="log">The log to check.</param>
        /// <returns>True if critical privileges are present.</returns>
        private bool ContainsCriticalPrivileges(LogEntry log)
        {
            var criticalPrivileges = new[]
            {
                "SeDebugPrivilege",
                "SeTcbPrivilege", 
                "SeCreateTokenPrivilege",
                "SeImpersonatePrivilege",
                "SeAssignPrimaryTokenPrivilege",
                "SeLoadDriverPrivilege",
                "SeBackupPrivilege",
                "SeRestorePrivilege"
            };

            var logText = log.Message?.ToLowerInvariant() ?? "";
            return criticalPrivileges.Any(priv => logText.Contains(priv.ToLowerInvariant()));
        }

        /// <summary>
        /// Calculates confidence score for privilege escalation.
        /// </summary>
        /// <param name="logs">Related logs.</param>
        /// <returns>Confidence score between 0.0 and 1.0.</returns>
        private double CalculatePrivilegeEscalationConfidence(List<LogEntry> logs)
        {
            var baseConfidence = 0.6;
            var volumeBonus = Math.Min(0.2, logs.Count * 0.05);
            var diversityBonus = Math.Min(0.2, logs.Select(l => l.EventId).Distinct().Count() * 0.1);
            
            return Math.Min(1.0, baseConfidence + volumeBonus + diversityBonus);
        }

        /// <summary>
        /// Determines severity based on event count.
        /// </summary>
        /// <param name="count">Number of events.</param>
        /// <param name="threshold">Base threshold.</param>
        /// <returns>Severity level string.</returns>
        private string DetermineSeverityByCount(int count, int threshold)
        {
            if (count >= threshold * 3) return "Critical";
            if (count >= threshold * 2) return "High";
            return "Medium";
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["PrivilegeUseThreshold"] = _privilegeUseThreshold,
                ["TimeWindowMinutes"] = _timeWindowMinutes,
                ["SupportedEventIds"] = new[] { "4672", "4673", "4674", "4697", "4698", "4699", "4700", "4701", "4702" }
            };
        }
    }
} 