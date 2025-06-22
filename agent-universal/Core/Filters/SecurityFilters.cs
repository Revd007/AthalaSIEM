using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.Agent.Core.Filters
{
    /// <summary>
    /// Enterprise-grade security relevance filter that processes logs based on configurable security levels.
    /// Supports dynamic configuration for different organizational security requirements.
    /// </summary>
    public class EnterpriseSecurityRelevanceFilter : ILogFilter
    {
        private readonly ILogger<EnterpriseSecurityRelevanceFilter> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsProcessed;
        private long _logsPassed;
        private HashSet<string> _allowedSecurityLevels = new();

        /// <inheritdoc />
        public string Name => "Enterprise Security Relevance Filter";

        /// <inheritdoc />
        public string Description => "Filters logs based on configurable security relevance levels for enterprise environments";

        /// <inheritdoc />
        public int Priority => 100; // High priority for security filtering

        /// <summary>
        /// Initializes a new instance of the EnterpriseSecurityRelevanceFilter.
        /// </summary>
        /// <param name="logger">Logger instance for this filter.</param>
        public EnterpriseSecurityRelevanceFilter(ILogger<EnterpriseSecurityRelevanceFilter> logger)
        {
            _logger = logger;
            
            // Default security levels - can be overridden by configuration
            _allowedSecurityLevels = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Critical", "High", "Medium"
            };
        }

        /// <summary>
        /// Initializes the filter with configuration settings.
        /// </summary>
        /// <param name="config">Configuration dictionary containing filter settings.</param>
        public void Initialize(Dictionary<string, object> config)
        {
            if (config.TryGetValue("AllowedSecurityLevels", out var levels))
            {
                if (levels is string[] levelArray)
                {
                    _allowedSecurityLevels = new HashSet<string>(levelArray, StringComparer.OrdinalIgnoreCase);
                }
                else if (levels is string levelString)
                {
                    _allowedSecurityLevels = new HashSet<string>(
                        levelString.Split(',').Select(s => s.Trim()),
                        StringComparer.OrdinalIgnoreCase);
                }
            }

            _logger.LogInformation("Security relevance filter initialized with levels: {Levels}", 
                string.Join(", ", _allowedSecurityLevels));
        }

        /// <inheritdoc />
        public Task<bool> ShouldProcessAsync(LogEntry log)
        {
            _processingTimer.Start();
            _logsProcessed++;

            try
            {
                var shouldProcess = _allowedSecurityLevels.Contains(log.SecurityRelevance);
                
                if (shouldProcess)
                {
                    _logsPassed++;
                }

                return Task.FromResult(shouldProcess);
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsProcessed"] = _logsProcessed,
                ["LogsPassed"] = _logsPassed,
                ["LogsFiltered"] = _logsProcessed - _logsPassed,
                ["FilterEfficiency"] = _logsProcessed > 0 ? (double)(_logsProcessed - _logsPassed) / _logsProcessed * 100 : 0,
                ["AverageProcessingTimeMs"] = _logsProcessed > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsProcessed : 0,
                ["AllowedSecurityLevels"] = _allowedSecurityLevels.ToArray()
            };
        }
    }

    /// <summary>
    /// Comprehensive Windows Event ID filter for enterprise SIEM environments.
    /// Supports hundreds of security-relevant event IDs organized by category and threat type.
    /// </summary>
    public class EnterpriseWindowsEventIdFilter : ILogFilter
    {
        private readonly ILogger<EnterpriseWindowsEventIdFilter> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsProcessed;
        private long _logsPassed;
        
        // Comprehensive event ID collections organized by security category
        private Dictionary<string, HashSet<string>> _eventIdCategories = new();
        private HashSet<string> _enabledCategories = new();
        private HashSet<string> _allMonitoredEventIds = new();

        /// <inheritdoc />
        public string Name => "Enterprise Windows Event ID Filter";

        /// <inheritdoc />
        public string Description => "Comprehensive Windows Event ID filtering for enterprise security monitoring";

        /// <inheritdoc />
        public int Priority => 90;

        /// <summary>
        /// Initializes a new instance of the EnterpriseWindowsEventIdFilter.
        /// </summary>
        /// <param name="logger">Logger instance for this filter.</param>
        public EnterpriseWindowsEventIdFilter(ILogger<EnterpriseWindowsEventIdFilter> logger)
        {
            _logger = logger;
            InitializeEventIdCategories();
            
            // Default: Enable all categories
            _enabledCategories = new HashSet<string>(_eventIdCategories.Keys, StringComparer.OrdinalIgnoreCase);
            UpdateMonitoredEventIds();
        }

        /// <summary>
        /// Initializes the comprehensive event ID categories for enterprise monitoring.
        /// </summary>
        private void InitializeEventIdCategories()
        {
            _eventIdCategories = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase)
            {
                // Authentication Events (Logon/Logoff)
                ["Authentication"] = new HashSet<string>
                {
                    "4624", "4625", "4634", "4647", "4648", "4672", "4673", "4674", "4675", "4776", "4777",
                    "4778", "4779", "4800", "4801", "4802", "4803", "5376", "5377", "5378", "5632", "5633"
                },

                // Account Management
                ["AccountManagement"] = new HashSet<string>
                {
                    "4720", "4722", "4723", "4724", "4725", "4726", "4727", "4728", "4729", "4730", "4731",
                    "4732", "4733", "4734", "4735", "4737", "4738", "4739", "4740", "4741", "4742", "4743"
                },

                // Privilege Use
                ["PrivilegeUse"] = new HashSet<string>
                {
                    "4672", "4673", "4674", "4675", "4694", "4695", "4696", "4697", "4704", "4705"
                },

                // Object Access
                ["ObjectAccess"] = new HashSet<string>
                {
                    "4656", "4657", "4658", "4659", "4660", "4661", "4662", "4663", "4664", "4665",
                    "4666", "4667", "4668", "4670", "4671", "5140", "5142", "5143", "5144", "5145"
                },

                // Policy Changes
                ["PolicyChange"] = new HashSet<string>
                {
                    "4713", "4714", "4715", "4716", "4717", "4718", "4719", "4864", "4865", "4866",
                    "4867", "4902", "4904", "4905", "4906", "4907", "4908", "4912"
                },

                // System Events
                ["SystemEvents"] = new HashSet<string>
                {
                    "1100", "1102", "1104", "1105", "1108", "4608", "4609", "4610", "4611", "4612",
                    "4614", "4615", "4616", "4618", "4621", "4622", "5024", "5025", "5027", "5028"
                },

                // Process Activity
                ["ProcessActivity"] = new HashSet<string>
                {
                    "4688", "4689", "4696", "4697", "4698", "4699", "4700", "4701", "4702"
                },

                // Network Activity
                ["NetworkActivity"] = new HashSet<string>
                {
                    "5152", "5153", "5154", "5155", "5156", "5157", "5158", "5159"
                },

                // PowerShell Activity
                ["PowerShell"] = new HashSet<string>
                {
                    "4103", "4104", "4105", "4106", "24577", "24578", "53504"
                },

                // Certificate Services
                ["CertificateServices"] = new HashSet<string>
                {
                    "4868", "4869", "4870", "4871", "4872", "4873", "4874", "4875", "4876", "4877"
                },

                // Kerberos
                ["Kerberos"] = new HashSet<string>
                {
                    "4768", "4769", "4770", "4771", "4772", "4773", "4774", "4775"
                },

                // Terminal Services
                ["TerminalServices"] = new HashSet<string>
                {
                    "21", "22", "23", "24", "25", "131", "1149"
                }
            };
        }

        /// <summary>
        /// Initializes the filter with configuration settings.
        /// </summary>
        /// <param name="config">Configuration dictionary containing filter settings.</param>
        public void Initialize(Dictionary<string, object> config)
        {
            if (config.TryGetValue("EnabledCategories", out var categories))
            {
                if (categories is string[] categoryArray)
                {
                    _enabledCategories = new HashSet<string>(categoryArray, StringComparer.OrdinalIgnoreCase);
                }
                else if (categories is string categoryString)
                {
                    _enabledCategories = new HashSet<string>(
                        categoryString.Split(',').Select(s => s.Trim()),
                        StringComparer.OrdinalIgnoreCase);
                }
            }

            UpdateMonitoredEventIds();

            _logger.LogInformation("Enterprise Windows Event ID filter initialized with {CategoryCount} categories and {EventIdCount} event IDs",
                _enabledCategories.Count, _allMonitoredEventIds.Count);
        }

        /// <summary>
        /// Updates the collection of all monitored event IDs based on enabled categories.
        /// </summary>
        private void UpdateMonitoredEventIds()
        {
            _allMonitoredEventIds.Clear();
            foreach (var category in _enabledCategories)
            {
                if (_eventIdCategories.TryGetValue(category, out var eventIds))
                {
                    foreach (var eventId in eventIds)
                    {
                        _allMonitoredEventIds.Add(eventId);
                    }
                }
            }
        }

        /// <inheritdoc />
        public Task<bool> ShouldProcessAsync(LogEntry log)
        {
            _processingTimer.Start();
            _logsProcessed++;

            try
            {
                // Process all non-Windows events
                if (string.IsNullOrEmpty(log.EventId))
                {
                    _logsPassed++;
                    return Task.FromResult(true);
                }

                var shouldProcess = _allMonitoredEventIds.Contains(log.EventId);
                
                if (shouldProcess)
                {
                    _logsPassed++;
                }

                return Task.FromResult(shouldProcess);
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsProcessed"] = _logsProcessed,
                ["LogsPassed"] = _logsPassed,
                ["LogsFiltered"] = _logsProcessed - _logsPassed,
                ["FilterEfficiency"] = _logsProcessed > 0 ? (double)(_logsProcessed - _logsPassed) / _logsProcessed * 100 : 0,
                ["AverageProcessingTimeMs"] = _logsProcessed > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsProcessed : 0,
                ["EnabledCategories"] = _enabledCategories.ToArray(),
                ["MonitoredEventIdCount"] = _allMonitoredEventIds.Count,
                ["AvailableCategories"] = _eventIdCategories.Keys.ToArray()
            };
        }
    }

    /// <summary>
    /// Enterprise log level filter that processes logs based on configurable severity levels.
    /// Supports different filtering strategies for various operational modes.
    /// </summary>
    public class EnterpriseLogLevelFilter : ILogFilter
    {
        private readonly ILogger<EnterpriseLogLevelFilter> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsProcessed;
        private long _logsPassed;
        
        private HashSet<string> _allowedLogLevels = new();
        private bool _invertFilter; // If true, filter OUT the specified levels instead of filtering IN

        /// <inheritdoc />
        public string Name => "Enterprise Log Level Filter";

        /// <inheritdoc />
        public string Description => "Filters logs based on configurable log levels with support for inclusion/exclusion modes";

        /// <inheritdoc />
        public int Priority => 85;

        /// <summary>
        /// Initializes a new instance of the EnterpriseLogLevelFilter.
        /// </summary>
        /// <param name="logger">Logger instance for this filter.</param>
        public EnterpriseLogLevelFilter(ILogger<EnterpriseLogLevelFilter> logger)
        {
            _logger = logger;
            
            // Default: Filter out Debug and Verbose levels
            _allowedLogLevels = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Critical", "Error", "Warning", "Information"
            };
        }

        /// <summary>
        /// Initializes the filter with configuration settings.
        /// </summary>
        /// <param name="config">Configuration dictionary containing filter settings.</param>
        public void Initialize(Dictionary<string, object> config)
        {
            if (config.TryGetValue("AllowedLogLevels", out var levels))
            {
                if (levels is string[] levelArray)
                {
                    _allowedLogLevels = new HashSet<string>(levelArray, StringComparer.OrdinalIgnoreCase);
                }
                else if (levels is string levelString)
                {
                    _allowedLogLevels = new HashSet<string>(
                        levelString.Split(',').Select(s => s.Trim()),
                        StringComparer.OrdinalIgnoreCase);
                }
            }

            if (config.TryGetValue("InvertFilter", out var invert) && invert is bool invertValue)
            {
                _invertFilter = invertValue;
            }

            var filterMode = _invertFilter ? "exclusion" : "inclusion";
            _logger.LogInformation("Enterprise log level filter initialized in {FilterMode} mode with levels: {Levels}",
                filterMode, string.Join(", ", _allowedLogLevels));
        }

        /// <inheritdoc />
        public Task<bool> ShouldProcessAsync(LogEntry log)
        {
            _processingTimer.Start();
            _logsProcessed++;

            try
            {
                var levelMatches = _allowedLogLevels.Contains(log.Level);
                var shouldProcess = _invertFilter ? !levelMatches : levelMatches;
                
                if (shouldProcess)
                {
                    _logsPassed++;
                }

                return Task.FromResult(shouldProcess);
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsProcessed"] = _logsProcessed,
                ["LogsPassed"] = _logsPassed,
                ["LogsFiltered"] = _logsProcessed - _logsPassed,
                ["FilterEfficiency"] = _logsProcessed > 0 ? (double)(_logsProcessed - _logsPassed) / _logsProcessed * 100 : 0,
                ["AverageProcessingTimeMs"] = _logsProcessed > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsProcessed : 0,
                ["AllowedLogLevels"] = _allowedLogLevels.ToArray(),
                ["FilterMode"] = _invertFilter ? "Exclusion" : "Inclusion"
            };
        }
    }
} 