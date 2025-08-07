using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.Runtime.Versioning;
using Microsoft.Win32;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Models;
using Core = AthalaSIEM.UniversalAgent.Core;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Windows Registry Monitoring Collector for AthalaSIEM Universal Agent
    /// Monitors critical registry keys for security-relevant changes
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class WindowsRegistryCollector : ILogCollector
    {
        public string CollectorName => "Windows Registry Monitor";
        public Core.OperatingSystem SupportedOS => Core.OperatingSystem.Windows;
        public bool IsActive { get; private set; }
        public long LogsCollected { get; private set; }

        private readonly ILogger<WindowsRegistryCollector> _logger;
        private readonly IConfiguration? _configuration;
        private readonly List<LogEntry> _collectedLogs = new List<LogEntry>();
        private readonly Dictionary<string, Dictionary<string, object>> _registryBaseline = new();
        private readonly List<RegistryMonitorRule> _monitorRules = new List<RegistryMonitorRule>();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private Timer? _scanTimer;

        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        public WindowsRegistryCollector(ILogger<WindowsRegistryCollector> logger, IConfiguration? configuration = null)
        {
            _logger = logger;
            _configuration = configuration;
        }

        public async Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("Initializing Windows Registry Monitor");

                // Initialize critical registry monitoring rules
                InitializeCriticalRegistryRules();
                
                // Create initial baseline
                await CreateRegistryBaselineAsync();
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize Windows Registry Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex, 
                    Message = "Failed to initialize Windows Registry Collector",
                    Source = CollectorName
                });
                return false;
            }
        }

        public Task StartCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = true;
            
            // Use configurable scan interval (default 10 minutes if not configured by backend)
            var defaultScanIntervalMinutes = 10; // Default fallback - will be configurable by backend
            _scanTimer = new Timer(PerformRegistryScan, null, TimeSpan.Zero, TimeSpan.FromMinutes(defaultScanIntervalMinutes));
            
            _logger.LogInformation("Windows Registry Monitor started - scanning every {Interval} minutes (configurable by backend)", defaultScanIntervalMinutes);
            
            return Task.CompletedTask;
        }

        public Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = false;
            _scanTimer?.Dispose();
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
                    ["MonitoredRules"] = _monitorRules.Count,
                    ["BaselineKeys"] = _registryBaseline.Count,
                    ["BufferedLogs"] = _collectedLogs.Count
                }
            });
        }

        /// <summary>
        /// Initialize critical registry monitoring rules (following SIEM security patterns)
        /// </summary>
        private void InitializeCriticalRegistryRules()
        {
            _monitorRules.AddRange(new[]
            {
                // Startup Programs (Critical for malware persistence)
                new RegistryMonitorRule 
                { 
                    HiveRoot = RegistryHive.LocalMachine, 
                    KeyPath = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
                    Description = "Startup Programs - Local Machine",
                    SecurityRelevance = "Critical"
                },
                new RegistryMonitorRule 
                { 
                    HiveRoot = RegistryHive.CurrentUser, 
                    KeyPath = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
                    Description = "Startup Programs - Current User",
                    SecurityRelevance = "High"
                },
                
                // System Services (Critical for system integrity)
                new RegistryMonitorRule 
                { 
                    HiveRoot = RegistryHive.LocalMachine, 
                    KeyPath = @"SYSTEM\CurrentControlSet\Services",
                    Description = "Windows Services Configuration",
                    SecurityRelevance = "Critical",
                    MonitorSubKeys = false // Too many subkeys, monitor main key only
                },
                
                // Security Policy (Critical for security config)
                new RegistryMonitorRule 
                { 
                    HiveRoot = RegistryHive.LocalMachine, 
                    KeyPath = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies",
                    Description = "Windows Security Policies",
                    SecurityRelevance = "High"
                },
                
                // System Configuration
                new RegistryMonitorRule 
                { 
                    HiveRoot = RegistryHive.LocalMachine, 
                    KeyPath = @"SYSTEM\CurrentControlSet\Control\Session Manager",
                    Description = "Session Manager Configuration",
                    SecurityRelevance = "Medium"
                }
            });
        }

        /// <summary>
        /// Create initial registry baseline for comparison
        /// </summary>
        private Task CreateRegistryBaselineAsync()
        {
            foreach (var rule in _monitorRules)
            {
                try
                {
                    var keyData = ReadRegistryKey(rule.HiveRoot, rule.KeyPath, rule.MonitorSubKeys);
                    var baselineKey = $"{rule.HiveRoot}\\{rule.KeyPath}";
                    _registryBaseline[baselineKey] = keyData;
                    
                    _logger.LogDebug("Created baseline for registry key: {Key}", baselineKey);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to create baseline for registry key: {HiveRoot}\\{KeyPath}", 
                        rule.HiveRoot, rule.KeyPath);
                }
            }
            
            _logger.LogInformation("Registry baseline created for {Count} keys", _registryBaseline.Count);
            
            return Task.CompletedTask;
        }

        /// <summary>
        /// Perform registry scan and detect changes
        /// </summary>
        private void PerformRegistryScan(object? state)
        {
            if (!IsActive) return;

            try
            {
                _logger.LogDebug("Starting registry scan");

                foreach (var rule in _monitorRules)
                {
                    var baselineKey = $"{rule.HiveRoot}\\{rule.KeyPath}";
                    
                    if (!_registryBaseline.ContainsKey(baselineKey))
                        continue;

                    var currentData = ReadRegistryKey(rule.HiveRoot, rule.KeyPath, rule.MonitorSubKeys);
                    var baselineData = _registryBaseline[baselineKey];
                    
                    var changes = DetectChanges(baselineData, currentData, rule);
                    
                    foreach (var change in changes)
                    {
                        var logEntry = CreateRegistryChangeEvent(change, rule);
                        if (logEntry != null)
                        {
                            _collectedLogs.Add(logEntry);
                            LogsCollected++;
                            
                            LogCollected?.Invoke(this, new LogCollectedEventArgs 
                            { 
                                Logs = new[] { logEntry },
                                Source = CollectorName,
                                CollectionTime = DateTime.UtcNow
                            });
                        }
                    }
                    
                    // Update baseline with current data
                    _registryBaseline[baselineKey] = currentData;
                }
                
                // Prevent memory overflow using configurable limits
                var maxLogs = _configuration?.GetValue<int>("Collectors:2:Properties:MaxCollectedLogs", 2000) ?? 2000;
                var removeCount = _configuration?.GetValue<int>("Collectors:2:Properties:LogRemovalCount", 1000) ?? 1000;
                
                if (_collectedLogs.Count > maxLogs)
                {
                    _collectedLogs.RemoveRange(0, removeCount);
                    _logger.LogDebug("Registry log collection limit reached. Removed {RemoveCount} oldest logs. Max={MaxLogs}", 
                        removeCount, maxLogs);
                }

                _logger.LogDebug("Registry scan completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during registry scan");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex,
                    Source = CollectorName,
                    Message = "Error during registry scan"
                });
            }
        }

        /// <summary>
        /// Read registry key data
        /// </summary>
        private Dictionary<string, object> ReadRegistryKey(RegistryHive hive, string keyPath, bool includeSubKeys = false)
        {
            var data = new Dictionary<string, object>();
            
            try
            {
                using var baseKey = RegistryKey.OpenBaseKey(hive, RegistryView.Registry64);
                using var key = baseKey.OpenSubKey(keyPath, false);
                
                if (key == null) return data;

                // Read values
                foreach (var valueName in key.GetValueNames().Take(50)) // Limit for performance
                {
                    try
                    {
                        var value = key.GetValue(valueName);
                        var valueType = key.GetValueKind(valueName);
                        
                        data[$"VALUE:{valueName}"] = new
                        {
                            Value = value?.ToString() ?? "",
                            Type = valueType.ToString(),
                            Size = value?.ToString()?.Length ?? 0
                        };
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to read registry value: {ValueName}", valueName);
                    }
                }

                // Read subkeys (names only for performance)
                if (includeSubKeys)
                {
                    foreach (var subKeyName in key.GetSubKeyNames().Take(20)) // Limit for performance
                    {
                        data[$"SUBKEY:{subKeyName}"] = new
                        {
                            Name = subKeyName,
                            Type = "SubKey"
                        };
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Could not read registry key: {HiveRoot}\\{KeyPath}", hive, keyPath);
            }
            
            return data;
        }

        /// <summary>
        /// Detect changes between baseline and current registry data
        /// </summary>
        private List<RegistryChange> DetectChanges(Dictionary<string, object> baseline, 
            Dictionary<string, object> current, RegistryMonitorRule rule)
        {
            var changes = new List<RegistryChange>();
            
            // Detect new entries
            foreach (var kvp in current)
            {
                if (!baseline.ContainsKey(kvp.Key))
                {
                    changes.Add(new RegistryChange
                    {
                        ChangeType = "Added",
                        KeyPath = $"{rule.HiveRoot}\\{rule.KeyPath}",
                        ValueName = kvp.Key,
                        NewValue = kvp.Value,
                        Rule = rule
                    });
                }
            }
            
            // Detect removed entries
            foreach (var kvp in baseline)
            {
                if (!current.ContainsKey(kvp.Key))
                {
                    changes.Add(new RegistryChange
                    {
                        ChangeType = "Removed",
                        KeyPath = $"{rule.HiveRoot}\\{rule.KeyPath}",
                        ValueName = kvp.Key,
                        OldValue = kvp.Value,
                        Rule = rule
                    });
                }
            }
            
            // Detect modified entries
            foreach (var kvp in current)
            {
                if (baseline.ContainsKey(kvp.Key))
                {
                    var oldValue = baseline[kvp.Key]?.ToString() ?? "";
                    var newValue = kvp.Value?.ToString() ?? "";
                    
                    if (oldValue != newValue)
                    {
                        changes.Add(new RegistryChange
                        {
                            ChangeType = "Modified",
                            KeyPath = $"{rule.HiveRoot}\\{rule.KeyPath}",
                            ValueName = kvp.Key,
                            OldValue = baseline[kvp.Key],
                            NewValue = kvp.Value,
                            Rule = rule
                        });
                    }
                }
            }
            
            return changes;
        }

        /// <summary>
        /// Create registry change event following SIEM patterns
        /// </summary>
        private LogEntry? CreateRegistryChangeEvent(RegistryChange change, RegistryMonitorRule rule)
        {
            try
            {
                var logEntry = new LogEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("REG"),
                    Timestamp = DateTime.UtcNow,
                    Source = "WindowsRegistry",
                    Level = DetermineEventLevel(change, rule),
                    Message = $"Registry {change.ChangeType}: {change.KeyPath}\\{change.ValueName}",
                    EventId = $"REG_{change.ChangeType.ToUpper()}",
                    Category = "RegistryMonitoring",
                    SecurityRelevance = rule.SecurityRelevance,
                    CollectorType = "WindowsRegistry",
                    Properties = new Dictionary<string, object>
                    {
                        ["ChangeType"] = change.ChangeType,
                        ["KeyPath"] = change.KeyPath,
                        ["ValueName"] = change.ValueName,
                        ["OldValue"] = change.OldValue?.ToString() ?? "",
                        ["NewValue"] = change.NewValue?.ToString() ?? "",
                        ["RuleDescription"] = rule.Description,
                        ["SecurityRelevance"] = rule.SecurityRelevance,
                        ["ComputerName"] = Environment.MachineName,
                        ["ThreatIndicators"] = AnalyzeThreatIndicators(change)
                    }
                };

                return logEntry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating registry change event");
                return null;
            }
        }

        /// <summary>
        /// Determine event level based on change and rule
        /// </summary>
        private string DetermineEventLevel(RegistryChange change, RegistryMonitorRule rule)
        {
            return rule.SecurityRelevance switch
            {
                "Critical" => "Error",
                "High" => "Warning",
                "Medium" => "Information",
                _ => "Information"
            };
        }

        /// <summary>
        /// Analyze threat indicators for registry changes
        /// </summary>
        private List<string> AnalyzeThreatIndicators(RegistryChange change)
        {
            var indicators = new List<string>();
            var valueName = change.ValueName?.ToLowerInvariant() ?? "";
            var newValue = change.NewValue?.ToString()?.ToLowerInvariant() ?? "";
            
            // Suspicious executable paths
            if (newValue.Contains(".exe") || newValue.Contains(".dll") || newValue.Contains(".scr"))
            {
                indicators.Add("executable_in_registry");
            }
            
            // Script-based threats
            if (newValue.Contains("powershell") || newValue.Contains("cmd.exe") || 
                newValue.Contains("wscript") || newValue.Contains("cscript"))
            {
                indicators.Add("script_execution");
            }
            
            // Network-related changes
            if (newValue.Contains("http://") || newValue.Contains("https://") || 
                newValue.Contains("ftp://"))
            {
                indicators.Add("network_activity");
            }
            
            // Startup/persistence mechanisms
            if (change.KeyPath.Contains("Run") || change.KeyPath.Contains("Services"))
            {
                indicators.Add("persistence_mechanism");
            }
            
            return indicators;
        }

        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            _scanTimer?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }

    // NOTE: RegistryMonitorRule and RegistryChange models have been moved to 
    // AthalaSIEM.UniversalAgent.Models.CollectorModels.cs for clean architecture separation
} 
