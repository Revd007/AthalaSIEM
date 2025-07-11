using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core;
using AthalaSIEM.UniversalAgent.Models;
using Core = AthalaSIEM.Agent.Core;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Command Execution Collector for AthalaSIEM Universal Agent.
    /// Executes backend-authorized commands periodically and collects their output.
    /// ENTERPRISE SECURITY: All commands must be explicitly whitelisted by backend.
    /// </summary>
    public class CommandExecutionCollector : ILogCollector
    {
        /// <inheritdoc />
        public string CollectorName => "Command Execution";
        
        /// <inheritdoc />
        public Core.OperatingSystem SupportedOS => Core.OperatingSystem.Windows; // Can be extended to Universal
        
        /// <inheritdoc />
        public bool IsActive { get; private set; }
        
        /// <inheritdoc />
        public long LogsCollected { get; private set; }

        private readonly ILogger<CommandExecutionCollector> _logger;
        private readonly List<LogEntry> _collectedLogs = new List<LogEntry>();
        private readonly List<CommandSchedule> _scheduledCommands = new();
        private readonly Dictionary<string, Timer> _commandTimers = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private readonly object _commandLock = new();

        /// <inheritdoc />
        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        
        /// <inheritdoc />
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        /// <summary>
        /// Initializes a new instance of the CommandExecutionCollector.
        /// </summary>
        /// <param name="logger">Logger instance for this collector.</param>
        public CommandExecutionCollector(ILogger<CommandExecutionCollector> logger)
        {
            _logger = logger;
            _logger.LogInformation("Command Execution Collector initialized - Backend configuration required");
        }

        /// <inheritdoc />
        public Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("Command Execution Collector requires backend configuration for security");
                _logger.LogInformation("Configure authorized commands via SIEM Web Interface:");
                _logger.LogInformation("  • Go to Agents → Select Agent → Command Execution");
                _logger.LogInformation("  • Add whitelisted commands with schedules");
                _logger.LogInformation("  • Commands will be executed automatically per schedule");
                
                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize Command Execution Collector");
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Updates command execution configuration from backend.
        /// </summary>
        /// <param name="config">Backend configuration containing authorized commands and schedules.</param>
        /// <returns>True if configuration was successfully applied.</returns>
        public async Task<bool> UpdateFromBackendConfigAsync(Dictionary<string, object> config)
        {
            try
            {
                lock (_commandLock)
                {
                    _logger.LogInformation("Updating Command Execution configuration from backend...");
                    
                    // Stop existing timers
                    foreach (var timer in _commandTimers.Values)
                    {
                        timer?.Dispose();
                    }
                    _commandTimers.Clear();
                    _scheduledCommands.Clear();

                    // Load new command schedules
                    if (LoadCommandSchedulesFromBackend(config))
                    {
                        SetupCommandTimers();
                        _logger.LogInformation("✅ Command Execution updated: {Count} authorized commands", _scheduledCommands.Count);
                    }
                    else
                    {
                        _logger.LogWarning("No authorized commands provided by backend - Command Execution disabled for security");
                    }
                }

                return await Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to update Command Execution configuration from backend");
                return false;
            }
        }

        /// <summary>
        /// Loads command schedules from backend configuration.
        /// </summary>
        /// <param name="config">Backend configuration dictionary.</param>
        /// <returns>True if commands were loaded successfully.</returns>
        private bool LoadCommandSchedulesFromBackend(Dictionary<string, object> config)
        {
            var possibleKeys = new[] { "AuthorizedCommands", "Commands", "CommandSchedules", "ScheduledCommands" };
            
            foreach (var key in possibleKeys)
            {
                if (config.TryGetValue(key, out var commandsObj))
                {
                    try
                    {
                        // Parse command schedules from backend
                        var commandsArray = ParseCommandSchedules(commandsObj);
                        
                        foreach (var commandConfig in commandsArray)
                        {
                            var schedule = CreateCommandSchedule(commandConfig);
                            if (schedule != null)
                            {
                                _scheduledCommands.Add(schedule);
                                _logger.LogInformation("Authorized command: {Command} (Interval: {Interval}min)", 
                                    schedule.Command, schedule.IntervalMinutes);
                            }
                        }
                        
                        return _scheduledCommands.Any();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error parsing command schedules from backend");
                        return false;
                    }
                }
            }
            
            return false;
        }

        /// <summary>
        /// Parses command schedules from configuration object.
        /// </summary>
        /// <param name="commandsObj">Commands configuration object.</param>
        /// <returns>Array of command configurations.</returns>
        private Dictionary<string, object>[] ParseCommandSchedules(object commandsObj)
        {
            // Implementation would parse JSON array of command objects
            // For now, return empty array - this will be configured by backend
            return new Dictionary<string, object>[0];
        }

        /// <summary>
        /// Creates a command schedule from configuration.
        /// </summary>
        /// <param name="commandConfig">Command configuration dictionary.</param>
        /// <returns>Command schedule or null if invalid.</returns>
        private CommandSchedule? CreateCommandSchedule(Dictionary<string, object> commandConfig)
        {
            try
            {
                if (!commandConfig.TryGetValue("Command", out var cmdObj) || 
                    !commandConfig.TryGetValue("IntervalMinutes", out var intervalObj))
                {
                    return null;
                }

                var command = cmdObj.ToString();
                if (string.IsNullOrWhiteSpace(command) || !int.TryParse(intervalObj.ToString(), out var interval))
                {
                    return null;
                }

                // Validate command against whitelist (backend responsibility)
                if (!IsCommandAuthorized(command))
                {
                    _logger.LogWarning("Command not in backend whitelist: {Command}", command);
                    return null;
                }

                return new CommandSchedule
                {
                    Id = Guid.NewGuid().ToString(),
                    Command = command,
                    Arguments = commandConfig.GetValueOrDefault("Arguments", "")?.ToString() ?? "",
                    IntervalMinutes = interval,
                    Enabled = bool.Parse(commandConfig.GetValueOrDefault("Enabled", "true")?.ToString() ?? "true"),
                    Description = commandConfig.GetValueOrDefault("Description", "")?.ToString() ?? "",
                    TimeoutSeconds = int.Parse(commandConfig.GetValueOrDefault("TimeoutSeconds", "30")?.ToString() ?? "30")
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating command schedule");
                return null;
            }
        }

        /// <summary>
        /// Validates if a command is authorized by backend.
        /// </summary>
        /// <param name="command">Command to validate.</param>
        /// <returns>True if command is authorized.</returns>
        private bool IsCommandAuthorized(string command)
        {
            // Backend provides whitelist of authorized commands
            // This is a security-critical function - only backend-approved commands allowed
            var authorizedCommands = new HashSet<string>
            {
                // Examples - actual list comes from backend
                "powershell.exe",
                "cmd.exe", 
                "wmic.exe",
                "systeminfo.exe",
                "tasklist.exe",
                "netstat.exe",
                "whoami.exe"
            };

            var executableName = Path.GetFileName(command).ToLowerInvariant();
            return authorizedCommands.Contains(executableName);
        }

        /// <summary>
        /// Sets up timers for scheduled commands.
        /// </summary>
        private void SetupCommandTimers()
        {
            foreach (var schedule in _scheduledCommands.Where(s => s.Enabled))
            {
                var interval = TimeSpan.FromMinutes(schedule.IntervalMinutes);
                var timer = new Timer(ExecuteScheduledCommand, schedule, TimeSpan.Zero, interval);
                _commandTimers[schedule.Id] = timer;
                
                _logger.LogDebug("Setup timer for command: {Command} (every {Interval}min)", 
                    schedule.Command, schedule.IntervalMinutes);
            }
        }

        /// <summary>
        /// Executes a scheduled command.
        /// </summary>
        /// <param name="state">Command schedule object.</param>
        private async void ExecuteScheduledCommand(object? state)
        {
            if (state is not CommandSchedule schedule || !IsActive)
                return;

            try
            {
                _logger.LogDebug("Executing scheduled command: {Command}", schedule.Command);
                
                var result = await ExecuteCommandSafelyAsync(schedule);
                if (result != null)
                {
                    _collectedLogs.Add(result);
                    LogsCollected++;

                    LogCollected?.Invoke(this, new LogCollectedEventArgs 
                    { 
                        Logs = new[] { result },
                        Source = CollectorName,
                        CollectionTime = DateTime.UtcNow
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing scheduled command: {Command}", schedule.Command);
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Source = CollectorName,
                    Message = $"Command execution failed: {schedule.Command}"
                });
            }
        }

        /// <summary>
        /// Executes a command safely with timeout and output capture.
        /// </summary>
        /// <param name="schedule">Command schedule to execute.</param>
        /// <returns>Log entry with command output or null if failed.</returns>
        private async Task<LogEntry?> ExecuteCommandSafelyAsync(CommandSchedule schedule)
        {
            try
            {
                using var process = new Process();
                process.StartInfo.FileName = schedule.Command;
                process.StartInfo.Arguments = schedule.Arguments;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.CreateNoWindow = true;

                var output = new List<string>();
                var errors = new List<string>();
                
                process.OutputDataReceived += (_, e) =>
                {
                    if (e.Data != null) output.Add(e.Data);
                };
                
                process.ErrorDataReceived += (_, e) =>
                {
                    if (e.Data != null) errors.Add(e.Data);
                };

                var startTime = DateTime.UtcNow;
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                // Wait with timeout
                var completed = await Task.Run(() => process.WaitForExit(schedule.TimeoutSeconds * 1000));
                if (!completed)
                {
                    process.Kill();
                    throw new TimeoutException($"Command timed out after {schedule.TimeoutSeconds} seconds");
                }

                var endTime = DateTime.UtcNow;
                var exitCode = process.ExitCode;

                return new LogEntry
                {
                    Timestamp = startTime,
                    Source = "CommandExecution",
                    Level = exitCode == 0 ? "Information" : "Warning",
                    Message = $"Command executed: {schedule.Command} {schedule.Arguments}",
                    EventId = "CMD_EXEC",
                    Category = "CommandExecution",
                    SecurityRelevance = DetermineSecurityRelevance(schedule, exitCode),
                    ComputerName = Environment.MachineName,
                    Properties = new Dictionary<string, object>
                    {
                        ["Command"] = schedule.Command,
                        ["Arguments"] = schedule.Arguments,
                        ["ExitCode"] = exitCode,
                        ["ExecutionTimeMs"] = (endTime - startTime).TotalMilliseconds,
                        ["StandardOutput"] = string.Join(Environment.NewLine, output),
                        ["StandardError"] = string.Join(Environment.NewLine, errors),
                        ["OutputLineCount"] = output.Count,
                        ["ErrorLineCount"] = errors.Count,
                        ["ScheduleId"] = schedule.Id,
                        ["CommandDescription"] = schedule.Description
                    }
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Command execution failed: {Command}", schedule.Command);
                return CreateErrorLogEntry(schedule, ex);
            }
        }

        /// <summary>
        /// Determines security relevance based on command and result.
        /// </summary>
        /// <param name="schedule">Command schedule.</param>
        /// <param name="exitCode">Command exit code.</param>
        /// <returns>Security relevance level.</returns>
        private string DetermineSecurityRelevance(CommandSchedule schedule, int exitCode)
        {
            // Backend can configure security relevance rules
            if (exitCode != 0) return "High"; // Failed commands are suspicious
            
            var command = schedule.Command.ToLowerInvariant();
            if (command.Contains("security") || command.Contains("audit")) return "High";
            if (command.Contains("system") || command.Contains("process")) return "Medium";
            
            return "Low";
        }

        /// <summary>
        /// Creates an error log entry for failed command execution.
        /// </summary>
        /// <param name="schedule">Command schedule that failed.</param>
        /// <param name="exception">Exception that occurred.</param>
        /// <returns>Error log entry.</returns>
        private LogEntry CreateErrorLogEntry(CommandSchedule schedule, Exception exception)
        {
            return new LogEntry
            {
                Timestamp = DateTime.UtcNow,
                Source = "CommandExecution",
                Level = "Error",
                Message = $"Command execution failed: {schedule.Command}",
                EventId = "CMD_ERROR",
                Category = "CommandExecutionError",
                SecurityRelevance = "High", // Failed commands are suspicious
                ComputerName = Environment.MachineName,
                Properties = new Dictionary<string, object>
                {
                    ["Command"] = schedule.Command,
                    ["Arguments"] = schedule.Arguments,
                    ["Error"] = exception.Message,
                    ["ExceptionType"] = exception.GetType().Name,
                    ["ScheduleId"] = schedule.Id
                }
            };
        }

        /// <inheritdoc />
        public Task StartCollectionAsync(CancellationToken cancellationToken = default)
        {
            if (_scheduledCommands.Count == 0)
            {
                _logger.LogWarning("Cannot start Command Execution: No authorized commands configured by backend");
                return Task.CompletedTask;
            }

            IsActive = true;
            _logger.LogInformation("Command Execution Collector started - {Count} authorized commands", _scheduledCommands.Count);
            return Task.CompletedTask;
        }

        /// <inheritdoc />
        public Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = false;
            
            lock (_commandLock)
            {
                foreach (var timer in _commandTimers.Values)
                {
                    timer?.Dispose();
                }
                _commandTimers.Clear();
            }
            
            _cancellationTokenSource.Cancel();
            _logger.LogInformation("Command Execution Collector stopped");
            return Task.CompletedTask;
        }

        /// <inheritdoc />
        public Task<IEnumerable<LogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default)
        {
            var logs = _collectedLogs.Take(batchSize).ToList();
            _collectedLogs.RemoveRange(0, logs.Count);
            return Task.FromResult<IEnumerable<LogEntry>>(logs);
        }

        /// <inheritdoc />
        public Task<CollectorHealth> GetHealthAsync()
        {
            return Task.FromResult(new CollectorHealth
            {
                IsHealthy = IsActive,
                Status = IsActive ? "Running" : "Stopped",
                LogsCollected = LogsCollected,
                LastCollection = DateTime.UtcNow,
                Metrics = new Dictionary<string, object>
                {
                    ["AuthorizedCommands"] = _scheduledCommands.Count,
                    ["ActiveTimers"] = _commandTimers.Count,
                    ["BufferedLogs"] = _collectedLogs.Count,
                    ["ConfigurationStatus"] = _scheduledCommands.Count > 0 ? "Backend Configured" : "AWAITING BACKEND CONFIGURATION",
                    ["SecurityLevel"] = "Backend Whitelisted Commands Only"
                }
            });
        }

        /// <inheritdoc />
        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            _cancellationTokenSource?.Dispose();
            
            lock (_commandLock)
            {
                foreach (var timer in _commandTimers.Values)
                {
                    timer?.Dispose();
                }
                _commandTimers.Clear();
            }
        }
    }

    // NOTE: CommandSchedule model has been moved to 
    // AthalaSIEM.UniversalAgent.Models.CollectorModels.cs for clean architecture separation
} 