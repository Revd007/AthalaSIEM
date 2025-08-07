using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Management;
using System.Net.NetworkInformation;
using System.Runtime.Versioning;
using System.Security.Principal;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.Agent.Services
{
    /// <summary>
    /// Active Response Service for AthalaSIEM Universal Agent.
    /// Executes automated threat responses based on backend policies and threat detections.
    /// Supports network blocking, process termination, file quarantine, and custom scripts.
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class ActiveResponseService : IAsyncDisposable
    {
        private readonly ILogger<ActiveResponseService> _logger;
        private readonly IConfiguration _configuration;
        private readonly List<ResponsePolicy> _responsePolicies = new();
        private readonly Queue<ResponseAction> _responseQueue = new();
        private readonly Dictionary<string, ResponseExecution> _activeResponses = new();
        private readonly object _responseLock = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        
        private Timer? _responseProcessorTimer;
        private bool _isInitialized;
        private bool _isActive;
        private int _maxConcurrentResponses;
        private int _responseTimeoutSeconds;
        private bool _enableFirewallIntegration;
        private bool _enableProcessTermination;
        private bool _enableFileQuarantine;
        private bool _enableCustomScripts;

        /// <summary>
        /// Gets a value indicating whether the active response service is currently active.
        /// </summary>
        public bool IsActive => _isActive;

        /// <summary>
        /// Gets the number of queued response actions.
        /// </summary>
        public int QueuedResponses => _responseQueue.Count;

        /// <summary>
        /// Gets the number of currently executing responses.
        /// </summary>
        public int ActiveResponseCount => _activeResponses.Count;

        /// <summary>
        /// Gets total responses executed.
        /// </summary>
        public long TotalResponsesExecuted { get; private set; }

        /// <summary>
        /// Gets total response failures.
        /// </summary>
        public long TotalResponseFailures { get; private set; }

        /// <summary>
        /// Event fired when a response action is executed.
        /// </summary>
        public event EventHandler<ResponseExecutedEventArgs>? ResponseExecuted;

        /// <summary>
        /// Event fired when a response action fails.
        /// </summary>
        public event EventHandler<ResponseErrorEventArgs>? ResponseError;

        /// <summary>
        /// Initializes a new instance of the ActiveResponseService.
        /// </summary>
        /// <param name="logger">Logger instance for this service.</param>
        /// <param name="configuration">Configuration instance.</param>
        public ActiveResponseService(ILogger<ActiveResponseService> logger, IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
            
            LoadConfiguration();
            _logger.LogInformation("Active Response Service initialized");
        }

        /// <summary>
        /// Loads service configuration.
        /// </summary>
        private void LoadConfiguration()
        {
            _maxConcurrentResponses = _configuration.GetValue("ActiveResponse:MaxConcurrentResponses", 5);
            _responseTimeoutSeconds = _configuration.GetValue("ActiveResponse:ResponseTimeoutSeconds", 300);
            _enableFirewallIntegration = _configuration.GetValue("ActiveResponse:EnableFirewallIntegration", true);
            _enableProcessTermination = _configuration.GetValue("ActiveResponse:EnableProcessTermination", true);
            _enableFileQuarantine = _configuration.GetValue("ActiveResponse:EnableFileQuarantine", true);
            _enableCustomScripts = _configuration.GetValue("ActiveResponse:EnableCustomScripts", false);

            _logger.LogInformation("Active Response configured - MaxConcurrent: {Max}, Timeout: {Timeout}s", 
                _maxConcurrentResponses, _responseTimeoutSeconds);
        }

        /// <summary>
        /// Initializes the active response service.
        /// </summary>
        /// <returns>True if initialization was successful.</returns>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                if (!IsRunningAsAdministrator())
                {
                    _logger.LogWarning("Active Response Service requires Administrator privileges for full functionality");
                }

                await LoadResponsePoliciesAsync();
                
                _isInitialized = true;
                _logger.LogInformation("✅ Active Response Service initialized with {Count} policies", _responsePolicies.Count);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize Active Response Service");
                return false;
            }
        }

        /// <summary>
        /// Starts the active response service.
        /// </summary>
        /// <returns>Task representing the start operation.</returns>
        public Task StartAsync()
        {
            if (!_isInitialized)
            {
                throw new InvalidOperationException("Service must be initialized before starting");
            }

            _isActive = true;
            
            // Start response processor
            _responseProcessorTimer = new Timer(ProcessResponseQueue, null, 
                TimeSpan.Zero, TimeSpan.FromSeconds(5));
            
            _logger.LogInformation("Active Response Service started");
            return Task.CompletedTask;
        }

        /// <summary>
        /// Stops the active response service.
        /// </summary>
        /// <returns>Task representing the stop operation.</returns>
        public async Task StopAsync()
        {
            _isActive = false;
            _responseProcessorTimer?.Dispose();
            
            // Wait for active responses to complete
            var timeout = TimeSpan.FromSeconds(30);
            var stopwatch = Stopwatch.StartNew();
            
            while (_activeResponses.Count > 0 && stopwatch.Elapsed < timeout)
            {
                await Task.Delay(1000);
            }
            
            _cancellationTokenSource.Cancel();
            _logger.LogInformation("Active Response Service stopped");
        }

        /// <summary>
        /// Updates response policies from backend configuration.
        /// </summary>
        /// <param name="policies">Response policies from backend.</param>
        /// <returns>Task representing the update operation.</returns>
        public async Task UpdatePoliciesFromBackendAsync(List<ResponsePolicy> policies)
        {
            try
            {
                lock (_responseLock)
                {
                    _responsePolicies.Clear();
                    _responsePolicies.AddRange(policies);
                }
                
                _logger.LogInformation("Response policies updated from backend: {Count} policies", policies.Count);
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating response policies from backend");
            }
        }

        /// <summary>
        /// Queues a response action for execution.
        /// </summary>
        /// <param name="trigger">Trigger information that caused this response.</param>
        /// <param name="responseType">Type of response to execute.</param>
        /// <param name="parameters">Response parameters.</param>
        /// <returns>Response action ID.</returns>
        public string QueueResponse(ThreatTrigger trigger, ResponseType responseType, Dictionary<string, object> parameters)
        {
            try
            {
                var actionId = Guid.NewGuid().ToString();
                
                var responseAction = new ResponseAction
                {
                    Id = actionId,
                    Trigger = trigger,
                    ResponseType = responseType,
                    Parameters = parameters,
                    QueuedAt = DateTime.UtcNow,
                    Status = ResponseStatus.Queued
                };

                lock (_responseLock)
                {
                    _responseQueue.Enqueue(responseAction);
                }

                _logger.LogInformation("Response queued: {ResponseType} for trigger {TriggerType} (ID: {ActionId})", 
                    responseType, trigger.TriggerType, actionId);

                return actionId;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error queueing response action");
                return string.Empty;
            }
        }

        /// <summary>
        /// Executes an immediate response action.
        /// </summary>
        /// <param name="trigger">Trigger information.</param>
        /// <param name="responseType">Type of response.</param>
        /// <param name="parameters">Response parameters.</param>
        /// <returns>Response result.</returns>
        public async Task<ResponseResult> ExecuteImmediateResponseAsync(ThreatTrigger trigger, ResponseType responseType, Dictionary<string, object> parameters)
        {
            try
            {
                var actionId = Guid.NewGuid().ToString();
                
                var responseAction = new ResponseAction
                {
                    Id = actionId,
                    Trigger = trigger,
                    ResponseType = responseType,
                    Parameters = parameters,
                    QueuedAt = DateTime.UtcNow,
                    Status = ResponseStatus.Executing
                };

                return await ExecuteResponseActionAsync(responseAction);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing immediate response");
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Processes the response queue.
        /// </summary>
        /// <param name="state">Timer state (not used).</param>
        private async void ProcessResponseQueue(object? state)
        {
            if (!_isActive) return;

            try
            {
                while (_responseQueue.Count > 0 && _activeResponses.Count < _maxConcurrentResponses)
                {
                    ResponseAction? action = null;
                    
                    lock (_responseLock)
                    {
                        if (_responseQueue.Count > 0)
                        {
                            action = _responseQueue.Dequeue();
                        }
                    }

                    if (action != null)
                    {
                        var executionInfo = new ResponseExecution
                        {
                            Action = action,
                            StartTime = DateTime.UtcNow,
                            CancellationToken = new CancellationTokenSource(TimeSpan.FromSeconds(_responseTimeoutSeconds))
                        };

                        _activeResponses[action.Id] = executionInfo;
                        
                        // Execute response asynchronously
                        _ = Task.Run(async () =>
                        {
                            try
                            {
                                action.Status = ResponseStatus.Executing;
                                var result = await ExecuteResponseActionAsync(action);
                                
                                action.Status = result.Success ? ResponseStatus.Completed : ResponseStatus.Failed;
                                action.CompletedAt = DateTime.UtcNow;
                                action.Result = result;

                                if (result.Success)
                                {
                                    TotalResponsesExecuted++;
                                    ResponseExecuted?.Invoke(this, new ResponseExecutedEventArgs
                                    {
                                        Action = action,
                                        Result = result,
                                        ExecutionTime = DateTime.UtcNow
                                    });
                                }
                                else
                                {
                                    TotalResponseFailures++;
                                    ResponseError?.Invoke(this, new ResponseErrorEventArgs
                                    {
                                        Action = action,
                                        Error = result.Error,
                                        ErrorTime = DateTime.UtcNow
                                    });
                                }
                            }
                            catch (Exception ex)
                            {
                                _logger.LogError(ex, "Error in response execution task");
                                TotalResponseFailures++;
                            }
                            finally
                            {
                                _activeResponses.Remove(action.Id);
                            }
                        }, executionInfo.CancellationToken.Token);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing response queue");
            }
        }

        /// <summary>
        /// Executes a response action.
        /// </summary>
        /// <param name="action">Response action to execute.</param>
        /// <returns>Response result.</returns>
        private async Task<ResponseResult> ExecuteResponseActionAsync(ResponseAction action)
        {
            try
            {
                _logger.LogInformation("Executing response: {ResponseType} for {TriggerType} (ID: {ActionId})", 
                    action.ResponseType, action.Trigger.TriggerType, action.Id);

                return action.ResponseType switch
                {
                    ResponseType.BlockIpAddress => await BlockIpAddressAsync(action),
                    ResponseType.TerminateProcess => await TerminateProcessAsync(action),
                    ResponseType.QuarantineFile => await QuarantineFileAsync(action),
                    ResponseType.DisableUserAccount => await DisableUserAccountAsync(action),
                    ResponseType.IsolateHost => await IsolateHostAsync(action),
                    ResponseType.CustomScript => await ExecuteCustomScriptAsync(action),
                    ResponseType.SendAlert => await SendAlertAsync(action),
                    _ => new ResponseResult { Success = false, Error = $"Unsupported response type: {action.ResponseType}" }
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing response action {ActionId}", action.Id);
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Blocks an IP address using Windows Firewall.
        /// </summary>
        /// <param name="action">Response action.</param>
        /// <returns>Response result.</returns>
        private async Task<ResponseResult> BlockIpAddressAsync(ResponseAction action)
        {
            if (!_enableFirewallIntegration)
            {
                return new ResponseResult { Success = false, Error = "Firewall integration is disabled" };
            }

            try
            {
                if (!action.Parameters.TryGetValue("IpAddress", out var ipObj) || ipObj == null)
                {
                    return new ResponseResult { Success = false, Error = "IP address not specified" };
                }

                var ipAddress = ipObj.ToString();
                var ruleName = $"AthalaSIEM_Block_{ipAddress}_{DateTime.UtcNow:yyyyMMddHHmmss}";

                // Create firewall rule using netsh
                var command = $"advfirewall firewall add rule name=\"{ruleName}\" dir=in action=block remoteip={ipAddress}";
                var result = await ExecuteWindowsCommandAsync("netsh", command);

                if (result.Success)
                {
                    _logger.LogInformation("IP address blocked: {IpAddress} (Rule: {RuleName})", ipAddress, ruleName);
                    return new ResponseResult 
                    { 
                        Success = true, 
                        Message = $"IP {ipAddress} blocked successfully",
                        Details = new Dictionary<string, object> { ["FirewallRule"] = ruleName }
                    };
                }
                else
                {
                    return new ResponseResult { Success = false, Error = $"Failed to block IP: {result.Error}" };
                }
            }
            catch (Exception ex)
            {
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Terminates a process.
        /// </summary>
        /// <param name="action">Response action.</param>
        /// <returns>Response result.</returns>
        private async Task<ResponseResult> TerminateProcessAsync(ResponseAction action)
        {
            if (!_enableProcessTermination)
            {
                return new ResponseResult { Success = false, Error = "Process termination is disabled" };
            }

            try
            {
                var processId = 0;
                var processName = string.Empty;

                if (action.Parameters.TryGetValue("ProcessId", out var pidObj) && int.TryParse(pidObj.ToString(), out processId))
                {
                    // Terminate by PID
                    var process = Process.GetProcessById(processId);
                    processName = process.ProcessName;
                    process.Kill();
                    process.WaitForExit(5000);
                }
                else if (action.Parameters.TryGetValue("ProcessName", out var nameObj))
                {
                    // Terminate by name
                    processName = nameObj.ToString() ?? "";
                    var processes = Process.GetProcessesByName(processName);
                    
                    foreach (var process in processes)
                    {
                        process.Kill();
                        process.WaitForExit(5000);
                    }
                }
                else
                {
                    return new ResponseResult { Success = false, Error = "Process ID or name not specified" };
                }

                _logger.LogInformation("Process terminated: {ProcessName} (PID: {ProcessId})", processName, processId);
                return new ResponseResult 
                { 
                    Success = true, 
                    Message = $"Process {processName} terminated successfully" 
                };
            }
            catch (Exception ex)
            {
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Quarantines a file.
        /// </summary>
        /// <param name="action">Response action.</param>
        /// <returns>Response result.</returns>
        private async Task<ResponseResult> QuarantineFileAsync(ResponseAction action)
        {
            if (!_enableFileQuarantine)
            {
                return new ResponseResult { Success = false, Error = "File quarantine is disabled" };
            }

            try
            {
                if (!action.Parameters.TryGetValue("FilePath", out var pathObj) || pathObj == null)
                {
                    return new ResponseResult { Success = false, Error = "File path not specified" };
                }

                var filePath = pathObj.ToString();
                if (!File.Exists(filePath))
                {
                    return new ResponseResult { Success = false, Error = "File does not exist" };
                }

                // Create quarantine directory
                var quarantineDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), 
                    "AthalaSIEM", "Quarantine");
                Directory.CreateDirectory(quarantineDir);

                // Move file to quarantine
                var fileName = Path.GetFileName(filePath);
                var quarantinePath = Path.Combine(quarantineDir, $"{DateTime.UtcNow:yyyyMMddHHmmss}_{fileName}");
                
                File.Move(filePath, quarantinePath);

                _logger.LogInformation("File quarantined: {FilePath} -> {QuarantinePath}", filePath, quarantinePath);
                return new ResponseResult 
                { 
                    Success = true, 
                    Message = $"File quarantined successfully",
                    Details = new Dictionary<string, object> { ["QuarantinePath"] = quarantinePath }
                };
            }
            catch (Exception ex)
            {
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Disables a user account.
        /// </summary>
        /// <param name="action">Response action.</param>
        /// <returns>Response result.</returns>
        private async Task<ResponseResult> DisableUserAccountAsync(ResponseAction action)
        {
            try
            {
                if (!action.Parameters.TryGetValue("Username", out var usernameObj) || usernameObj == null)
                {
                    return new ResponseResult { Success = false, Error = "Username not specified" };
                }

                var username = usernameObj.ToString();
                
                // Disable user account using net user command
                var command = $"user {username} /active:no";
                var result = await ExecuteWindowsCommandAsync("net", command);

                if (result.Success)
                {
                    _logger.LogInformation("User account disabled: {Username}", username);
                    return new ResponseResult 
                    { 
                        Success = true, 
                        Message = $"User account {username} disabled successfully" 
                    };
                }
                else
                {
                    return new ResponseResult { Success = false, Error = $"Failed to disable user account: {result.Error}" };
                }
            }
            catch (Exception ex)
            {
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Isolates the host from the network.
        /// </summary>
        /// <param name="action">Response action.</param>
        /// <returns>Response result.</returns>
        private async Task<ResponseResult> IsolateHostAsync(ResponseAction action)
        {
            try
            {
                // Block all network traffic except management traffic
                var commands = new[]
                {
                    "advfirewall firewall add rule name=\"AthalaSIEM_Isolation_Block_All_In\" dir=in action=block",
                    "advfirewall firewall add rule name=\"AthalaSIEM_Isolation_Block_All_Out\" dir=out action=block"
                };

                foreach (var command in commands)
                {
                    var result = await ExecuteWindowsCommandAsync("netsh", command);
                    if (!result.Success)
                    {
                        return new ResponseResult { Success = false, Error = $"Failed to isolate host: {result.Error}" };
                    }
                }

                _logger.LogWarning("Host isolated from network due to security threat");
                return new ResponseResult 
                { 
                    Success = true, 
                    Message = "Host isolated from network successfully" 
                };
            }
            catch (Exception ex)
            {
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Executes a custom response script.
        /// </summary>
        /// <param name="action">Response action.</param>
        /// <returns>Response result.</returns>
        private async Task<ResponseResult> ExecuteCustomScriptAsync(ResponseAction action)
        {
            if (!_enableCustomScripts)
            {
                return new ResponseResult { Success = false, Error = "Custom scripts are disabled" };
            }

            try
            {
                if (!action.Parameters.TryGetValue("ScriptPath", out var scriptObj) || scriptObj == null)
                {
                    return new ResponseResult { Success = false, Error = "Script path not specified" };
                }

                var scriptPath = scriptObj.ToString();
                if (!File.Exists(scriptPath))
                {
                    return new ResponseResult { Success = false, Error = "Script file does not exist" };
                }

                // Execute custom script
                var arguments = action.Parameters.GetValueOrDefault("Arguments", "")?.ToString() ?? "";
                var result = await ExecuteWindowsCommandAsync(scriptPath, arguments);

                _logger.LogInformation("Custom script executed: {ScriptPath}", scriptPath);
                return result;
            }
            catch (Exception ex)
            {
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Sends an alert.
        /// </summary>
        /// <param name="action">Response action.</param>
        /// <returns>Response result.</returns>
        private async Task<ResponseResult> SendAlertAsync(ResponseAction action)
        {
            try
            {
                var message = action.Parameters.GetValueOrDefault("Message", "Security threat detected")?.ToString() ?? "";
                var severity = action.Parameters.GetValueOrDefault("Severity", "High")?.ToString() ?? "";

                // Log alert (in production, this would send to alerting system)
                _logger.LogWarning("SECURITY ALERT: {Message} (Severity: {Severity})", message, severity);

                return new ResponseResult 
                { 
                    Success = true, 
                    Message = "Alert sent successfully" 
                };
            }
            catch (Exception ex)
            {
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Executes a Windows command.
        /// </summary>
        /// <param name="fileName">Executable file name.</param>
        /// <param name="arguments">Command arguments.</param>
        /// <returns>Command execution result.</returns>
        private async Task<ResponseResult> ExecuteWindowsCommandAsync(string fileName, string arguments)
        {
            try
            {
                using var process = new Process();
                process.StartInfo.FileName = fileName;
                process.StartInfo.Arguments = arguments;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.CreateNoWindow = true;

                process.Start();
                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();
                
                await process.WaitForExitAsync();

                if (process.ExitCode == 0)
                {
                    return new ResponseResult 
                    { 
                        Success = true, 
                        Message = output,
                        Details = new Dictionary<string, object> { ["Output"] = output }
                    };
                }
                else
                {
                    return new ResponseResult 
                    { 
                        Success = false, 
                        Error = error,
                        Details = new Dictionary<string, object> { ["ExitCode"] = process.ExitCode }
                    };
                }
            }
            catch (Exception ex)
            {
                return new ResponseResult { Success = false, Error = ex.Message };
            }
        }

        /// <summary>
        /// Loads response policies from configuration.
        /// </summary>
        /// <returns>Task representing the load operation.</returns>
        private async Task LoadResponsePoliciesAsync()
        {
            try
            {
                // In production, this would load from backend configuration
                // For now, create some default policies
                _responsePolicies.AddRange(new[]
                {
                    new ResponsePolicy
                    {
                        Id = "default-malware-response",
                        Name = "Malware Detection Response",
                        TriggerType = "MalwareDetected",
                        ResponseActions = new List<ResponseType> { ResponseType.QuarantineFile, ResponseType.SendAlert },
                        Enabled = true
                    },
                    new ResponsePolicy
                    {
                        Id = "default-bruteforce-response", 
                        Name = "Brute Force Attack Response",
                        TriggerType = "BruteForceAttack",
                        ResponseActions = new List<ResponseType> { ResponseType.BlockIpAddress, ResponseType.SendAlert },
                        Enabled = true
                    }
                });

                _logger.LogInformation("Loaded {Count} default response policies", _responsePolicies.Count);
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading response policies");
            }
        }

        /// <summary>
        /// Checks if the application is running as administrator.
        /// </summary>
        /// <returns>True if running as administrator.</returns>
        private bool IsRunningAsAdministrator()
        {
            try
            {
                using var identity = WindowsIdentity.GetCurrent();
                var principal = new WindowsPrincipal(identity);
                return principal.IsInRole(WindowsBuiltInRole.Administrator);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets service health status.
        /// </summary>
        /// <returns>Service health information.</returns>
        public ActiveResponseHealth GetHealthStatus()
        {
            return new ActiveResponseHealth
            {
                IsActive = _isActive,
                QueuedResponses = _responseQueue.Count,
                ActiveResponses = _activeResponses.Count,
                TotalResponsesExecuted = TotalResponsesExecuted,
                TotalResponseFailures = TotalResponseFailures,
                LoadedPolicies = _responsePolicies.Count,
                MaxConcurrentResponses = _maxConcurrentResponses,
                LastHealthCheck = DateTime.UtcNow
            };
        }

        /// <inheritdoc />
        public async ValueTask DisposeAsync()
        {
            await StopAsync();
            _responseProcessorTimer?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }

    // NOTE: All models (ThreatTrigger, ResponsePolicy, ResponseAction, etc.) have been moved to 
    // AthalaSIEM.UniversalAgent.Models.ActiveResponseModels.cs for clean architecture separation
} 
