using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using System.Net.Http;
using System.Diagnostics;
using System.Linq;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Container Collector for Docker and Kubernetes logs
    /// </summary>
    public class ContainerCollector : ILogCollector
    {
        private readonly ILogger<ContainerCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings = new();
        
        // Configuration
        private bool _enableDockerLogs = true;
        private bool _enableKubernetesLogs = true;
        private bool _enableContainerEvents = true;
        private string _dockerSocketPath = "/var/run/docker.sock";
        private string _kubernetesConfigPath = "";
        private string _logDirectory = "/var/log/containers";
        private int _collectionInterval = 30;
        private List<string> _excludeContainers = new();
        private List<string> _includeNamespaces = new();
        
        private CancellationTokenSource? _cancellationTokenSource;

        public event EventHandler<NormalizedLogEntry>? LogCollected;
        public string CollectorType => "Container";
        public CollectorStatus Status => _isRunning ? (_isPaused ? CollectorStatus.Paused : CollectorStatus.Running) : 
                                        (!string.IsNullOrEmpty(_errorMessage) ? CollectorStatus.Error : CollectorStatus.Stopped);
        public string ErrorMessage => _errorMessage;
        public bool IsRunning => _isRunning;
        public bool IsPaused => _isPaused;
        public CollectorSettings Settings => _settings;

        public ContainerCollector(ILogger<ContainerCollector> logger, ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
        }

        public bool Initialize(CollectorSettings settings)
        {
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _logger.LogInformation("Initializing Container Collector");

            try
            {
                ParseSettings();
                _logger.LogInformation("Container Collector initialized - Docker: {Docker}, K8s: {K8s}", 
                    _enableDockerLogs, _enableKubernetesLogs);
                return true;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to initialize Container Collector");
                return false;
            }
        }

        public async Task StartAsync()
        {
            if (_isRunning) return;

            try
            {
                _logger.LogInformation("Starting Container Collector");
                _cancellationTokenSource = new CancellationTokenSource();

                if (_enableDockerLogs)
                {
                    _ = Task.Run(() => CollectDockerLogsAsync(_cancellationTokenSource.Token));
                }

                if (_enableKubernetesLogs)
                {
                    _ = Task.Run(() => CollectKubernetesLogsAsync(_cancellationTokenSource.Token));
                }

                if (_enableContainerEvents)
                {
                    _ = Task.Run(() => CollectContainerEventsAsync(_cancellationTokenSource.Token));
                }

                _isRunning = true;
                _isPaused = false;
                _errorMessage = string.Empty;

                _logger.LogInformation("Container Collector started successfully");
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to start Container Collector");
                throw;
            }
        }

        public async Task StopAsync()
        {
            if (!_isRunning) return;

            try
            {
                _logger.LogInformation("Stopping Container Collector");
                
                _cancellationTokenSource?.Cancel();
                _isRunning = false;
                
                _logger.LogInformation("Container Collector stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping Container Collector");
            }
        }

        public Task PauseAsync()
        {
            _isPaused = true;
            _logger.LogInformation("Container Collector paused");
            return Task.CompletedTask;
        }

        public Task ResumeAsync()
        {
            _isPaused = false;
            _logger.LogInformation("Container Collector resumed");
            return Task.CompletedTask;
        }

        public async Task<int> CollectLogsAsync(CancellationToken cancellationToken)
        {
            if (_isPaused || !_isRunning)
                return 0;

            int collectedCount = 0;

            try
            {
                if (_enableDockerLogs)
                {
                    await CollectDockerContainerLogs();
                    collectedCount++;
                }

                if (_enableKubernetesLogs)
                {
                    await CollectKubernetesPodLogs();
                    collectedCount++;
                }

                if (_enableContainerEvents)
                {
                    await CollectDockerEvents();
                    await CollectKubernetesEvents();
                    collectedCount++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting container logs");
                _errorMessage = ex.Message;
            }

            return collectedCount;
        }

        private void ParseSettings()
        {
            if (_settings.Properties.ContainsKey("EnableDockerLogs"))
            {
                bool.TryParse(_settings.Properties["EnableDockerLogs"], out _enableDockerLogs);
            }

            if (_settings.Properties.ContainsKey("EnableKubernetesLogs"))
            {
                bool.TryParse(_settings.Properties["EnableKubernetesLogs"], out _enableKubernetesLogs);
            }

            if (_settings.Properties.ContainsKey("EnableContainerEvents"))
            {
                bool.TryParse(_settings.Properties["EnableContainerEvents"], out _enableContainerEvents);
            }

            if (_settings.Properties.ContainsKey("DockerSocketPath"))
            {
                _dockerSocketPath = _settings.Properties["DockerSocketPath"];
            }

            if (_settings.Properties.ContainsKey("LogDirectory"))
            {
                _logDirectory = _settings.Properties["LogDirectory"];
            }

            if (_settings.Properties.ContainsKey("CollectionInterval"))
            {
                int.TryParse(_settings.Properties["CollectionInterval"], out _collectionInterval);
            }

            if (_settings.Properties.ContainsKey("ExcludeContainers"))
            {
                _excludeContainers = new List<string>(_settings.Properties["ExcludeContainers"].Split(','));
            }

            if (_settings.Properties.ContainsKey("IncludeNamespaces"))
            {
                _includeNamespaces = new List<string>(_settings.Properties["IncludeNamespaces"].Split(','));
            }
        }

        private async Task CollectDockerLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting Docker logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectDockerContainerLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting Docker logs");
                    await Task.Delay(TimeSpan.FromSeconds(30), cancellationToken);
                }
            }
        }

        private async Task CollectKubernetesLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting Kubernetes logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectKubernetesPodLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting Kubernetes logs");
                    await Task.Delay(TimeSpan.FromSeconds(30), cancellationToken);
                }
            }
        }

        private async Task CollectContainerEventsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting Container events collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectDockerEvents();
                        await CollectKubernetesEvents();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting container events");
                    await Task.Delay(TimeSpan.FromSeconds(30), cancellationToken);
                }
            }
        }

        private async Task CollectDockerContainerLogs()
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "docker",
                        Arguments = "ps --format \"{{.Names}}\"",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var containers = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                foreach (var container in containers.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                {
                    if (_excludeContainers.Contains(container.Trim())) continue;

                    await CollectContainerLogs(container.Trim());
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting Docker containers");
            }
        }

        private async Task CollectContainerLogs(string containerName)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "docker",
                        Arguments = $"logs --since 30s --timestamps {containerName}",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                
                // Read stdout
                var stdout = process.StandardOutput.ReadToEnd();
                var stderr = process.StandardError.ReadToEnd();
                
                await process.WaitForExitAsync();

                // Process stdout logs
                await ProcessContainerLogOutput(containerName, stdout, "stdout");
                
                // Process stderr logs
                if (!string.IsNullOrEmpty(stderr))
                {
                    await ProcessContainerLogOutput(containerName, stderr, "stderr");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting logs from container {Container}", containerName);
            }
        }

        private async Task ProcessContainerLogOutput(string containerName, string logOutput, string stream)
        {
            if (string.IsNullOrEmpty(logOutput)) return;

            var lines = logOutput.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            
            foreach (var line in lines)
            {
                try
                {
                    var logEntry = ParseContainerLogLine(line, containerName, stream);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing log line from container {Container}", containerName);
                }
            }
        }

        private async Task CollectKubernetesPodLogs()
        {
            try
            {
                if (!Directory.Exists(_logDirectory))
                {
                    _logger.LogWarning("Kubernetes log directory not found: {Directory}", _logDirectory);
                    return;
                }

                var logFiles = Directory.GetFiles(_logDirectory, "*.log");
                
                foreach (var logFile in logFiles)
                {
                    await ProcessKubernetesLogFile(logFile);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting Kubernetes pod logs");
            }
        }

        private async Task ProcessKubernetesLogFile(string logFile)
        {
            try
            {
                var fileName = Path.GetFileName(logFile);
                var parts = fileName.Replace(".log", "").Split('_');
                
                if (parts.Length < 3) return;

                var podName = parts[0];
                var namespace_ = parts[1];
                var containerName = parts[2];

                // Check namespace filter
                if (_includeNamespaces.Count > 0 && !_includeNamespaces.Contains(namespace_))
                {
                    return;
                }

                var lines = await File.ReadAllLinesAsync(logFile);
                var recentLines = lines.Skip(Math.Max(0, lines.Length - 100)); // Get last 100 lines

                foreach (var line in recentLines)
                {
                    var logEntry = ParseKubernetesLogLine(line, podName, namespace_, containerName);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing Kubernetes log file {File}", logFile);
            }
        }

        private async Task CollectDockerEvents()
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "docker",
                        Arguments = "events --since 30s --format \"{{.Time}} {{.Type}} {{.Action}} {{.Actor.Attributes.name}}\"",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var events = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                foreach (var eventLine in events.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                {
                    var logEntry = ParseDockerEvent(eventLine);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting Docker events");
            }
        }

        private async Task CollectKubernetesEvents()
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "kubectl",
                        Arguments = "get events --all-namespaces -o json",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var eventsJson = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                if (!string.IsNullOrEmpty(eventsJson))
                {
                    ProcessKubernetesEvents(eventsJson);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting Kubernetes events");
            }
        }

        private NormalizedLogEntry? ParseContainerLogLine(string logLine, string containerName, string stream)
        {
            try
            {
                // Docker log format: 2024-01-01T12:00:00.000000000Z message
                var timestampMatch = Regex.Match(logLine, @"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(.*)");
                
                DateTime timestamp = DateTime.UtcNow;
                string message = logLine;

                if (timestampMatch.Success)
                {
                    DateTime.TryParse(timestampMatch.Groups[1].Value, out timestamp);
                    message = timestampMatch.Groups[2].Value;
                }

                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = timestamp,
                    Level = stream == "stderr" ? "Error" : "Information",
                    Source = $"Docker/{containerName}",
                    Category = "Container",
                    EventId = "DOCKER_LOG",
                    Message = message,
                    Details = JsonSerializer.Serialize(new
                    {
                        container_name = containerName,
                        stream = stream,
                        raw_message = logLine
                    }),
                    Tags = new List<string> { "docker", "container", containerName, stream },
                    Severity = stream == "stderr" ? "Medium" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing container log line");
                return null;
            }
        }

        private NormalizedLogEntry? ParseKubernetesLogLine(string logLine, string podName, string namespace_, string containerName)
        {
            try
            {
                // Kubernetes log format: timestamp stream message
                var parts = logLine.Split(' ', 3);
                if (parts.Length < 3) return null;

                DateTime.TryParse(parts[0], out var timestamp);
                var stream = parts[1];
                var message = parts[2];

                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = timestamp != DateTime.MinValue ? timestamp : DateTime.UtcNow,
                    Level = stream == "stderr" ? "Error" : "Information",
                    Source = $"Kubernetes/{namespace_}/{podName}",
                    Category = "Pod",
                    EventId = "K8S_LOG",
                    Message = message,
                    Details = JsonSerializer.Serialize(new
                    {
                        pod_name = podName,
                        namespace_ = namespace_,
                        container_name = containerName,
                        stream = stream,
                        raw_message = logLine
                    }),
                    Tags = new List<string> { "kubernetes", "pod", namespace_, podName, containerName },
                    Severity = stream == "stderr" ? "Medium" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing Kubernetes log line");
                return null;
            }
        }

        private NormalizedLogEntry? ParseDockerEvent(string eventLine)
        {
            try
            {
                var parts = eventLine.Split(' ', 4);
                if (parts.Length < 4) return null;

                DateTime.TryParse(parts[0], out var timestamp);
                var type = parts[1];
                var action = parts[2];
                var actor = parts[3];

                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = timestamp != DateTime.MinValue ? timestamp : DateTime.UtcNow,
                    Level = "Information",
                    Source = "Docker/Events",
                    Category = "ContainerEvent",
                    EventId = $"DOCKER_{action.ToUpper()}",
                    Message = $"Container {action}: {actor}",
                    Details = JsonSerializer.Serialize(new
                    {
                        event_type = type,
                        action = action,
                        actor = actor,
                        raw_event = eventLine
                    }),
                    Tags = new List<string> { "docker", "event", type, action },
                    Severity = action.Contains("die") || action.Contains("kill") ? "High" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing Docker event");
                return null;
            }
        }

        private void ProcessKubernetesEvents(string eventsJson)
        {
            try
            {
                using var document = JsonDocument.Parse(eventsJson);
                var events = document.RootElement.GetProperty("items");

                foreach (var eventElement in events.EnumerateArray())
                {
                    var logEntry = ParseKubernetesEvent(eventElement);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing Kubernetes events");
            }
        }

        private NormalizedLogEntry? ParseKubernetesEvent(JsonElement eventElement)
        {
            try
            {
                var metadata = eventElement.GetProperty("metadata");
                var involvedObject = eventElement.GetProperty("involvedObject");
                
                var name = metadata.TryGetProperty("name", out var nameElement) ? nameElement.GetString() : "";
                var namespace_ = metadata.TryGetProperty("namespace", out var nsElement) ? nsElement.GetString() : "default";
                var reason = eventElement.TryGetProperty("reason", out var reasonElement) ? reasonElement.GetString() : "";
                var message = eventElement.TryGetProperty("message", out var msgElement) ? msgElement.GetString() : "";
                var type = eventElement.TryGetProperty("type", out var typeElement) ? typeElement.GetString() : "";
                var objectKind = involvedObject.TryGetProperty("kind", out var kindElement) ? kindElement.GetString() : "";
                var objectName = involvedObject.TryGetProperty("name", out var objNameElement) ? objNameElement.GetString() : "";

                DateTime timestamp = DateTime.UtcNow;
                if (eventElement.TryGetProperty("firstTimestamp", out var tsElement))
                {
                    DateTime.TryParse(tsElement.GetString(), out timestamp);
                }

                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = timestamp,
                    Level = type == "Warning" ? "Warning" : "Information",
                    Source = $"Kubernetes/{namespace_}/Events",
                    Category = "KubernetesEvent",
                    EventId = $"K8S_{reason?.ToUpper()}",
                    Message = $"{objectKind} {objectName}: {message}",
                    Details = JsonSerializer.Serialize(new
                    {
                        event_name = name,
                        namespace_ = namespace_,
                        reason = reason,
                        type = type,
                        object_kind = objectKind,
                        object_name = objectName,
                        raw_event = eventElement.ToString()
                    }),
                    Tags = new List<string> { "kubernetes", "event", namespace_, objectKind?.ToLower() ?? "", reason?.ToLower() ?? "" },
                    Severity = type == "Warning" ? "Medium" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing Kubernetes event");
                return null;
            }
        }

        public void Dispose()
        {
            StopAsync().Wait();
            _cancellationTokenSource?.Dispose();
        }
    }
} 