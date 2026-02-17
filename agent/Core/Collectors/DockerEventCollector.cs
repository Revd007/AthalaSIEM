using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;
using System.Collections.Generic;

namespace AthalaSIEM.Agent.Core.Collectors;

public class DockerEventCollector : ICollector
{
    private readonly ILogger<DockerEventCollector> _logger;
    private readonly string _dockerHost;
    private readonly bool _enabled;
    private readonly HttpClient _httpClient;
    private Task? _pollingTask;
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    private int _dockerUnavailableLogCount;
    private const int DockerUnavailableLogInterval = 12; // Log at most once every 12 retries (~1 min at 5s delay)

    public DockerEventCollector(
        ILogger<DockerEventCollector> logger,
        string dockerHost = "unix:///var/run/docker.sock",
        bool enabled = true)
    {
        _logger = logger;
        _dockerHost = dockerHost;
        _enabled = enabled;
        _httpClient = new HttpClient();
    }

    public string Name => "DockerEventCollector";
    public string SourceType => "Docker";
    public bool IsEnabled => _enabled;

    public event EventHandler<IRawEvent>? EventCollected;

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (!IsEnabled)
            return Task.CompletedTask;

        // Silent pre-check: verify Docker is reachable before starting the polling loop
        if (!IsDockerAvailable())
        {
            _logger.LogInformation(
                "Docker is not installed or the daemon is not running on this machine. " +
                "DockerEventCollector is disabled. To enable, install Docker and restart the agent.");
            return Task.CompletedTask;
        }

        _pollingTask = Task.Run(() => PollDockerEventsAsync(_cancellationTokenSource.Token), cancellationToken);
        _logger.LogInformation("Started Docker event collector targeting {DockerHost}", _dockerHost);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Checks if Docker is available by looking for the Docker socket (Windows named pipe or Linux socket).
    /// </summary>
    private bool IsDockerAvailable()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                // Windows: Docker Desktop exposes \\.\pipe\docker_engine
                return File.Exists(@"\\.\pipe\docker_engine");
            }
            else
            {
                // Linux/macOS: Docker socket at /var/run/docker.sock
                return File.Exists("/var/run/docker.sock");
            }
        }
        catch
        {
            return false;
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _cancellationTokenSource.Cancel();

        if (_pollingTask != null)
        {
            await Task.WhenAny(_pollingTask, Task.Delay(5000, CancellationToken.None));
        }

        _httpClient.Dispose();
    }

    /// <summary>
    /// Normalizes Docker host URL so HttpClient can use it. unix:// and tcp:// are converted to http(s)://.
    /// </summary>
    private static string NormalizeDockerHostUrl(string dockerHost)
    {
        var s = dockerHost.TrimEnd('/');
        if (s.StartsWith("unix://", StringComparison.OrdinalIgnoreCase))
            return "http://localhost";
        if (s.StartsWith("tcp://", StringComparison.OrdinalIgnoreCase))
            return "http://" + s.Substring(6);
        if (s.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || s.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            return s;
        return "http://" + s;
    }

    private async Task PollDockerEventsAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                var baseUrl = NormalizeDockerHostUrl(_dockerHost);
                var url = baseUrl.TrimEnd('/') + "/events";
                var response = await _httpClient.GetStreamAsync(url, cancellationToken);
                using var reader = new StreamReader(response);

                string? line;
                while (!cancellationToken.IsCancellationRequested)
                {
                    try
                    {
                        line = await reader.ReadLineAsync();
                        if (line == null)
                        {
                            await Task.Delay(100, cancellationToken);
                            continue;
                        }

                        var rawEvent = new RawEvent
                        {
                            Id = Guid.NewGuid().ToString(),
                            Timestamp = DateTime.UtcNow,
                            CollectorName = Name,
                            SourceType = SourceType,
                            RawData = System.Text.Encoding.UTF8.GetBytes(line),
                            Metadata = new Dictionary<string, string>()
                        };

                        EventCollected?.Invoke(this, rawEvent);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing Docker event");
                        await Task.Delay(1000, cancellationToken);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (HttpRequestException ex) when (ex.InnerException is System.Net.Sockets.SocketException)
            {
                _dockerUnavailableLogCount++;
                if (_dockerUnavailableLogCount == 1)
                    _logger.LogInformation("Docker not detected at {DockerHost}; collector will retry when available. To disable, remove DockerEventCollector from pipeline config.", _dockerHost);
                else if (_dockerUnavailableLogCount % DockerUnavailableLogInterval == 0)
                    _logger.LogDebug("Docker still unavailable at {DockerHost}", _dockerHost);
                await Task.Delay(5000, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error polling Docker events");
                await Task.Delay(5000, cancellationToken);
            }
        }
    }
}
