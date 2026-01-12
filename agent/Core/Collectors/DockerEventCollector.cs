using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Collectors;

public class DockerEventCollector : ICollector
{
    private readonly ILogger<DockerEventCollector> _logger;
    private readonly string _dockerHost;
    private readonly bool _enabled;
    private readonly HttpClient _httpClient;
    private Task? _pollingTask;
    private readonly CancellationTokenSource _cancellationTokenSource = new();

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

        _pollingTask = Task.Run(() => PollDockerEventsAsync(_cancellationTokenSource.Token), cancellationToken);
        _logger.LogInformation("Started Docker event collector");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _cancellationTokenSource.Cancel();
        _httpClient.Dispose();
        return Task.CompletedTask;
    }

    private async Task PollDockerEventsAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                var url = _dockerHost.Replace("unix://", "http://localhost") + "/events";
                var response = await _httpClient.GetStreamAsync(url, cancellationToken);
                using var reader = new StreamReader(response);

                string? line;
                while (!cancellationToken.IsCancellationRequested && (line = await reader.ReadLineAsync()) != null)
                {
                    try
                    {
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
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing Docker event");
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error polling Docker events");
                await Task.Delay(5000, cancellationToken);
            }
        }
    }
}
