using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Collectors;

public class SyslogCollector : ICollector
{
    private readonly ILogger<SyslogCollector> _logger;
    private readonly int _port;
    private readonly bool _enabled;
    private UdpClient? _udpListener;
    private TcpListener? _tcpListener;
    private Task? _udpTask;
    private Task? _tcpTask;
    private readonly CancellationTokenSource _cancellationTokenSource = new();

    public SyslogCollector(
        ILogger<SyslogCollector> logger,
        int port = 514,
        bool enabled = true)
    {
        _logger = logger;
        _port = port;
        _enabled = enabled;
    }

    public string Name => "SyslogCollector";
    public string SourceType => "Syslog";
    public bool IsEnabled => _enabled;

    public event EventHandler<IRawEvent>? EventCollected;

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (!IsEnabled)
            return Task.CompletedTask;

        try
        {
            _udpListener = new UdpClient(_port);
            _udpTask = Task.Run(() => ListenUdpAsync(_cancellationTokenSource.Token), cancellationToken);

            _tcpListener = new TcpListener(IPAddress.Any, _port);
            _tcpListener.Start();
            _tcpTask = Task.Run(() => ListenTcpAsync(_cancellationTokenSource.Token), cancellationToken);

            _logger.LogInformation("Started Syslog collector on port {Port}", _port);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start Syslog collector");
        }

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _cancellationTokenSource.Cancel();
        _udpListener?.Close();
        _tcpListener?.Stop();
        return Task.CompletedTask;
    }

    private async Task ListenUdpAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                var result = await _udpListener!.ReceiveAsync();
                var message = System.Text.Encoding.UTF8.GetString(result.Buffer);

                var rawEvent = new RawEvent
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    CollectorName = Name,
                    SourceType = SourceType,
                    RawData = result.Buffer,
                    Metadata = new Dictionary<string, string>
                    {
                        ["remote_endpoint"] = result.RemoteEndPoint?.ToString() ?? string.Empty,
                        ["protocol"] = "UDP"
                    }
                };

                EventCollected?.Invoke(this, rawEvent);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error receiving UDP syslog message");
                await Task.Delay(1000, cancellationToken);
            }
        }
    }

    private async Task ListenTcpAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                var client = await _tcpListener!.AcceptTcpClientAsync();
                _ = Task.Run(() => HandleTcpClientAsync(client, cancellationToken), cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error accepting TCP syslog connection");
                await Task.Delay(1000, cancellationToken);
            }
        }
    }

    private async Task HandleTcpClientAsync(TcpClient client, CancellationToken cancellationToken)
    {
        try
        {
            using var stream = client.GetStream();
            using var reader = new StreamReader(stream);

            while (!cancellationToken.IsCancellationRequested && client.Connected)
            {
                var message = await reader.ReadLineAsync();
                if (string.IsNullOrEmpty(message))
                    break;

                var rawEvent = new RawEvent
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    CollectorName = Name,
                    SourceType = SourceType,
                    RawData = System.Text.Encoding.UTF8.GetBytes(message),
                    Metadata = new Dictionary<string, string>
                    {
                        ["remote_endpoint"] = client.Client.RemoteEndPoint?.ToString() ?? string.Empty,
                        ["protocol"] = "TCP"
                    }
                };

                EventCollected?.Invoke(this, rawEvent);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling TCP syslog client");
        }
        finally
        {
            client.Close();
        }
    }
}
