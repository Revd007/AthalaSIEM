using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;
using System.Collections.Generic;

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
    private readonly CancellationTokenSource _cts = new();

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
            _udpTask = Task.Run(() => ListenUdpAsync(_cts.Token));

            _tcpListener = new TcpListener(IPAddress.Any, _port);
            _tcpListener.Start();
            _tcpTask = Task.Run(() => ListenTcpAsync(_cts.Token));

            _logger.LogInformation("Started Syslog collector on port {Port}", _port);
        }
        catch (SocketException ex) when (ex.SocketErrorCode == SocketError.AddressAlreadyInUse)
        {
            _logger.LogWarning("Syslog port {Port} already in use. Syslog collector disabled.", _port);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start Syslog collector on port {Port}", _port);
        }

        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _cts.Cancel();

        // Close sockets first - this unblocks any pending ReceiveAsync/AcceptTcpClientAsync
        _udpListener?.Close();
        _tcpListener?.Stop();

        // Wait for tasks to complete
        var tasks = new List<Task>();
        if (_udpTask != null) tasks.Add(_udpTask);
        if (_tcpTask != null) tasks.Add(_tcpTask);

        if (tasks.Count > 0)
        {
            await Task.WhenAny(Task.WhenAll(tasks), Task.Delay(5000, CancellationToken.None));
        }
    }

    private async Task ListenUdpAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                // .NET 8 overload with CancellationToken
                var result = await _udpListener!.ReceiveAsync(ct);

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
            catch (OperationCanceledException)
            {
                break;
            }
            catch (ObjectDisposedException)
            {
                // Socket closed during shutdown
                break;
            }
            catch (SocketException ex) when (ex.SocketErrorCode == SocketError.OperationAborted
                                           || ex.SocketErrorCode == SocketError.Interrupted)
            {
                // SocketError 995 (OperationAborted) - socket was closed during pending I/O. Normal on shutdown.
                break;
            }
            catch (Exception ex) when (!ct.IsCancellationRequested)
            {
                _logger.LogError(ex, "Error receiving UDP syslog message");
                await SafeDelay(1000, ct);
            }
        }

        _logger.LogDebug("UDP syslog listener stopped");
    }

    private async Task ListenTcpAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                // .NET 8 overload with CancellationToken
                var client = await _tcpListener!.AcceptTcpClientAsync(ct);
                _ = Task.Run(() => HandleTcpClientAsync(client, ct));
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (SocketException ex) when (ex.SocketErrorCode == SocketError.OperationAborted
                                           || ex.SocketErrorCode == SocketError.Interrupted)
            {
                break;
            }
            catch (Exception ex) when (!ct.IsCancellationRequested)
            {
                _logger.LogError(ex, "Error accepting TCP syslog connection");
                await SafeDelay(1000, ct);
            }
        }

        _logger.LogDebug("TCP syslog listener stopped");
    }

    private async Task HandleTcpClientAsync(TcpClient client, CancellationToken ct)
    {
        var remoteEp = string.Empty;
        try
        {
            remoteEp = client.Client.RemoteEndPoint?.ToString() ?? string.Empty;
            using var stream = client.GetStream();
            using var reader = new StreamReader(stream);

            while (!ct.IsCancellationRequested && client.Connected)
            {
                var message = await reader.ReadLineAsync(ct);
                if (string.IsNullOrEmpty(message))
                    break;

                var rawEvent = new RawEvent
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    CollectorName = Name,
                    SourceType = SourceType,
                    RawData = Encoding.UTF8.GetBytes(message),
                    Metadata = new Dictionary<string, string>
                    {
                        ["remote_endpoint"] = remoteEp,
                        ["protocol"] = "TCP"
                    }
                };

                EventCollected?.Invoke(this, rawEvent);
            }
        }
        catch (OperationCanceledException)
        {
            // Normal shutdown
        }
        catch (Exception ex) when (!ct.IsCancellationRequested)
        {
            _logger.LogError(ex, "Error handling TCP syslog client from {Endpoint}", remoteEp);
        }
        finally
        {
            client.Close();
        }
    }

    private static async Task SafeDelay(int ms, CancellationToken ct)
    {
        try { await Task.Delay(ms, ct); }
        catch (OperationCanceledException) { }
    }
}
