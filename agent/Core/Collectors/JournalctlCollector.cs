using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;
using System.Collections.Generic;

namespace AthalaSIEM.Agent.Core.Collectors;

public class JournalctlCollector : ICollector
{
    private readonly ILogger<JournalctlCollector> _logger;
    private readonly bool _enabled;
    private Process? _journalctlProcess;
    private readonly CancellationTokenSource _cancellationTokenSource = new();

    public JournalctlCollector(
        ILogger<JournalctlCollector> logger,
        bool enabled = true)
    {
        _logger = logger;
        _enabled = enabled;
    }

    public string Name => "JournalctlCollector";
    public string SourceType => "Journalctl";
    public bool IsEnabled => _enabled && OperatingSystem.IsLinux();

    public event EventHandler<IRawEvent>? EventCollected;

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (!IsEnabled)
            return Task.CompletedTask;

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "journalctl",
                Arguments = "--output=json --follow --no-pager",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            _journalctlProcess = Process.Start(startInfo);
            if (_journalctlProcess == null)
            {
                _logger.LogError("Failed to start journalctl process");
                return Task.CompletedTask;
            }

            _ = Task.Run(() => ReadJournalctlOutputAsync(_cancellationTokenSource.Token), cancellationToken);
            _logger.LogInformation("Started Journalctl collector");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start Journalctl collector");
        }

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _cancellationTokenSource.Cancel();
        try
        {
            _journalctlProcess?.Kill();
            _journalctlProcess?.Dispose();
        }
        catch { }
        return Task.CompletedTask;
    }

    private async Task ReadJournalctlOutputAsync(CancellationToken cancellationToken)
    {
        if (_journalctlProcess?.StandardOutput == null)
            return;

        using var reader = _journalctlProcess.StandardOutput;
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
                _logger.LogError(ex, "Error processing journalctl line");
            }
        }
    }
}
