using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Buffer;

public class DiskBackedBuffer : IBuffer
{
    private readonly ConcurrentQueue<INormalizedEvent> _memoryQueue = new();
    private readonly string _diskBufferPath;
    private readonly ILogger<DiskBackedBuffer> _logger;
    private readonly int _maxMemoryCount;
    private readonly long _maxDiskSizeBytes;
    private long _currentDiskSizeBytes = 0;
    private readonly object _diskLock = new();

    public DiskBackedBuffer(
        ILogger<DiskBackedBuffer> logger,
        string diskBufferPath,
        int maxMemoryCount = 1000,
        long maxDiskSizeBytes = 100 * 1024 * 1024)
    {
        _logger = logger;
        _diskBufferPath = diskBufferPath;
        _maxMemoryCount = maxMemoryCount;
        _maxDiskSizeBytes = maxDiskSizeBytes;

        Directory.CreateDirectory(Path.GetDirectoryName(diskBufferPath)!);
    }

    public int Count => _memoryQueue.Count + GetDiskCount();
    public long SizeBytes => _currentDiskSizeBytes;

    public async Task<bool> AddAsync(INormalizedEvent normalizedEvent, CancellationToken cancellationToken)
    {
        if (_memoryQueue.Count < _maxMemoryCount)
        {
            _memoryQueue.Enqueue(normalizedEvent);
            return true;
        }

        return await WriteToDiskAsync(normalizedEvent, cancellationToken);
    }

    public async Task<IEnumerable<INormalizedEvent>> DrainAsync(int maxCount, CancellationToken cancellationToken)
    {
        var events = new List<INormalizedEvent>();
        var count = 0;

        while (count < maxCount && _memoryQueue.TryDequeue(out var evt))
        {
            events.Add(evt);
            count++;
        }

        if (count < maxCount)
        {
            var diskEvents = await ReadFromDiskAsync(maxCount - count, cancellationToken);
            events.AddRange(diskEvents);
        }

        return events;
    }

    public Task ClearAsync(CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            while (_memoryQueue.TryDequeue(out _)) { }
            
            lock (_diskLock)
            {
                if (Directory.Exists(Path.GetDirectoryName(_diskBufferPath)!))
                {
                    var files = Directory.GetFiles(Path.GetDirectoryName(_diskBufferPath)!, "buffer_*.json");
                    foreach (var file in files)
                    {
                        try
                        {
                            File.Delete(file);
                        }
                        catch { }
                    }
                }
                _currentDiskSizeBytes = 0;
            }
        }, cancellationToken);
    }

    private async Task<bool> WriteToDiskAsync(INormalizedEvent evt, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            lock (_diskLock)
            {
                if (_currentDiskSizeBytes >= _maxDiskSizeBytes)
                {
                    _logger.LogWarning("Disk buffer full, rejecting event {EventId}", evt.Id);
                    return false;
                }

                var filePath = Path.Combine(
                    Path.GetDirectoryName(_diskBufferPath)!,
                    $"buffer_{evt.Id}_{DateTime.UtcNow:yyyyMMddHHmmss}.json");

                try
                {
                    var json = JsonSerializer.Serialize(evt);
                    File.WriteAllText(filePath, json);
                    _currentDiskSizeBytes += new FileInfo(filePath).Length;
                    return true;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to write event to disk buffer");
                    return false;
                }
            }
        }, cancellationToken);
    }

    private async Task<IEnumerable<INormalizedEvent>> ReadFromDiskAsync(int maxCount, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var events = new List<INormalizedEvent>();

            lock (_diskLock)
            {
                var files = Directory.GetFiles(Path.GetDirectoryName(_diskBufferPath)!, "buffer_*.json")
                    .OrderBy(f => File.GetCreationTime(f))
                    .Take(maxCount);

                foreach (var file in files)
                {
                    try
                    {
                        var json = File.ReadAllText(file);
                        var evt = JsonSerializer.Deserialize<NormalizedEvent>(json);
                        if (evt != null)
                        {
                            events.Add(evt);
                            File.Delete(file);
                            _currentDiskSizeBytes -= new FileInfo(file).Length;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to read event from disk buffer: {FilePath}", file);
                    }
                }
            }

            return events;
        }, cancellationToken);
    }

    private int GetDiskCount()
    {
        lock (_diskLock)
        {
            if (!Directory.Exists(Path.GetDirectoryName(_diskBufferPath)!))
                return 0;

            return Directory.GetFiles(Path.GetDirectoryName(_diskBufferPath)!, "buffer_*.json").Length;
        }
    }
}
