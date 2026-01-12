using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;

namespace AthalaSIEM.Agent.Core.Buffer;

public class MemoryBuffer : IBuffer
{
    private readonly ConcurrentQueue<INormalizedEvent> _queue = new();
    private readonly ILogger<MemoryBuffer> _logger;
    private readonly int _maxCount;
    private readonly long _maxSizeBytes;
    private long _currentSizeBytes = 0;

    public MemoryBuffer(
        ILogger<MemoryBuffer> logger,
        int maxCount = 10000,
        long maxSizeBytes = 100 * 1024 * 1024)
    {
        _logger = logger;
        _maxCount = maxCount;
        _maxSizeBytes = maxSizeBytes;
    }

    public int Count => _queue.Count;
    public long SizeBytes => _currentSizeBytes;

    public Task<bool> AddAsync(INormalizedEvent normalizedEvent, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            if (_queue.Count >= _maxCount)
            {
                _logger.LogWarning("Buffer full, rejecting event {EventId}", normalizedEvent.Id);
                return false;
            }

            var estimatedSize = EstimateSize(normalizedEvent);
            if (_currentSizeBytes + estimatedSize > _maxSizeBytes)
            {
                _logger.LogWarning("Buffer size limit reached, rejecting event {EventId}", normalizedEvent.Id);
                return false;
            }

            _queue.Enqueue(normalizedEvent);
            Interlocked.Add(ref _currentSizeBytes, estimatedSize);
            return true;
        }, cancellationToken);
    }

    public Task<IEnumerable<INormalizedEvent>> DrainAsync(int maxCount, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            var events = new List<INormalizedEvent>();
            var count = 0;

            while (count < maxCount && _queue.TryDequeue(out var evt))
            {
                events.Add(evt);
                Interlocked.Add(ref _currentSizeBytes, -EstimateSize(evt));
                count++;
            }

            return (IEnumerable<INormalizedEvent>)events;
        }, cancellationToken);
    }

    public Task ClearAsync(CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            while (_queue.TryDequeue(out _)) { }
            Interlocked.Exchange(ref _currentSizeBytes, 0);
        }, cancellationToken);
    }

    private long EstimateSize(INormalizedEvent evt)
    {
        return 1024;
    }
}
