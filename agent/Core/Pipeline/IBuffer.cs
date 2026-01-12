using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Core.Pipeline;

public interface IBuffer
{
    int Count { get; }
    long SizeBytes { get; }
    Task<bool> AddAsync(INormalizedEvent normalizedEvent, CancellationToken cancellationToken);
    Task<IEnumerable<INormalizedEvent>> DrainAsync(int maxCount, CancellationToken cancellationToken);
    Task ClearAsync(CancellationToken cancellationToken);
}
