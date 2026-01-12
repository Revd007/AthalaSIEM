using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Core.Pipeline;

public interface IExporter
{
    string Name { get; }
    bool IsEnabled { get; }
    Task<bool> ExportAsync(IEnumerable<INormalizedEvent> events, CancellationToken cancellationToken);
    Task<bool> ExportAsync(INormalizedEvent normalizedEvent, CancellationToken cancellationToken);
}
