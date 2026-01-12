using System;
using System.Threading;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Core.Pipeline;

public interface ICollector
{
    string Name { get; }
    string SourceType { get; }
    bool IsEnabled { get; }
    Task StartAsync(CancellationToken cancellationToken);
    Task StopAsync(CancellationToken cancellationToken);
    event EventHandler<IRawEvent>? EventCollected;
}
