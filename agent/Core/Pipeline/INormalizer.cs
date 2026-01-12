namespace AthalaSIEM.Agent.Core.Pipeline;

public interface INormalizer
{
    string Name { get; }
    Task<INormalizedEvent> NormalizeAsync(IParsedEvent parsedEvent, CancellationToken cancellationToken);
}
