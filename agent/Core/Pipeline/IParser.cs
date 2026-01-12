namespace AthalaSIEM.Agent.Core.Pipeline;

public interface IParser
{
    string Name { get; }
    bool CanParse(IRawEvent rawEvent);
    Task<IParsedEvent> ParseAsync(IRawEvent rawEvent, CancellationToken cancellationToken);
}
