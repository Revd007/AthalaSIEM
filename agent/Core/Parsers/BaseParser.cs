namespace AthalaSIEM.Agent.Core.Parsers;

public abstract class BaseParser : IParser
{
    public abstract string Name { get; }
    public abstract bool CanParse(IRawEvent rawEvent);

    public async Task<IParsedEvent> ParseAsync(IRawEvent rawEvent, CancellationToken cancellationToken)
    {
        return await Task.Run(() => ParseInternal(rawEvent), cancellationToken);
    }

    protected abstract IParsedEvent ParseInternal(IRawEvent rawEvent);
}
