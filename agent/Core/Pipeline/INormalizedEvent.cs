namespace AthalaSIEM.Agent.Core.Pipeline;

public interface INormalizedEvent
{
    string Id { get; }
    DateTime Timestamp { get; }
    AthalaEcsFields Ecs { get; }
    Dictionary<string, object> RawEvent { get; }
    Dictionary<string, object> Extensions { get; }
}
