namespace AthalaSIEM.Agent.Core.Pipeline;

public class NormalizedEvent : INormalizedEvent
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public AthalaEcsFields Ecs { get; set; } = new();
    public Dictionary<string, object> RawEvent { get; set; } = new();
    public Dictionary<string, object> Extensions { get; set; } = new();
}
