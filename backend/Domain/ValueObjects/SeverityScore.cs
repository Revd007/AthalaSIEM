namespace Backend.Domain.ValueObjects;

public class SeverityScore
{
    public int BaseSeverity { get; set; }
    public int ThreatIntelligenceScore { get; set; }
    public int CorrelationScore { get; set; }
    public int AnomalyScore { get; set; }
    public int TechniqueScore { get; set; }
    
    public int TotalScore => BaseSeverity + ThreatIntelligenceScore + CorrelationScore + AnomalyScore + TechniqueScore;
    
    public AlertSeverityLevel CalculateSeverity()
    {
        return TotalScore switch
        {
            >= 15 => AlertSeverityLevel.Critical,
            >= 10 => AlertSeverityLevel.High,
            >= 5 => AlertSeverityLevel.Medium,
            >= 2 => AlertSeverityLevel.Low,
            _ => AlertSeverityLevel.Info
        };
    }
}

public enum AlertSeverityLevel
{
    Info = 1,
    Low = 2,
    Medium = 4,
    High = 7,
    Critical = 10
}
