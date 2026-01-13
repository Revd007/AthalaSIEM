using System.Collections.Generic;

namespace Backend.DTOs
{
    public class EventDistributionDto
    {
        public string Name { get; set; } = string.Empty;
        public int Value { get; set; }
    }

    public class DeviceAnalyticsDto
    {
        public List<DeviceTypeDto> DeviceData { get; set; } = new();
        public List<SeverityDistributionDto> SeverityData { get; set; } = new();
    }

    public class DeviceTypeDto
    {
        public string Name { get; set; } = string.Empty;
        public int Value { get; set; }
        public string Type { get; set; } = string.Empty;
    }

    public class SeverityDistributionDto
    {
        public string Name { get; set; } = string.Empty;
        public int Value { get; set; }
        public string Color { get; set; } = string.Empty;
    }

    public class SecurityMetricsDto
    {
        public List<MonthlyMetricDto> MonthlyData { get; set; } = new();
        public List<SecurityKpiDto> Kpis { get; set; } = new();
    }

    public class MonthlyMetricDto
    {
        public string Month { get; set; } = string.Empty;
        public int Incidents { get; set; }
        public int Resolved { get; set; }
        public double Mttr { get; set; }
    }

    public class SecurityKpiDto
    {
        public string Title { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string Change { get; set; } = string.Empty;
        public string Trend { get; set; } = string.Empty;
    }

    public class AIAnalyticsDto
    {
        public List<AnomalyDataPointDto> AnomalyData { get; set; } = new();
        public List<ThreatDistributionDto> ThreatDistribution { get; set; } = new();
    }

    public class AnomalyDataPointDto
    {
        public string Timestamp { get; set; } = string.Empty;
        public int Baseline { get; set; }
        public int Actual { get; set; }
        public int Predicted { get; set; }
    }

    public class ThreatDistributionDto
    {
        public string Name { get; set; } = string.Empty;
        public int Value { get; set; }
        public string Color { get; set; } = string.Empty;
    }

    public class BehavioralAnalyticsDto
    {
        public List<BehaviorDataPointDto> BehaviorData { get; set; } = new();
        public List<BehavioralAnomalyDto> Anomalies { get; set; } = new();
    }

    public class BehaviorDataPointDto
    {
        public string Time { get; set; } = string.Empty;
        public int NormalScore { get; set; }
        public int UserScore { get; set; }
    }

    public class BehavioralAnomalyDto
    {
        public int Id { get; set; }
        public string User { get; set; } = string.Empty;
        public string Activity { get; set; } = string.Empty;
        public int RiskScore { get; set; }
        public string Time { get; set; } = string.Empty;
    }

    public class PredictiveAnalyticsDto
    {
        public List<PredictionDataPointDto> Predictions { get; set; } = new();
        public List<RiskFactorDto> RiskFactors { get; set; } = new();
    }

    public class PredictionDataPointDto
    {
        public string Time { get; set; } = string.Empty;
        public int Actual { get; set; }
        public int Predicted { get; set; }
    }

    public class RiskFactorDto
    {
        public string Title { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string Impact { get; set; } = string.Empty;
        public string Recommendation { get; set; } = string.Empty;
    }
}
