using System;
using System.Collections.Generic;

namespace Backend.DTOs
{
    public class AutomatedActionDto
    {
        public string Id { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public string Trigger { get; set; } = string.Empty;
        public string Status { get; set; } = "pending";
        public DateTime Timestamp { get; set; }
        public string Target { get; set; } = string.Empty;
        public string Details { get; set; } = string.Empty;
        public string? Result { get; set; }
    }

    public class AutomatedRuleDto
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string Status { get; set; } = "active";
        public int Triggers { get; set; }
        public DateTime? LastTriggered { get; set; }
        public string ActionType { get; set; } = string.Empty;
        public object? Conditions { get; set; }
    }

    public class CreateAutomatedRuleRequest
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string ActionType { get; set; } = string.Empty;
        public object? Conditions { get; set; }
    }

    public class UpdateStatusRequest
    {
        public string Status { get; set; } = string.Empty;
    }

    public class AutomatedResponseStats
    {
        public int ActionsToday { get; set; }
        public double SuccessRate { get; set; }
        public double AverageResponseTime { get; set; }
        public int ActiveRules { get; set; }
        public int TotalActions { get; set; }
    }

    public class ResponseMetricDto
    {
        public string Time { get; set; } = string.Empty;
        public int Actions { get; set; }
        public double ResponseTime { get; set; }
    }
}
