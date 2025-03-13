using System;

namespace AthalaSIEM.Agent.Models
{
    public class CollectorStatusInfo
    {
        public string CollectorType { get; set; } = string.Empty;
        public bool IsRunning { get; set; }
        public DateTime LastCollectionTime { get; set; }
        public int LogsCollected { get; set; }
        public string LastError { get; set; } = string.Empty;
    }
} 