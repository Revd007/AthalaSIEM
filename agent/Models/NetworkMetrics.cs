using System.Collections.Generic;

namespace AthalaSIEM.Agent.Models
{
    public class NetworkMetrics
    {
        public List<NetworkInterfaceMeasurement> Interfaces { get; set; } = new List<NetworkInterfaceMeasurement>();
        public long TotalBytesReceived { get; set; }
        public long TotalBytesSent { get; set; }
    }
} 