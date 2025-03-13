using System;

namespace AthalaSIEM.Agent.Models
{
    public class NetworkInterfaceMeasurement
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public long Speed { get; set; }
        public long BytesReceived { get; set; }
        public long BytesSent { get; set; }
    }
} 