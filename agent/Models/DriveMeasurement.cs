namespace AthalaSIEM.Agent.Models
{
    public class DriveMeasurement
    {
        public required string Name { get; set; }
        public long TotalBytes { get; set; }
        public long AvailableBytes { get; set; }
        public long UsedBytes { get; set; }
        public double UsedPercentage { get; set; }
    }
} 