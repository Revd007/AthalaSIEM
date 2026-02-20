using Microsoft.EntityFrameworkCore;
using Backend.Models;
using System.Text.Json;
using System;

namespace Backend.Data
{
    /// <summary>
    /// Application database context
    /// </summary>
    public class ApplicationDbContext : DbContext
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ApplicationDbContext"/> class
        /// </summary>
        /// <param name="options">The database context options</param>
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
            // No additional configuration that requires JWT here
        }

        /// <summary>
        /// Gets or sets the agents
        /// </summary>
        public DbSet<AgentModels> Agents { get; set; } = null!;

        /// <summary>
        /// Gets or sets the agent configurations
        /// </summary>
        public DbSet<AgentConfigModels> AgentConfigs { get; set; } = null!;

        /// <summary>
        /// Gets or sets the log entries
        /// </summary>
        public DbSet<LogEntryModels> LogEntries { get; set; } = null!;

        /// <summary>
        /// Gets or sets the health metrics
        /// </summary>
        public DbSet<HealthMetricModels> HealthMetrics { get; set; } = null!;

        /// <summary>
        /// Gets or sets the alerts
        /// </summary>
        public DbSet<AlertModels> Alerts { get; set; } = null!;

        /// <summary>
        /// Gets or sets the FIM configurations
        /// </summary>
        public DbSet<FIMConfiguration> FIMConfigurations { get; set; } = null!;

        /// <summary>
        /// Gets or sets the FIM templates
        /// </summary>
        public DbSet<FIMTemplate> FIMTemplates { get; set; } = null!;

        /// <summary>
        /// Gets or sets the FIM events
        /// </summary>
        public DbSet<FIMEvent> FIMEvents { get; set; } = null!;

        /// <summary>
        /// Gets or sets the security events
        /// </summary>
        public DbSet<SecurityEventModels> SecurityEvents { get; set; } = null!;

        /// <summary>
        /// Gets or sets the users
        /// </summary>
        public DbSet<UserModels> Users { get; set; } = null!;

        /// <summary>
        /// Gets or sets the roles
        /// </summary>
        public DbSet<RoleModels> Roles { get; set; } = null!;

        /// <summary>
        /// Gets or sets the user roles
        /// </summary>
        public DbSet<UserRoleModels> UserRoles { get; set; } = null!;

        /// <summary>
        /// Gets or sets the user security settings
        /// </summary>
        public DbSet<UserSecurityModels> UserSecurityModels { get; set; } = null!;

        /// <summary>
        /// Gets or sets the dashboards
        /// </summary>
        public DbSet<DashboardModels> Dashboards { get; set; } = null!;

        /// <summary>
        /// Gets or sets the reports
        /// </summary>
        public DbSet<ReportModels> Reports { get; set; } = null!;

        /// <summary>
        /// Gets or sets the alert rules
        /// </summary>
        public DbSet<AlertModels> AlertRules { get; set; } = null!;

        /// <summary>
        /// Gets or sets the agent health reports
        /// </summary>
        public DbSet<AgentHealthReport> AgentHealthReports { get; set; } = null!;

        /// <summary>
        /// Gets or sets the agent heartbeats
        /// </summary>
        public DbSet<AgentHeartbeatModels> AgentHeartbeats { get; set; } = null!;

        /// <summary>
        /// Gets or sets the log entry
        /// </summary>
        public DbSet<LogEntryModels> LogEntry { get; set; } = null!;

        /// <summary>
        /// Gets or sets the alert
        /// </summary>
        public DbSet<AlertModels> Alert { get; set; } = null!;

        /// <summary>
        /// Gets or sets the agent deployment tokens
        /// </summary>
        public DbSet<AthalaSIEM.Backend.Models.AgentDeploymentToken> AgentDeploymentTokens { get; set; } = null!;

        /// <summary>
        /// Gets or sets the system configuration
        /// </summary>
        public DbSet<AthalaSIEM.Backend.Models.SystemConfiguration> SystemConfiguration { get; set; } = null!;

        /// <summary>
        /// Gets or sets the file integrity events
        /// </summary>
        public DbSet<FileIntegrityEvent> FileIntegrityEvents { get; set; } = null!;

        /// <summary>
        /// Gets or sets the file integrity rules
        /// </summary>
        public DbSet<FileIntegrityRule> FileIntegrityRules { get; set; } = null!;

        /// <summary>
        /// Gets or sets the file integrity baselines
        /// </summary>
        public DbSet<FileIntegrityBaseline> FileIntegrityBaselines { get; set; } = null!;

        /// <summary>
        /// Gets or sets the threat intelligence feeds
        /// </summary>
        public DbSet<ThreatIntelligenceFeed> ThreatIntelligenceFeeds { get; set; } = null!;

        /// <summary>
        /// Gets or sets the threat indicators
        /// </summary>
        public DbSet<ThreatIndicator> ThreatIndicators { get; set; } = null!;

        /// <summary>
        /// Gets or sets the threat matches
        /// </summary>
        public DbSet<ThreatMatch> ThreatMatches { get; set; } = null!;

        /// <summary>
        /// Gets or sets the threat campaigns
        /// </summary>
        public DbSet<ThreatCampaign> ThreatCampaigns { get; set; } = null!;

        /// <summary>
        /// Gets or sets the attack techniques
        /// </summary>
        public DbSet<AttackTechnique> AttackTechniques { get; set; } = null!;

        /// <summary>
        /// Gets or sets the threat enrichments
        /// </summary>
        public DbSet<ThreatEnrichment> ThreatEnrichments { get; set; } = null!;

        /// <summary>
        /// Gets or sets the threat whitelist
        /// </summary>
        public DbSet<ThreatWhitelist> ThreatWhitelistEntries { get; set; } = null!;

        /// <summary>
        /// Gets or sets the alert metadata
        /// </summary>
        public DbSet<AlertMetadataModels> AlertMetadata { get; set; } = null!;

        /// <summary>
        /// Gets or sets the alert correlations
        /// </summary>
        public DbSet<AlertCorrelationModels> AlertCorrelations { get; set; } = null!;

        /// <summary>
        /// Gets or sets the alert rules
        /// </summary>
        public DbSet<AlertRuleModels> AlertRulesNew { get; set; } = null!;

        /// <summary>
        /// Gets or sets the normalized logs
        /// </summary>
        public DbSet<Backend.Domain.Entities.NormalizedLog> NormalizedLogs { get; set; } = null!;

        /// <summary>
        /// Gets or sets the collector configurations
        /// </summary>
        public DbSet<CollectorConfigurationModels> CollectorConfigurations { get; set; } = null!;

        /// <summary>
        /// Gets or sets the system settings (key-value per category).
        /// </summary>
        public DbSet<SystemSetting> SystemSettings { get; set; } = null!;

        /// <summary>
        /// Configures the model
        /// </summary>
        /// <param name="modelBuilder">The model builder</param>
        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Users table
            modelBuilder.Entity<UserModels>(entity =>
            {
                entity.ToTable("users");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Email).IsRequired().HasMaxLength(100);
                entity.Property(e => e.Username).IsRequired().HasMaxLength(50);
                entity.Property(e => e.PasswordHash).IsRequired();
                entity.HasIndex(e => e.Email).IsUnique();
                entity.HasIndex(e => e.Username).IsUnique();
            });

            // System settings table (category + key unique)
            modelBuilder.Entity<SystemSetting>(entity =>
            {
                entity.ToTable("system_settings");
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.Category, e.Key }).IsUnique();
            });

            // Agents table
            modelBuilder.Entity<AgentModels>(entity =>
            {
                entity.ToTable("agents");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(100);
                entity.Property(e => e.ApiKey).IsRequired().HasMaxLength(100);
                entity.Property(e => e.Status).HasConversion<string>();
                entity.Property(e => e.Type).HasConversion<string>();
                entity.HasIndex(e => e.ApiKey).IsUnique();
                
                entity.HasOne(a => a.Configuration)
                    .WithOne(c => c.Agent)
                    .HasForeignKey<AgentConfigModels>(c => c.AgentId);
                
                entity.HasOne(a => a.CreatedBy)
                    .WithMany()
                    .HasForeignKey(a => a.CreatedById)
                    .IsRequired(false)
                    .OnDelete(DeleteBehavior.SetNull);
            });

            // Log Entries table
            modelBuilder.Entity<LogEntryModels>(entity =>
            {
                entity.ToTable("log_entries");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Message).IsRequired();
                entity.Property(e => e.Level).IsRequired();
                entity.Property(e => e.Timestamp).IsRequired();
                entity.HasIndex(e => e.Timestamp);
                entity.HasIndex(e => e.Level);
                
                entity.HasOne(l => l.Agent)
                    .WithMany(a => a.LogEntries)
                    .HasForeignKey(l => l.AgentId);
            });

            // Health Metrics table
            modelBuilder.Entity<HealthMetricModels>(entity =>
            {
                entity.ToTable("health_metrics");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Timestamp).IsRequired();
                entity.HasIndex(e => e.Timestamp);
                
                entity.HasOne(h => h.Agent)
                    .WithMany(a => a.HealthMetrics)
                    .HasForeignKey(h => h.AgentId);
            });

            // Alerts table
            modelBuilder.Entity<AlertModels>(entity =>
            {
                entity.ToTable("alerts");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Message).IsRequired();
                entity.Property(e => e.Severity).IsRequired();
                entity.Property(e => e.Timestamp).IsRequired();
                entity.HasIndex(e => e.Timestamp);
                entity.HasIndex(e => e.Severity);
                
                entity.HasOne(a => a.Agent)
                    .WithMany(a => a.Alerts)
                    .HasForeignKey(a => a.AgentId);
            });

            // Security Events table
            modelBuilder.Entity<SecurityEventModels>(entity =>
            {
                entity.ToTable("security_events");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Message).IsRequired();
                entity.Property(e => e.Timestamp).IsRequired();
                entity.Property(e => e.Severity).HasConversion<string>();
                entity.HasIndex(e => e.Timestamp);
                
                entity.HasOne(s => s.Agent)
                    .WithMany(a => a.SecurityEvents)
                    .HasForeignKey(s => s.AgentId);
            });

            // Roles table
            modelBuilder.Entity<RoleModels>(entity =>
            {
                entity.ToTable("roles");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(50);
            });

            // User Roles table
            modelBuilder.Entity<UserRoleModels>(entity =>
            {
                entity.ToTable("user_roles");
                entity.HasKey(e => new { e.UserId, e.RoleId });
                
                entity.HasOne(ur => ur.User)
                    .WithMany(u => u.UserRoles)
                    .HasForeignKey(ur => ur.UserId);
                
                entity.HasOne(ur => ur.Role)
                    .WithMany(r => r.UserRoles)
                    .HasForeignKey(ur => ur.RoleId);
            });

            // User Security Settings table
            modelBuilder.Entity<UserSecurityModels>(entity =>
            {
                entity.ToTable("user_security_settings");
                entity.HasKey(e => e.UserId);
                
                entity.HasOne(uss => uss.User)
                    .WithOne()
                    .HasForeignKey<UserSecurityModels>(uss => uss.UserId)
                    .OnDelete(DeleteBehavior.Cascade);
            });

            // Dashboards table
            modelBuilder.Entity<DashboardModels>(entity =>
            {
                entity.ToTable("dashboards");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(100);
                entity.Property(e => e.Type).HasMaxLength(50);
                
                entity.HasOne(d => d.User)
                    .WithMany(u => u.Dashboards)
                    .HasForeignKey(d => d.UserId);
            });

            // Reports table
            modelBuilder.Entity<ReportModels>(entity =>
            {
                entity.ToTable("reports");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(100);
                
                entity.HasOne(r => r.User)
                    .WithMany(u => u.Reports)
                    .HasForeignKey(r => r.UserId);
            });

            // Alert Rules table
            modelBuilder.Entity<AlertRuleModels>(entity =>
            {
                entity.ToTable("alert_rules");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(100);
                entity.Property(e => e.Condition).IsRequired();
            });

            // Agent Health Reports table
            modelBuilder.Entity<AgentHealthReport>(entity =>
            {
                entity.ToTable("agent_health_reports");
                entity.HasKey(e => e.Id);
                
                entity.HasOne(hr => hr.Agent)
                    .WithMany(a => a.HealthReports)
                    .HasForeignKey(hr => hr.AgentId);
            });

            // Agent Heartbeats table
            modelBuilder.Entity<AgentHeartbeatModels>(entity =>
            {
                entity.ToTable("agent_heartbeats");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Timestamp).IsRequired();
                entity.HasIndex(e => e.Timestamp);
                
                entity.HasOne(h => h.Agent)
                    .WithMany(a => a.Heartbeats)
                    .HasForeignKey(h => h.AgentId);
            });

            // File Integrity Events table
            modelBuilder.Entity<FileIntegrityEvent>(entity =>
            {
                entity.ToTable("file_integrity_events");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.AgentId).IsRequired().HasMaxLength(50);
                entity.Property(e => e.FilePath).IsRequired().HasMaxLength(500);
                entity.Property(e => e.ChangeType).IsRequired().HasMaxLength(20);
                entity.Property(e => e.BaselineHash).HasMaxLength(128);
                entity.Property(e => e.CurrentHash).HasMaxLength(128);
                entity.Property(e => e.FileAttributes).HasMaxLength(100);
                entity.Property(e => e.Severity).IsRequired().HasMaxLength(20);
                entity.Property(e => e.AcknowledgedBy).HasMaxLength(50);
                entity.Property(e => e.DetectedAt).IsRequired();
                entity.Property(e => e.ProcessedAt).IsRequired();
                
                entity.HasIndex(e => e.AgentId);
                entity.HasIndex(e => e.DetectedAt);
                entity.HasIndex(e => e.Severity);
                entity.HasIndex(e => e.ChangeType);
                entity.HasIndex(e => e.IsAcknowledged);
                
                entity.HasOne(e => e.Agent)
                    .WithMany()
                    .HasForeignKey(e => e.AgentId)
                    .OnDelete(DeleteBehavior.Cascade);
            });

            // File Integrity Rules table
            modelBuilder.Entity<FileIntegrityRule>(entity =>
            {
                entity.ToTable("file_integrity_rules");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(100);
                entity.Property(e => e.Description).HasMaxLength(500);
                entity.Property(e => e.MonitoredPaths).IsRequired();
                entity.Property(e => e.Severity).IsRequired().HasMaxLength(20);
                entity.Property(e => e.CreatedBy).HasMaxLength(50);
                entity.Property(e => e.CreatedAt).IsRequired();
                entity.Property(e => e.UpdatedAt).IsRequired();
                
                entity.HasIndex(e => e.Name).IsUnique();
                entity.HasIndex(e => e.IsEnabled);
                entity.HasIndex(e => e.CreatedAt);
            });

            // File Integrity Baselines table
            modelBuilder.Entity<FileIntegrityBaseline>(entity =>
            {
                entity.ToTable("file_integrity_baselines");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.AgentId).IsRequired().HasMaxLength(50);
                entity.Property(e => e.FilePath).IsRequired().HasMaxLength(500);
                entity.Property(e => e.FileHash).IsRequired().HasMaxLength(128);
                entity.Property(e => e.FileAttributes).HasMaxLength(100);
                entity.Property(e => e.BaselineCreatedAt).IsRequired();
                entity.Property(e => e.BaselineUpdatedAt).IsRequired();
                
                entity.HasIndex(e => new { e.AgentId, e.FilePath }).IsUnique();
                entity.HasIndex(e => e.AgentId);
                entity.HasIndex(e => e.IsActive);
                entity.HasIndex(e => e.BaselineCreatedAt);
                
                entity.HasOne(b => b.Agent)
                    .WithMany()
                    .HasForeignKey(b => b.AgentId)
                    .OnDelete(DeleteBehavior.Cascade);
            });

            // Threat Intelligence Feeds table
            modelBuilder.Entity<ThreatIntelligenceFeed>(entity =>
            {
                entity.ToTable("threat_intelligence_feeds");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(100);
                entity.Property(e => e.FeedType).IsRequired().HasMaxLength(50);
                entity.Property(e => e.FeedUrl).IsRequired();
                entity.Property(e => e.ApiKey).HasMaxLength(100);
                entity.Property(e => e.Username).HasMaxLength(100);
                entity.Property(e => e.Password).HasMaxLength(100);
                entity.Property(e => e.Priority).IsRequired().HasMaxLength(20);
                entity.Property(e => e.Source).IsRequired().HasMaxLength(50);
                entity.Property(e => e.CreatedBy).HasMaxLength(50);
                entity.Property(e => e.CreatedAt).IsRequired();
                entity.Property(e => e.LastUpdated).IsRequired();

                entity.HasIndex(e => e.Name).IsUnique();
                entity.HasIndex(e => e.IsActive);
                entity.HasIndex(e => e.LastUpdated);
                entity.HasIndex(e => e.Priority);
            });

            // Threat Indicators table
            modelBuilder.Entity<ThreatIndicator>(entity =>
            {
                entity.ToTable("threat_indicators");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Type).IsRequired().HasMaxLength(50);
                entity.Property(e => e.Value).IsRequired().HasMaxLength(500);
                entity.Property(e => e.Confidence).IsRequired().HasMaxLength(20);
                entity.Property(e => e.Severity).IsRequired().HasMaxLength(20);
                entity.Property(e => e.ThreatType).HasMaxLength(200);
                entity.Property(e => e.MalwareFamily).HasMaxLength(100);
                entity.Property(e => e.Description).HasMaxLength(500);
                entity.Property(e => e.FeedId).IsRequired();
                entity.Property(e => e.Source).HasMaxLength(100);
                entity.Property(e => e.FirstSeen).IsRequired();
                entity.Property(e => e.LastSeen).IsRequired();

                entity.HasIndex(e => new { e.Type, e.Value });
                entity.HasIndex(e => e.Type);
                entity.HasIndex(e => e.Severity);
                entity.HasIndex(e => e.Confidence);
                entity.HasIndex(e => e.FeedId);
                entity.HasIndex(e => e.IsActive);
                entity.HasIndex(e => e.FirstSeen);
                entity.HasIndex(e => e.LastSeen);

                entity.HasOne(i => i.Feed)
                    .WithMany()
                    .HasForeignKey(i => i.FeedId)
                    .OnDelete(DeleteBehavior.Cascade);
            });

            // Threat Matches table
            modelBuilder.Entity<ThreatMatch>(entity =>
            {
                entity.ToTable("threat_matches");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.IndicatorId).IsRequired();
                entity.Property(e => e.LogEntryId).IsRequired();
                entity.Property(e => e.MatchedValue).IsRequired().HasMaxLength(500);
                entity.Property(e => e.MatchedField).IsRequired().HasMaxLength(100);
                entity.Property(e => e.Confidence).IsRequired().HasMaxLength(20);
                entity.Property(e => e.Severity).IsRequired().HasMaxLength(20);
                entity.Property(e => e.DetectedAt).IsRequired();
                entity.Property(e => e.AcknowledgedBy).HasMaxLength(50);
                entity.Property(e => e.Notes).HasMaxLength(500);

                entity.HasIndex(e => e.IndicatorId);
                entity.HasIndex(e => e.LogEntryId);
                entity.HasIndex(e => e.DetectedAt);
                entity.HasIndex(e => e.Severity);
                entity.HasIndex(e => e.IsAcknowledged);
                entity.HasIndex(e => e.IsFalsePositive);

                entity.HasOne(m => m.Indicator)
                    .WithMany(i => i.Matches)
                    .HasForeignKey(m => m.IndicatorId)
                    .OnDelete(DeleteBehavior.Cascade);

                entity.HasOne(m => m.LogEntry)
                    .WithMany()
                    .HasForeignKey(m => m.LogEntryId)
                    .OnDelete(DeleteBehavior.Cascade);
            });

            // Threat Campaigns table
            modelBuilder.Entity<ThreatCampaign>(entity =>
            {
                entity.ToTable("threat_campaigns");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(200);
                entity.Property(e => e.Description).HasMaxLength(1000);
                entity.Property(e => e.Actor).HasMaxLength(100);
                entity.Property(e => e.Severity).IsRequired().HasMaxLength(50);
                entity.Property(e => e.CreatedBy).HasMaxLength(50);
                entity.Property(e => e.FirstDetected).IsRequired();
                entity.Property(e => e.LastDetected).IsRequired();
                entity.Property(e => e.CreatedAt).IsRequired();

                entity.HasIndex(e => e.Name).IsUnique();
                entity.HasIndex(e => e.Severity);
                entity.HasIndex(e => e.IsActive);
                entity.HasIndex(e => e.FirstDetected);
                entity.HasIndex(e => e.LastDetected);
            });

            // Attack Techniques table
            modelBuilder.Entity<AttackTechnique>(entity =>
            {
                entity.ToTable("attack_techniques");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.TechniqueId).IsRequired().HasMaxLength(20);
                entity.Property(e => e.Name).IsRequired().HasMaxLength(200);
                entity.Property(e => e.Description).HasMaxLength(1000);
                entity.Property(e => e.Tactic).IsRequired().HasMaxLength(50);
                entity.Property(e => e.Platform).IsRequired().HasMaxLength(50);

                entity.HasIndex(e => e.TechniqueId).IsUnique();
                entity.HasIndex(e => e.Tactic);
                entity.HasIndex(e => e.Platform);
                entity.HasIndex(e => e.IsActive);
            });

            // Threat Enrichment table
            modelBuilder.Entity<ThreatEnrichment>(entity =>
            {
                entity.ToTable("threat_enrichments");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.IndicatorValue).IsRequired().HasMaxLength(500);
                entity.Property(e => e.IndicatorType).IsRequired().HasMaxLength(50);
                entity.Property(e => e.EnrichmentSource).IsRequired().HasMaxLength(100);
                entity.Property(e => e.EnrichmentData).IsRequired();
                entity.Property(e => e.Status).IsRequired().HasMaxLength(20);
                entity.Property(e => e.EnrichedAt).IsRequired();

                entity.HasIndex(e => new { e.IndicatorValue, e.IndicatorType, e.EnrichmentSource });
                entity.HasIndex(e => e.EnrichedAt);
                entity.HasIndex(e => e.Status);
            });

            // Threat Whitelist table
            modelBuilder.Entity<ThreatWhitelist>(entity =>
            {
                entity.ToTable("threat_whitelists");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.Type).IsRequired().HasMaxLength(50);
                entity.Property(e => e.Value).IsRequired().HasMaxLength(500);
                entity.Property(e => e.Reason).HasMaxLength(500);
                entity.Property(e => e.CreatedBy).HasMaxLength(50);
                entity.Property(e => e.CreatedAt).IsRequired();

                entity.HasIndex(e => new { e.Type, e.Value }).IsUnique();
                entity.HasIndex(e => e.IsActive);
                entity.HasIndex(e => e.CreatedAt);
            });

            // NormalizedLog entity configuration
            modelBuilder.Entity<Backend.Domain.Entities.NormalizedLog>(entity =>
            {
                entity.ToTable("normalized_logs");
                entity.HasKey(e => e.Id);
                entity.Property(e => e.LogEntryId).IsRequired();
                entity.Property(e => e.Timestamp).IsRequired();
                entity.Property(e => e.CreatedAt).IsRequired();
                
                // Ignore navigation property to domain LogEntry since we use LogEntryModels
                entity.Ignore(e => e.LogEntry);
                
                // Configure JSON properties
                entity.Property(e => e.MetadataJson)
                    .HasColumnName("metadata_json");
            });
        }
    }
}