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
        }
    }
}