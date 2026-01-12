using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Backend.Migrations
{
    /// <inheritdoc />
    public partial class AddNormalizedLogsAndDetectionRules : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // Create normalized_logs table
            migrationBuilder.CreateTable(
                name: "normalized_logs",
                columns: table => new
                {
                    id = table.Column<string>(type: "text", nullable: false),
                    log_entry_id = table.Column<string>(type: "text", nullable: false),
                    timestamp = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    agent_id = table.Column<string>(type: "text", nullable: true),
                    agent_name = table.Column<string>(type: "text", nullable: true),
                    host_name = table.Column<string>(type: "text", nullable: true),
                    host_ip = table.Column<string>(type: "text", nullable: true),
                    user_name = table.Column<string>(type: "text", nullable: true),
                    user_id = table.Column<string>(type: "text", nullable: true),
                    user_domain = table.Column<string>(type: "text", nullable: true),
                    process_name = table.Column<string>(type: "text", nullable: true),
                    process_id = table.Column<int>(type: "integer", nullable: true),
                    process_path = table.Column<string>(type: "text", nullable: true),
                    process_command_line = table.Column<string>(type: "text", nullable: true),
                    process_hash = table.Column<string>(type: "text", nullable: true),
                    parent_process_name = table.Column<string>(type: "text", nullable: true),
                    parent_process_id = table.Column<int>(type: "integer", nullable: true),
                    source_ip = table.Column<string>(type: "text", nullable: true),
                    source_port = table.Column<int>(type: "integer", nullable: true),
                    destination_ip = table.Column<string>(type: "text", nullable: true),
                    destination_port = table.Column<int>(type: "integer", nullable: true),
                    protocol = table.Column<string>(type: "text", nullable: true),
                    event_action = table.Column<string>(type: "text", nullable: true),
                    event_category = table.Column<string>(type: "text", nullable: true),
                    event_type = table.Column<string>(type: "text", nullable: true),
                    event_outcome = table.Column<string>(type: "text", nullable: true),
                    event_code = table.Column<string>(type: "text", nullable: true),
                    file_path = table.Column<string>(type: "text", nullable: true),
                    file_name = table.Column<string>(type: "text", nullable: true),
                    file_hash = table.Column<string>(type: "text", nullable: true),
                    file_size = table.Column<long>(type: "bigint", nullable: true),
                    siem_rule_id = table.Column<string>(type: "text", nullable: true),
                    siem_technique_id = table.Column<string>(type: "text", nullable: true),
                    siem_confidence = table.Column<double>(type: "double precision", nullable: true),
                    siem_severity = table.Column<int>(type: "integer", nullable: true),
                    siem_correlation_id = table.Column<string>(type: "text", nullable: true),
                    metadata_json = table.Column<string>(type: "text", nullable: true),
                    created_at = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_normalized_logs", x => x.id);
                    table.ForeignKey(
                        name: "fk_normalized_logs_log_entries_log_entry_id",
                        column: x => x.log_entry_id,
                        principalTable: "log_entries",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            // Create indexes for fast queries
            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_log_entry_id",
                table: "normalized_logs",
                column: "log_entry_id",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_timestamp",
                table: "normalized_logs",
                column: "timestamp");

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_agent_id",
                table: "normalized_logs",
                column: "agent_id");

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_source_ip",
                table: "normalized_logs",
                column: "source_ip");

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_destination_ip",
                table: "normalized_logs",
                column: "destination_ip");

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_process_name",
                table: "normalized_logs",
                column: "process_name");

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_user_name",
                table: "normalized_logs",
                column: "user_name");

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_siem_correlation_id",
                table: "normalized_logs",
                column: "siem_correlation_id");

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_siem_technique_id",
                table: "normalized_logs",
                column: "siem_technique_id");

            migrationBuilder.CreateIndex(
                name: "ix_normalized_logs_event_action",
                table: "normalized_logs",
                column: "event_action");

            // Add deduplication_key to alerts table
            migrationBuilder.AddColumn<string>(
                name: "deduplication_key",
                table: "alerts",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "occurrence_count",
                table: "alerts",
                type: "integer",
                nullable: false,
                defaultValue: 1);

            migrationBuilder.AddColumn<DateTime>(
                name: "first_occurrence",
                table: "alerts",
                type: "timestamp with time zone",
                nullable: true);

            migrationBuilder.AddColumn<DateTime>(
                name: "last_occurrence",
                table: "alerts",
                type: "timestamp with time zone",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "rule_id",
                table: "alerts",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "correlation_id",
                table: "alerts",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<double>(
                name: "confidence",
                table: "alerts",
                type: "double precision",
                nullable: false,
                defaultValue: 0.0);

            migrationBuilder.AddColumn<string>(
                name: "detection_reason",
                table: "alerts",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "detection_metadata_json",
                table: "alerts",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "technique_ids_json",
                table: "alerts",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "related_log_ids_json",
                table: "alerts",
                type: "text",
                nullable: true);

            // Create indexes on alerts
            migrationBuilder.CreateIndex(
                name: "ix_alerts_deduplication_key",
                table: "alerts",
                column: "deduplication_key");

            migrationBuilder.CreateIndex(
                name: "ix_alerts_rule_id",
                table: "alerts",
                column: "rule_id");

            migrationBuilder.CreateIndex(
                name: "ix_alerts_correlation_id",
                table: "alerts",
                column: "correlation_id");

            migrationBuilder.CreateIndex(
                name: "ix_alerts_timestamp",
                table: "alerts",
                column: "timestamp");

            // Update alert_rules table if needed
            migrationBuilder.Sql(@"
                ALTER TABLE alert_rules 
                ADD COLUMN IF NOT EXISTS rule_type VARCHAR(50),
                ADD COLUMN IF NOT EXISTS threshold_count INTEGER,
                ADD COLUMN IF NOT EXISTS threshold_window_seconds INTEGER,
                ADD COLUMN IF NOT EXISTS technique_ids_json TEXT,
                ADD COLUMN IF NOT EXISTS whitelist_ips_json TEXT,
                ADD COLUMN IF NOT EXISTS whitelist_users_json TEXT,
                ADD COLUMN IF NOT EXISTS whitelist_processes_json TEXT,
                ADD COLUMN IF NOT EXISTS match_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS false_positive_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS last_matched_at TIMESTAMP WITH TIME ZONE;
            ");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(name: "normalized_logs");
            
            migrationBuilder.DropIndex(name: "ix_alerts_deduplication_key", table: "alerts");
            migrationBuilder.DropIndex(name: "ix_alerts_rule_id", table: "alerts");
            migrationBuilder.DropIndex(name: "ix_alerts_correlation_id", table: "alerts");
            
            migrationBuilder.DropColumn(name: "deduplication_key", table: "alerts");
            migrationBuilder.DropColumn(name: "occurrence_count", table: "alerts");
            migrationBuilder.DropColumn(name: "first_occurrence", table: "alerts");
            migrationBuilder.DropColumn(name: "last_occurrence", table: "alerts");
            migrationBuilder.DropColumn(name: "rule_id", table: "alerts");
            migrationBuilder.DropColumn(name: "correlation_id", table: "alerts");
            migrationBuilder.DropColumn(name: "confidence", table: "alerts");
            migrationBuilder.DropColumn(name: "detection_reason", table: "alerts");
            migrationBuilder.DropColumn(name: "detection_metadata_json", table: "alerts");
            migrationBuilder.DropColumn(name: "technique_ids_json", table: "alerts");
            migrationBuilder.DropColumn(name: "related_log_ids_json", table: "alerts");
        }
    }
}
