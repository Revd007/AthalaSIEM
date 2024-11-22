import argparse
import os
import subprocess
import sys
import logging
from typing import List

class LinuxAgentInstaller:
    def __init__(self):
        self.config_dir = "/etc/donquixote-athala"
        self.service_name = "donquixote-athala-agent"
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("LinuxInstaller")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def install(self, server_address: str, log_sources: List[str]):
        try:
            self.logger.info("Starting Athala SIEM agent installation...")
            
            # Create configuration directory
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Save configuration
            self._save_config(server_address, log_sources)
            
            # Create systemd service
            self._create_systemd_service()
            
            # Enable and start service
            self._setup_service()
            
            self.logger.info("Installation completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Installation failed: {str(e)}")
            sys.exit(1)

    def _save_config(self, server_address: str, log_sources: List[str]):
        config_path = f"{self.config_dir}/agent.conf"
        self.logger.info(f"Saving configuration to {config_path}")
        
        with open(config_path, "w") as f:
            f.write(f"SERVER_ADDRESS={server_address}\n")
            f.write(f"LOG_SOURCES={','.join(log_sources)}\n")

    def _create_systemd_service(self):
        service_content = f"""[Unit]
Description=Donquixote Athala SIEM Agent
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/athala-agent
Restart=always
User=root

[Install]
WantedBy=multi-user.target
"""
        
        service_path = f"/etc/systemd/system/{self.service_name}.service"
        self.logger.info(f"Creating systemd service at {service_path}")
        
        with open(service_path, "w") as f:
            f.write(service_content)

    def _setup_service(self):
        commands = [
            ["systemctl", "daemon-reload"],
            ["systemctl", "enable", self.service_name],
            ["systemctl", "start", self.service_name]
        ]
        
        for cmd in commands:
            self.logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Athala SIEM Linux Agent Installer")
    parser.add_argument("--server", required=True, help="AthalaSIEM server address (e.g., server.example.com:8080)")
    parser.add_argument("--logs", nargs="+", default=["/var/log/syslog", "/var/log/auth.log"],
                        help="List of log files to monitor")
    
    args = parser.parse_args()
    
    installer = LinuxAgentInstaller()
    installer.install(args.server, args.logs)

if __name__ == "__main__":
    main()