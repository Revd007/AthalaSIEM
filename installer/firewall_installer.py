import argparse
import paramiko
import sys
import logging
from typing import Dict

class FirewallAgentInstaller:
    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("FirewallInstaller")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def install(self, device_type: str, device_ip: str, username: str, 
                password: str, server_address: str):
        try:
            self.logger.info(f"Configuring {device_type} device at {device_ip}")
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.logger.info("Establishing SSH connection...")
            ssh.connect(device_ip, username=username, password=password)
            
            # Configure based on device type
            config_method = getattr(self, f"_configure_{device_type.lower()}", None)
            if config_method:
                config_method(ssh, server_address)
            else:
                raise ValueError(f"Unsupported device type: {device_type}")
            
            self.logger.info("Configuration completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Configuration failed: {str(e)}")
            sys.exit(1)
        finally:
            ssh.close()

    def _configure_cisco(self, ssh: paramiko.SSHClient, server_address: str):
        commands = [
            "conf t",
            f"logging host {server_address}",
            "logging trap debugging",
            "logging facility local6",
            "end",
            "write memory"
        ]
        
        self._execute_commands(ssh, commands)

    def _configure_fortinet(self, ssh: paramiko.SSHClient, server_address: str):
        commands = [
            "config log syslogd setting",
            "set status enable",
            f"set server {server_address}",
            "set facility local6",
            "end"
        ]
        
        self._execute_commands(ssh, commands)

    def _configure_paloalto(self, ssh: paramiko.SSHClient, server_address: str):
        commands = [
            "configure",
            "set deviceconfig system syslog server primary",
            f"set server {server_address}",
            "set facility local6",
            "commit"
        ]
        
        self._execute_commands(ssh, commands)

    def _execute_commands(self, ssh: paramiko.SSHClient, commands: list):
        for cmd in commands:
            self.logger.info(f"Executing: {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status != 0:
                error = stderr.read().decode().strip()
                raise Exception(f"Command failed: {error}")

def main():
    parser = argparse.ArgumentParser(description="Athala SIEM Firewall Agent Installer")
    parser.add_argument("--type", required=True, choices=["cisco", "fortinet", "paloalto"],
                        help="Firewall device type")
    parser.add_argument("--ip", required=True, help="Device IP address")
    parser.add_argument("--username", required=True, help="SSH username")
    parser.add_argument("--password", required=True, help="SSH password")
    parser.add_argument("--server", required=True, help="AthalaSIEM server address")
    
    args = parser.parse_args()
    
    installer = FirewallAgentInstaller()
    installer.install(args.type, args.ip, args.username, args.password, args.server)

if __name__ == "__main__":
    main()