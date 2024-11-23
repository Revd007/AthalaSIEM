import logging
import sys
import paramiko
from typing import Dict, Optional

class FirewallInstaller:
    def __init__(self):
        self.logger = self._setup_logging()
        self.supported_vendors = {
            'mikrotik': self._install_mikrotik,
            'juniper': self._install_juniper,
            'cisco': self._install_cisco,
            'paloalto': self._install_paloalto,
            'iforte': self._install_iforte
        }

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("FirewallInstaller")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def install(self, vendor: str, connection_params: Dict[str, str]):
        """
        Main installation method that handles different firewall vendors
        """
        try:
            vendor = vendor.lower()
            if vendor not in self.supported_vendors:
                self.logger.error(f"Unsupported firewall vendor: {vendor}")
                return False

            self.logger.info(f"Starting installation for {vendor} firewall...")
            return self.supported_vendors[vendor](connection_params)

        except Exception as e:
            self.logger.error(f"Installation failed: {str(e)}")
            return False

    def _create_ssh_connection(self, params: Dict[str, str]) -> Optional[paramiko.SSHClient]:
        """Create SSH connection to firewall"""
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=params['host'],
                username=params['username'],
                password=params['password'],
                port=int(params.get('port', 22))
            )
            return ssh
        except Exception as e:
            self.logger.error(f"SSH connection failed: {str(e)}")
            return None

    def _install_mikrotik(self, params: Dict[str, str]) -> bool:
        """Install agent on Mikrotik firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # Mikrotik-specific installation commands
            commands = [
                "/system script add name=athala-agent source=\"{YOUR_SCRIPT}\"",
                "/system scheduler add name=athala-agent-scheduler interval=1m on-event=athala-agent"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on Mikrotik firewall")
            return True

        except Exception as e:
            self.logger.error(f"Mikrotik installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

    def _install_juniper(self, params: Dict[str, str]) -> bool:
        """Install agent on Juniper firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # Juniper-specific installation commands
            commands = [
                "set system scripts file athala-agent.py contents \"{YOUR_SCRIPT}\"",
                "set event-options generate-event athala-timer time-interval 60",
                "set event-options policy athala-policy events athala-timer then execute-commands commands athala-agent.py"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on Juniper firewall")
            return True

        except Exception as e:
            self.logger.error(f"Juniper installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

    def _install_cisco(self, params: Dict[str, str]) -> bool:
        """Install agent on Cisco firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # Cisco-specific installation commands
            commands = [
                "conf t",
                "file prompt quiet",
                "copy tftp://server/athala-agent.tcl flash:",
                "tclsh flash:athala-agent.tcl"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on Cisco firewall")
            return True

        except Exception as e:
            self.logger.error(f"Cisco installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

    def _install_paloalto(self, params: Dict[str, str]) -> bool:
        """Install agent on Palo Alto firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # Palo Alto-specific installation commands
            commands = [
                "set cli scripting-mode on",
                "copy tftp://server/athala-agent.xml to athala-agent.xml",
                "load config partial from athala-agent.xml",
                "commit"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on Palo Alto firewall")
            return True

        except Exception as e:
            self.logger.error(f"Palo Alto installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

    def _install_iforte(self, params: Dict[str, str]) -> bool:
        """Install agent on Iforte firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # Iforte-specific installation commands
            commands = [
                "system script create name=athala-agent source=\"{YOUR_SCRIPT}\"",
                "system scheduler add name=athala-scheduler interval=1m script=athala-agent"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on Iforte firewall")
            return True

        except Exception as e:
            self.logger.error(f"Iforte installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

    def _install_pfsense(self, params: Dict[str, str]) -> bool:
        """Install agent on pfSense firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # pfSense-specific installation commands
            commands = [
                "pkg install -y python3",
                "fetch https://server/athala-agent.py -o /usr/local/etc/athala-agent.py",
                "chmod +x /usr/local/etc/athala-agent.py",
                "echo '*/5 * * * * root python3 /usr/local/etc/athala-agent.py' >> /etc/crontab"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on pfSense firewall")
            return True

        except Exception as e:
            self.logger.error(f"pfSense installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

    def _install_fortinet(self, params: Dict[str, str]) -> bool:
        """Install agent on Fortinet firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # Fortinet-specific installation commands
            commands = [
                "config system automation-action",
                "edit \"athala-agent\"",
                "set action-type cli-script",
                "set script \"execute athala-agent.sh\"",
                "next",
                "end",
                "config system automation-trigger",
                "edit \"athala-schedule\"", 
                "set trigger-type scheduled",
                "set trigger-frequency hourly",
                "next",
                "end"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on Fortinet firewall")
            return True

        except Exception as e:
            self.logger.error(f"Fortinet installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

    def _install_cisco(self, params: Dict[str, str]) -> bool:
        """Install agent on Cisco firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # Cisco ASA specific installation commands
            commands = [
                "conf t",
                "logging enable",
                "logging trap informational",
                "logging host inside " + params.get('server_ip', ''),
                "logging facility 20",
                "write memory"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on Cisco firewall")
            return True

        except Exception as e:
            self.logger.error(f"Cisco installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

    def _install_checkpoint(self, params: Dict[str, str]) -> bool:
        """Install agent on Check Point firewall"""
        ssh = self._create_ssh_connection(params)
        if not ssh:
            return False

        try:
            # Check Point specific installation commands
            commands = [
                "add script athala-agent",
                "modify script athala-agent command \"python3 /var/log/athala-agent.py\"",
                "add scheduler task athala-task",
                "modify scheduler task athala-task recurrence every 5 minutes",
                "modify scheduler task athala-task command \"script athala-agent\"",
                "save config"
            ]

            for cmd in commands:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                if stderr.read():
                    raise Exception(f"Error executing command: {cmd}")

            self.logger.info("Successfully installed agent on Check Point firewall")
            return True

        except Exception as e:
            self.logger.error(f"Check Point installation failed: {str(e)}")
            return False
        finally:
            ssh.close()

