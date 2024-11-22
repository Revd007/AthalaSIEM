from typing import Dict, Any, Optional
import subprocess
import os
import logging
import winreg
from pathlib import Path
import json
import shutil
from urllib.request import urlretrieve

class InstallerManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.install_path = Path(config.get('install_path', 'C:/Program Files/AthalaSIEM'))
        self.temp_path = Path('temp')
        self.sql_express_url = "https://go.microsoft.com/fwlink/?linkid=866658"
        
    async def run_installation(self) -> Dict[str, Any]:
        """Run full installation process"""
        try:
            # Create directories
            self._create_directories()
            
            # Install SQL Server Express if needed
            if not self._check_sql_server_installed():
                await self._install_sql_server()
                
            # Install application files
            self._install_application()
            
            # Configure database
            await self._configure_database()
            
            # Install Windows service
            self._install_service()
            
            # Cleanup temporary files
            self._cleanup()
            
            return {
                'status': 'success',
                'install_path': str(self.install_path),
                'database': 'SQL Server Express',
                'service_name': 'AthalaSIEM'
            }
            
        except Exception as e:
            self.logger.error(f"Installation failed: {e}")
            self._cleanup()
            raise

    async def _install_sql_server(self) -> None:
        """Download and install SQL Server Express"""
        try:
            # Download SQL Server Express
            sql_installer = self.temp_path / "SQLEXPR.exe"
            self.logger.info("Downloading SQL Server Express...")
            urlretrieve(self.sql_express_url, sql_installer)
            
            # Prepare configuration file
            config_file = self._create_sql_config()
            
            # Install SQL Server Express
            self.logger.info("Installing SQL Server Express...")
            subprocess.run([
                str(sql_installer),
                "/IACCEPTSQLSERVERLICENSETERMS",
                "/CONFIGURATIONFILE=" + str(config_file),
                "/Q",  # Quiet mode
                "/HIDECONSOLE"
            ], check=True)
            
        except Exception as e:
            raise Exception(f"SQL Server installation failed: {e}")

    def _create_sql_config(self) -> Path:
        """Create SQL Server configuration file"""
        config_content = f"""
[OPTIONS]
IACCEPTSQLSERVERLICENSETERMS="True"
FEATURES=SQLENGINE
INSTANCENAME="SQLEXPRESS"
SECURITYMODE=SQL
SAPWD="{self.config.get('sa_password', 'AthalaSIEM@123')}"
SQLSYSADMINACCOUNTS="{os.getenv('USERNAME')}"
INSTALLSQLDATADIR="{self.install_path / 'Database'}"
TCPENABLED=1
NPENABLED=0
"""
        config_file = self.temp_path / "sql_config.ini"
        config_file.write_text(config_content)
        return config_file

    def _check_sql_server_installed(self) -> bool:
        """Check if SQL Server is already installed"""
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Microsoft SQL Server",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY
            ) as key:
                return True
        except WindowsError:
            return False