from typing import Dict, Any, Optional
import subprocess
import tempfile
import logging
from pathlib import Path
import asyncio
import os
import pyodbc

class SQLInstallationManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.download_url = "https://go.microsoft.com/fwlink/?linkid=866658"
        
    async def install_sql_express(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Install SQL Server Express"""
        try:
            # Download installer
            if progress_callback:
                progress_callback("Downloading SQL Server Express...", 10)
            
            installer_path = await self._download_installer()
            
            # Create configuration file
            if progress_callback:
                progress_callback("Preparing installation...", 30)
                
            config_file = self._create_config_file()
            
            # Run installation
            if progress_callback:
                progress_callback("Installing SQL Server Express...", 50)
                
            result = await self._run_installation(installer_path, config_file)
            
            # Verify installation
            if progress_callback:
                progress_callback("Verifying installation...", 90)
                
            verification = await self._verify_installation()
            
            return {
                'status': 'success' if verification['success'] else 'error',
                'details': verification
            }
            
        except Exception as e:
            self.logger.error(f"SQL Express installation failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _create_config_file(self) -> Path:
        """Create SQL Server configuration file"""
        config_content = f"""
[OPTIONS]
IACCEPTSQLSERVERLICENSETERMS="True"
FEATURES=SQLENGINE,CONN
INSTANCENAME="SQLEXPRESS"
INSTANCEID="SQLEXPRESS"
SECURITYMODE="SQL"
SAPWD="{self.config.get('sa_password', 'AthalaSIEM@123')}"
SQLSYSADMINACCOUNTS="{os.getenv('USERNAME')}"
SQLCOLLATION="SQL_Latin1_General_CP1_CI_AS"
TCPENABLED=1
NPENABLED=1
BROWSERSVCSTARTUPTYPE="Automatic"
ERRORREPORTING=0
SQMREPORTING=0
FILESTREAMLEVEL=0
"""
        config_path = self.temp_dir / "sql_config.ini"
        config_path.write_text(config_content)
        return config_path

    async def _verify_installation(self) -> Dict[str, Any]:
        """Verify SQL Server installation"""
        try:
            # Check service status
            service_check = subprocess.run(
                ["sc", "query", "MSSQL$SQLEXPRESS"],
                capture_output=True
            )
            
            # Try connection
            conn_str = (
                "Driver={SQL Server};"
                "Server=.\\SQLEXPRESS;"
                f"UID=sa;PWD={self.config['sa_password']}"
            )
            
            with pyodbc.connect(conn_str, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION")
                version = cursor.fetchone()[0]
                
            return {
                'success': True,
                'service_running': service_check.returncode == 0,
                'version': version
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _download_installer(self) -> Path:
        """Download SQL Server Express installer"""
        try:
            # Download installer
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            
            installer_path = self.temp_dir / "SQLServerExpress.exe"
            with open(installer_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return installer_path
            
        except Exception as e:
            self.logger.error(f"Failed to download SQL Server Express installer: {e}")
            raise
            
    async def _run_installation(self, installer_path: Path, config_file: Path) -> Dict[str, Any]:
        """Run SQL Server Express installation"""
        try:
            # Run installation
            result = subprocess.run(
                [str(installer_path), "/ConfigurationFile", str(config_file)],
                capture_output=True
            )
            
            return {
                'status': 'success' if result.returncode == 0 else 'error',
                'output': result.stdout.decode("utf-8"),
                'error': result.stderr.decode("utf-8")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to run SQL Server Express installation: {e}")
            raise