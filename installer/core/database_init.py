import sqlalchemy
from sqlalchemy import create_engine
import subprocess
import os
from typing import Optional

class DatabaseInitializer:
    def __init__(self, db_type: str, install_path: str):
        self.db_type = db_type
        self.install_path = install_path
        
    def initialize(self) -> Optional[str]:
        """Initialize database and return connection string"""
        if self.db_type == "SQLite":
            return self._init_sqlite()
        else:
            return self._init_sqlserver()
            
    def _init_sqlite(self) -> str:
        """Initialize SQLite database"""
        db_path = os.path.join(self.install_path, "data", "siem.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        engine = create_engine(f"sqlite:///{db_path}")
        
        # Import and create all models
        from database.models import Base
        Base.metadata.create_all(engine)
        
        return f"sqlite:///{db_path}"
        
    def _init_sqlserver(self) -> Optional[str]:
        """Initialize SQL Server database"""
        try:
            # Check if SQL Server Express is installed
            subprocess.run(
                ["sqlcmd", "-?"],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            # Install SQL Server Express
            self._install_sqlserver_express()
            
        # Create database
        conn_str = "mssql+pyodbc://./SQLEXPRESS/siem_db?driver=ODBC+Driver+17+for+SQL+Server"
        engine = create_engine(conn_str)
        
        # Import and create all models
        from database.models import Base
        Base.metadata.create_all(engine)
        
        return conn_str
        
    def _install_sqlserver_express(self):
        """Install SQL Server Express"""
        # Download SQL Server Express installer
        subprocess.run([
            "curl", "-o", "SQLEXPR.exe",
            "https://go.microsoft.com/fwlink/?linkid=866658"
        ], check=True)
        
        # Install SQL Server Express silently
        subprocess.run([
            "SQLEXPR.exe",
            "/IACCEPTSQLSERVERLICENSETERMS",
            "/Q",
            "/HIDECONSOLE"
        ], check=True)