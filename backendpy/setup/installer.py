import asyncio
from pathlib import Path
from setup.configuration_wizard import SetupWizard
from database.connection import init_db
import shutil
import sys

async def install():
    """Run installation process"""
    print("=== AthalaSIEM Installation ===")
    
    # Create required directories
    dirs = ["logs", "certs", "config", "data"]
    for dir in dirs:
        Path(dir).mkdir(exist_ok=True)
    
    # Run configuration wizard
    wizard = SetupWizard()
    config = await wizard.run_initial_setup()
    
    # Initialize database
    print("\nInitializing database...")
    await init_db()
    
    # Generate certificates if needed
    if config["ssl"]["enabled"] and config["ssl"]["type"] == "self-signed":
        print("\nGenerating self-signed SSL certificate...")
        # Certificate generation code here
        
    print("\nInstallation completed successfully!")
    print("\nYou can now start the server with: python main.py")

if __name__ == "__main__":
    asyncio.run(install()) 