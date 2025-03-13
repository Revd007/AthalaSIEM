import os
import shutil
from pathlib import Path

def generate_agent_installer(agent_type: str) -> Path:
    """Generate agent installer package"""
    if agent_type not in ['windows', 'linux', 'macos']:
        raise ValueError("Invalid agent type")

    # Base directory for installers
    installer_dir = Path("installers")
    installer_dir.mkdir(exist_ok=True)

    # Create temp directory for building installer
    build_dir = installer_dir / f"build_{agent_type}"
    build_dir.mkdir(exist_ok=True)

    try:
        # Copy agent files based on type
        agent_files_dir = Path(f"agent_files/{agent_type}")
        shutil.copytree(agent_files_dir, build_dir / "agent", dirs_exist_ok=True)

        # Create zip file
        output_file = installer_dir / f"siem-agent-{agent_type}.zip"
        shutil.make_archive(str(output_file.with_suffix('')), 'zip', build_dir)

        return output_file.with_suffix('.zip')

    finally:
        # Cleanup build directory
        shutil.rmtree(build_dir) 