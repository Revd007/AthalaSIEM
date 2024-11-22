import os
import shutil
import json
import hashlib
from datetime import datetime
from pathlib import Path
import logging

class ReleasePackager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.release_dir = Path("release")
        self.dist_dir = Path("dist")
        self.docs_dir = Path("docs")
        self.version = "1.0.0"  # Baca dari config atau version.txt
        
    def package_release(self):
        """Create release package"""
        try:
            # Build installer terlebih dahulu
            from build_installer import InstallerBuilder
            builder = InstallerBuilder()
            builder.build()
            
            # Lanjutkan dengan packaging
            self._create_release_dirs()
            self._copy_installer()
            self._copy_docs()
            checksums = self._generate_checksums()
            release_info = self._create_release_info(checksums)
            self._create_release_archive()
            
            self.logger.info("Release package created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating release package: {e}")
            return False
            
    def _create_release_dirs(self):
        """Create necessary directories"""
        dirs = [
            self.release_dir,
            self.release_dir / "bin",
            self.release_dir / "docs",
            self.release_dir / "config"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _copy_installer(self):
        """Copy installer and dependencies"""
        installer_path = self.dist_dir / "AthalaSIEM_Installer.exe"
        if installer_path.exists():
            shutil.copy2(installer_path, self.release_dir / "bin")
        else:
            raise FileNotFoundError("Installer not found. Run build_installer.py first")
            
    def _copy_docs(self):
        """Copy documentation files"""
        doc_files = [
            "installation.md",
            "user_guide.md",
            "api_reference.md",
            "troubleshooting.md"
        ]
        
        for doc in doc_files:
            src = self.docs_dir / doc
            if src.exists():
                shutil.copy2(src, self.release_dir / "docs")
                
        # Copy license and readme
        for file in ["LICENSE", "README.md"]:
            if Path(file).exists():
                shutil.copy2(file, self.release_dir)
                
    def _generate_checksums(self) -> dict:
        """Generate checksums for release files"""
        checksums = {}
        
        for file in self.release_dir.rglob("*"):
            if file.is_file():
                with open(file, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    checksums[str(file.relative_to(self.release_dir))] = file_hash
                    
        # Save checksums file
        checksum_file = self.release_dir / "checksums.json"
        with open(checksum_file, "w") as f:
            json.dump(checksums, f, indent=2)
            
        return checksums
        
    def _create_release_info(self, checksums: dict) -> dict:
        """Create release information file"""
        info = {
            "version": self.version,
            "release_date": datetime.now().isoformat(),
            "files": checksums,
            "requirements": {
                "os": "Windows 10/11",
                "python": ">=3.8",
                "ram": "8GB minimum",
                "disk": "10GB minimum"
            }
        }
        
        # Save release info
        info_file = self.release_dir / "release_info.json"
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)
            
        return info
        
    def _create_release_archive(self):
        """Create ZIP archive of release"""
        release_name = f"AthalaSIEM_v{self.version}"
        shutil.make_archive(
            release_name,
            "zip",
            self.release_dir
        )
        
        # Move to release directory
        if Path(f"{release_name}.zip").exists():
            shutil.move(
                f"{release_name}.zip",
                self.release_dir / f"{release_name}.zip"
            )

def main():
    logging.basicConfig(level=logging.INFO)
    packager = ReleasePackager()
    packager.package_release()

if __name__ == "__main__":
    main()