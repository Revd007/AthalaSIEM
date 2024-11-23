import PyInstaller.__main__
import os
import shutil
import subprocess
from pathlib import Path
import sys

class InstallerBuilder:
    def __init__(self):
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.root_dir = self.script_dir.parent
        self.build_dir = self.root_dir / "build"
        self.dist_dir = self.root_dir / "dist"
        self.temp_dir = self.build_dir / "temp"
        
    def check_npm(self):
        """Check if npm is installed and accessible"""
        try:
            # Cek npm dengan full path jika di Windows
            if sys.platform == 'win32':
                npm_cmd = 'where npm'
            else:
                npm_cmd = 'which npm'
                
            result = subprocess.run(npm_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("npm not found. Please install Node.js and npm first.")
                print("Download from: https://nodejs.org/")
                sys.exit(1)
                
            # Verify npm version
            npm_version = subprocess.run(['npm', '-v'], capture_output=True, text=True)
            print(f"Found npm version: {npm_version.stdout.strip()}")
            
        except Exception as e:
            print(f"Error checking npm: {e}")
            print("Please ensure Node.js and npm are installed and in your PATH")
            sys.exit(1)
    
    def build_frontend(self):
        """Build React frontend"""
        print("Building frontend...")
        frontend_dir = self.root_dir / "frontend" / "siem-frontend"
        
        if not frontend_dir.exists():
            print(f"Frontend directory not found at: {frontend_dir}")
            print("Please ensure the frontend directory exists")
            sys.exit(1)
            
        try:
            # Check npm first
            self.check_npm()
            
            print(f"Installing frontend dependencies in: {frontend_dir}")
            # Use npm with shell=True for Windows compatibility
            subprocess.run("npm install", cwd=frontend_dir, shell=True, check=True)
            
            print("Building frontend...")
            subprocess.run("npm run build", cwd=frontend_dir, shell=True, check=True)
            
            # Verify build directory exists
            frontend_build = frontend_dir / "build"
            if not frontend_build.exists():
                print(f"Frontend build directory not found at: {frontend_build}")
                sys.exit(1)
                
            # Copy build ke temporary directory
            frontend_dest = self.temp_dir / "frontend"
            print(f"Copying frontend build to: {frontend_dest}")
            shutil.copytree(frontend_build, frontend_dest, dirs_exist_ok=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Frontend build failed: {e}")
            print(f"Command output: {e.output if hasattr(e, 'output') else 'No output'}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error building frontend: {e}")
            sys.exit(1)

    def build_backend(self):
        """Build Python backend"""
        print("Building backend...")
        backend_dir = self.root_dir / "backend"
        
        # Create spec file for backend
        backend_spec = f"""
# -*- mode: python -*-
a = Analysis(
    ['{str(backend_dir / "main.py")}'],
    pathex=['{str(backend_dir)}'],
    binaries=[],
    datas=[
        ('{str(backend_dir / "config")}', 'config'),
        ('{str(backend_dir / "ai_engine")}', 'ai_engine'),
    ],
    hiddenimports=['win32timezone', 'asyncio'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='siem_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend'
)
"""
        spec_file = self.temp_dir / "backend.spec"
        spec_file.write_text(backend_spec)
        
        # Build backend using spec file
        PyInstaller.__main__.run([
            str(spec_file),
            '--distpath', str(self.temp_dir / "backend"),
            '--workpath', str(self.build_dir / "backend_build"),
            '--noconfirm'
        ])

    def build_installer(self):
        """Build final installer"""
        print("Building installer...")
        
        # Create installer spec
        installer_spec = f"""
# -*- mode: python -*-
a = Analysis(
    ['{str(self.root_dir / "installer/gui/windows_installer.py")}'],
    pathex=['{str(self.root_dir)}'],
    binaries=[],
    datas=[
        ('{str(self.temp_dir / "frontend")}', 'frontend'),
        ('{str(self.temp_dir / "backend/backend")}', 'backend'),
        ('{str(self.root_dir / "installer/gui/styles.qss")}', 'styles'),
        ('{str(self.root_dir / "config.yaml")}', 'config'),
    ],
    hiddenimports=['win32timezone'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AthalaSIEM_Installer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{str(self.root_dir / "assets/icon.ico")}'
)
"""
        spec_file = self.temp_dir / "installer.spec"
        spec_file.write_text(installer_spec)
        
        # Build installer
        PyInstaller.__main__.run([
            str(spec_file),
            '--distpath', str(self.dist_dir),
            '--workpath', str(self.build_dir / "installer_build"),
            '--noconfirm'
        ])

    def clean(self):
        """Clean build directories"""
        print("Cleaning build directories...")
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
        
    def build(self):
        """Run complete build process"""
        try:
            self.clean()
            
            # Create necessary directories
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Build components
            self.build_frontend()
            self.build_backend()
            self.build_installer()
            
            print("Build completed successfully!")
            print(f"Installer location: {self.dist_dir / 'AthalaSIEM_Installer.exe'}")
            
        except Exception as e:
            print(f"Build failed: {str(e)}")
            raise
        finally:
            # Cleanup temporary files
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

def main():
    builder = InstallerBuilder()
    builder.build()

if __name__ == "__main__":
    main()