import PyInstaller.__main__
import os
import shutil

def build_installer():
    # Hapus build directory jika ada
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")

    PyInstaller.__main__.run([
        'installer/gui/windows_installer.py',
        '--name=AthalaSIEM_Installer',
        '--onefile',
        '--windowed',
        '--icon=assets/icon.ico',
        '--add-data=installer/gui/styles.qss;styles',
        '--add-data=config/config.yaml;config',
        '--hidden-import=win32timezone'
    ])

if __name__ == "__main__":
    build_installer()