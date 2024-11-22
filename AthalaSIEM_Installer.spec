# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['installer\\gui\\windows_installer.py'],
    pathex=[],
    binaries=[],
    datas=[('E:\\AthalaSIEM\\AthalaSIEM\\installer/gui/styles.qss', 'styles'), ('E:\\AthalaSIEM\\AthalaSIEM\\config.yaml', 'config')],
    hiddenimports=['win32timezone'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
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
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\icon.ico'],
)
