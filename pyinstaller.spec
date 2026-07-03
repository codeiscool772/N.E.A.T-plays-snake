# PyInstaller spec for neat_snake (EXE packaging)
# Build with: pyinstaller neat_snake/pyinstaller.spec --noconfirm

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis(
    ['entrypoint.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=collect_submodules('neat_snake'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='NeatSnake',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
