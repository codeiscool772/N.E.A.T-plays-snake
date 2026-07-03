# Neat Snake EXE (PyInstaller) + GUI Start Wizard

This repo contains a small GUI wizard (`neat_snake/gui_wizard.py`) that lets you tweak key settings in `config.py`-style variables for a single run, then start:

- Smoke test (quick render)
- Train only
- Train then render best checkpoint
- Render best checkpoint (from `checkpoints/best_genome.json`)

## 1) Requirements
- Python 3.x
- PyInstaller

Install:
```bash
pip install pyinstaller
```

## 2) Build the EXE
From the repo root (same folder where `neat_snake/pyinstaller.spec` exists):

```bash
pyinstaller neat_snake/pyinstaller.spec --noconfirm
```

Output:
- `dist/NeatSnake.exe` (Windows)

## 3) Run
```bash
dist\NeatSnake.exe
```

## Notes
- The wizard updates values in-memory for the current session (it does not rewrite `neat_snake/config.py` on disk).
- Training creates/updates:
  - `checkpoints/best_genome.json`
