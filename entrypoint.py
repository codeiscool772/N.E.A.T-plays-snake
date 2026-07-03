from __future__ import annotations

import sys
from multiprocessing import freeze_support

from neat_snake.gui_wizard import GuiWizard


def main() -> None:
    # Required for PyInstaller + multiprocessing on Windows to prevent child processes
    # from re-running the packaged entrypoint (which is what creates duplicate GUI windows).
    freeze_support()

    # EXE supports two modes:
    #  1) GUI wizard (default)
    #  2) CLI forwarding to neat_snake.main (when launched by the GUI via subprocess)
    if "--run-main" in sys.argv:
        # Forward remaining args after --run-main
        args = sys.argv[sys.argv.index("--run-main") + 1 :]
        from neat_snake.main import main as neat_main

        # Replace argv for neat_snake.main parsing
        sys.argv = [sys.argv[0], *args]
        neat_main()
        return

    GuiWizard().mainloop()


if __name__ == "__main__":
    main()
