from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import sys
import subprocess

import neat_snake.config as config
from neat_snake.neat_brain import Genome


class GuiWizard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Neat Snake - Start Wizard (EXE)")
        self.geometry("720x520")

        self._worker_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        row = 0

        # Mode
        ttk.Label(main, text="Mode", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, sticky="w")
        row += 1
        self.mode_var = tk.StringVar(value="train_then_render")

        modes = [
            ("Train only (no render)", "train_only"),
            ("Train then render best", "train_then_render"),
            ("Render best checkpoint", "render_best"),
            ("Smoke test (quick demo)", "smoke"),
        ]
        for label, value in modes:
            ttk.Radiobutton(main, text=label, value=value, variable=self.mode_var).grid(
                row=row, column=0, sticky="w", pady=2
            )
            row += 1

        row += 1

        # Params
        params_label = ttk.Label(main, text="Settings", font=("Segoe UI", 11, "bold"))
        params_label.grid(row=row, column=0, sticky="w")
        row += 1

        form = ttk.Frame(main)
        form.grid(row=row, column=0, sticky="nsew")
        form.columnconfigure(1, weight=1)

        def add_spin(name: str, default: int, r: int, minv: int | None = None, maxv: int | None = None):
            ttk.Label(form, text=name).grid(row=r, column=0, sticky="w", pady=4)
            var = tk.IntVar(value=default)
            spin = ttk.Spinbox(
                form,
                from_=minv if minv is not None else 0,
                to=maxv if maxv is not None else 10_000_000,
                textvariable=var,
                width=10,
            )
            spin.grid(row=r, column=1, sticky="w", pady=4)
            return var

        def add_float(name: str, default: float, r: int, minv: float | None = None, maxv: float | None = None):
            ttk.Label(form, text=name).grid(row=r, column=0, sticky="w", pady=4)
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(form, textvariable=var, width=12)
            entry.grid(row=r, column=1, sticky="w", pady=4)
            return var

        r0 = 0
        self.var_gens = add_spin("Generations (train)", config.GENERATIONS, r0, minv=1)
        r0 += 1
        self.var_pop = add_spin("Population size (POP_SIZE)", config.POP_SIZE, r0, minv=1, maxv=10_000)
        r0 += 1
        self.var_parallel = add_spin("Parallel eval workers (PARALLEL_EVAL)", config.PARALLEL_EVAL, r0, minv=1, maxv=128)
        r0 += 1
        self.var_episodes = add_spin("Episodes per genome", config.EPISODES_PER_GENOME, r0, minv=1, maxv=50)
        r0 += 1

        self.var_watch_seed = add_spin("Render seed base", 123, r0, minv=0, maxv=2_000_000_000)
        r0 += 1

        self.var_render_sleep = add_float("Render sleep (sec)", config.RENDER_SLEEP_SEC, r0, minv=0.0, maxv=10.0)

        # Optional Advanced section
        adv_row_start = r0 + 1
        self.var_advanced = tk.BooleanVar(value=False)

        adv_check = ttk.Checkbutton(main, text="Advanced", variable=self.var_advanced, command=self._on_advanced_toggle)
        adv_check.grid(row=adv_row_start, column=0, sticky="w", pady=(12, 0))

        self.advanced_frame = ttk.LabelFrame(main, text="Advanced Settings", padding=12)
        self.advanced_frame.grid(row=adv_row_start + 1, column=0, sticky="nsew", pady=(6, 0))
        self.advanced_frame.grid_remove()

        # Scrollable container for advanced options (panel can be taller than window).
        self._adv_canvas = tk.Canvas(self.advanced_frame, highlightthickness=0, borderwidth=0)
        self._adv_scrollbar = ttk.Scrollbar(self.advanced_frame, orient="vertical", command=self._adv_canvas.yview)
        self._adv_canvas.configure(yscrollcommand=self._adv_scrollbar.set)

        self._adv_canvas.pack(side="left", fill="both", expand=True)
        self._adv_scrollbar.pack(side="right", fill="y")

        self._adv_inner = ttk.Frame(self._adv_canvas)
        self._adv_window_id = self._adv_canvas.create_window((0, 0), window=self._adv_inner, anchor="nw")

        def _on_adv_inner_configure(event: tk.Event) -> None:
            # Update scrollregion to include the whole inner frame.
            self._adv_canvas.configure(scrollregion=self._adv_canvas.bbox("all"))

        self._adv_inner.bind("<Configure>", _on_adv_inner_configure)

        def _on_adv_canvas_configure(event: tk.Event) -> None:
            # Make inner frame match canvas width so labels don't wrap oddly.
            self._adv_canvas.itemconfigure(self._adv_window_id, width=event.width)

        self._adv_canvas.bind("<Configure>", _on_adv_canvas_configure)

        adv_form = self._adv_inner

        def add_adv_spin(container: ttk.Frame, name: str, default: int, r: int, minv: int | None = None, maxv: int | None = None):
            ttk.Label(container, text=name).grid(row=r, column=0, sticky="w", pady=4)
            var = tk.IntVar(value=default)
            spin = ttk.Spinbox(
                container,
                from_=minv if minv is not None else -10_000_000,
                to=maxv if maxv is not None else 10_000_000,
                textvariable=var,
                width=10,
            )
            spin.grid(row=r, column=1, sticky="w", pady=4)
            return var

        def add_adv_float(container: ttk.Frame, name: str, default: float, r: int, minv: float | None = None, maxv: float | None = None):
            ttk.Label(container, text=name).grid(row=r, column=0, sticky="w", pady=4)
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(container, textvariable=var, width=12)
            if minv is not None and maxv is not None:
                # We still let the user type, but keep a bounded Spinbox-like feel by clamping on apply.
                pass
            entry.grid(row=r, column=1, sticky="w", pady=4)
            return var

        adv_form.columnconfigure(1, weight=1)

        ar = 0
        self.var_grid_w = add_adv_spin(adv_form, "Grid width (GRID_W)", config.GRID_W, ar, minv=4, maxv=200)
        ar += 1
        self.var_grid_h = add_adv_spin(adv_form, "Grid height (GRID_H)", config.GRID_H, ar, minv=4, maxv=200)
        ar += 1
        self.var_max_steps = add_adv_spin(adv_form, "Max steps (MAX_STEPS)", config.MAX_STEPS, ar, minv=10, maxv=2000)
        ar += 1

        self.var_reward_food = add_adv_float(adv_form, "Reward food (REWARD_FOOD)", config.REWARD_FOOD, ar, minv=0.0, maxv=10000.0)
        ar += 1
        self.var_step_penalty = add_adv_float(adv_form, "Step penalty (STEP_PENALTY)", config.STEP_PENALTY, ar, minv=0.0, maxv=10.0)
        ar += 1
        self.var_penalty_death = add_adv_float(adv_form, "Death penalty (PENALTY_DEATH)", config.PENALTY_DEATH, ar, minv=0.0, maxv=10000.0)
        ar += 1

        self.var_reward_closer = add_adv_float(adv_form, "Reward closer (REWARD_CLOSER)", config.REWARD_CLOSER, ar, minv=0.0, maxv=10000.0)
        ar += 1
        self.var_reward_farther = add_adv_float(adv_form, "Reward farther (REWARD_FARTHER)", config.REWARD_FARTHER, ar, minv=0.0, maxv=10000.0)
        ar += 1

        self.var_elites = add_adv_spin(adv_form, "Elites (ELITES)", config.ELITES, ar, minv=1, maxv=50)
        ar += 1
        self.var_tournament_k = add_adv_spin(adv_form, "Tournament K (TOURNAMENT_K)", config.TOURNAMENT_K, ar, minv=1, maxv=50)
        ar += 1

        # Mutation params (floats)
        self.var_add_node_prob = add_adv_float(adv_form, "Add node prob (ADD_NODE_PROB)", config.ADD_NODE_PROB, ar, minv=0.0, maxv=1.0)
        ar += 1
        self.var_add_conn_prob = add_adv_float(adv_form, "Add conn prob (ADD_CONN_PROB)", config.ADD_CONN_PROB, ar, minv=0.0, maxv=1.0)
        ar += 1
        self.var_weight_mut_prob = add_adv_float(adv_form, "Weight mut prob (WEIGHT_MUT_PROB)", config.WEIGHT_MUT_PROB, ar, minv=0.0, maxv=1.0)
        ar += 1
        self.var_weight_perturb_std = add_adv_float(adv_form, "Weight perturb std (WEIGHT_PERTURB_STD)", config.WEIGHT_PERTURB_STD, ar, minv=0.0, maxv=100.0)
        ar += 1
        self.var_weight_replace_prob = add_adv_float(adv_form, "Weight replace prob (WEIGHT_REPLACE_PROB)", config.WEIGHT_REPLACE_PROB, ar, minv=0.0, maxv=1.0)
        ar += 1
        self.var_weight_replace_range = add_adv_float(adv_form, "Weight replace range (WEIGHT_REPLACE_RANGE)", config.WEIGHT_REPLACE_RANGE, ar, minv=0.0, maxv=1000.0)
        ar += 1

        # Rendering toggle for completeness (used by README/possibly external; safe to override).
        self.var_render_text = tk.BooleanVar(value=getattr(config, "RENDER_TEXT", True))
        ttk.Checkbutton(adv_form, text="Enable turtle render text (RENDER_TEXT)", variable=self.var_render_text).grid(
            row=ar, column=0, columnspan=2, sticky="w", pady=4
        )

        row += 1
        row2 = row

        self.render_btn = ttk.Button(main, text="Start", command=self.on_start)
        self.render_btn.grid(row=row2, column=0, sticky="w", pady=(12, 0))
        row2 += 1

        self.stop_btn = ttk.Button(main, text="Stop (best-effort)", command=self.on_stop, state="disabled")
        self.stop_btn.grid(row=row2, column=0, sticky="w")
        row2 += 1

        self.status = tk.Text(main, height=10)
        self.status.grid(row=row2, column=0, sticky="nsew", pady=(12, 0))
        self.status.configure(state="disabled")

        main.rowconfigure(row2, weight=1)

    def log(self, msg: str) -> None:
        self.status.configure(state="normal")
        self.status.insert("end", msg + "\n")
        self.status.see("end")
        self.status.configure(state="disabled")

    def _on_advanced_toggle(self) -> None:
        if self.var_advanced.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    def _apply_overrides(self) -> None:
        # Update config in-memory (for the running EXE session).
        config.GENERATIONS = int(self.var_gens.get())
        config.POP_SIZE = int(self.var_pop.get())
        config.PARALLEL_EVAL = int(self.var_parallel.get())
        config.EPISODES_PER_GENOME = int(self.var_episodes.get())
        config.RENDER_SLEEP_SEC = float(self.var_render_sleep.get())

        if self.var_advanced.get():
            def clamp_int(v: int, lo: int, hi: int) -> int:
                return max(lo, min(hi, v))

            def clamp_float(v: float, lo: float, hi: float) -> float:
                return max(lo, min(hi, v))

            config.GRID_W = clamp_int(int(self.var_grid_w.get()), 4, 200)
            config.GRID_H = clamp_int(int(self.var_grid_h.get()), 4, 200)
            config.MAX_STEPS = clamp_int(int(self.var_max_steps.get()), 10, 2000)

            # Rewards/penalties: keep >= 0.0 for stability with this prototype.
            config.REWARD_FOOD = clamp_float(float(self.var_reward_food.get()), 0.0, 10000.0)
            config.STEP_PENALTY = clamp_float(float(self.var_step_penalty.get()), 0.0, 10.0)
            config.PENALTY_DEATH = clamp_float(float(self.var_penalty_death.get()), 0.0, 10000.0)

            config.REWARD_CLOSER = clamp_float(float(self.var_reward_closer.get()), 0.0, 10000.0)
            config.REWARD_FARTHER = clamp_float(float(self.var_reward_farther.get()), 0.0, 10000.0)

            config.ELITES = clamp_int(int(self.var_elites.get()), 1, 50)
            config.TOURNAMENT_K = clamp_int(int(self.var_tournament_k.get()), 1, 50)

            # Mutation probabilities: keep in [0,1]
            config.ADD_NODE_PROB = clamp_float(float(self.var_add_node_prob.get()), 0.0, 1.0)
            config.ADD_CONN_PROB = clamp_float(float(self.var_add_conn_prob.get()), 0.0, 1.0)
            config.WEIGHT_MUT_PROB = clamp_float(float(self.var_weight_mut_prob.get()), 0.0, 1.0)
            config.WEIGHT_PERTURB_STD = clamp_float(float(self.var_weight_perturb_std.get()), 0.0, 100.0)
            config.WEIGHT_REPLACE_PROB = clamp_float(float(self.var_weight_replace_prob.get()), 0.0, 1.0)
            config.WEIGHT_REPLACE_RANGE = clamp_float(float(self.var_weight_replace_range.get()), 0.0, 1000.0)

            # Optional completeness; may not affect core logic.
            config.RENDER_TEXT = bool(self.var_render_text.get())

    def on_start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Running", "A run is already in progress.")
            return

        self._stop_flag.clear()
        self._apply_overrides()

        mode = self.mode_var.get()

        def worker():
            try:
                self._set_buttons(running=True)

                self.log(f"Mode: {mode}")
                self.log(
                    f"gens={config.GENERATIONS}, pop={config.POP_SIZE}, parallel_eval={config.PARALLEL_EVAL}, episodes={config.EPISODES_PER_GENOME}"
                )

                if mode == "smoke":
                    g = Genome(rng=None, input_size=config.OBS_SIZE, output_size=4)  # type: ignore[arg-type]
                    # Smoke mode just renders a quick episode (uses internal random in Genome)
                    # If this line errors due to rng signature, training can still be used.
                    self.log("Starting smoke via subprocess...")
                    cmd = [
                        sys.executable,
                        "-m",
                        "neat_snake.entrypoint",
                        "--run-main",
                        "--smoke",
                    ]
                    # also pass render seed for determinism if main supports it later
                    subprocess.run(cmd, check=False)
                    self.log("Smoke finished.")
                    return

                if mode in ("train_only", "train_then_render"):
                    # HARD REQUIREMENT:
                    # Start should behave like "python -m neat_snake.main {selected options}"
                    # and must not run training in-process (which risks GUI re-entrancy).
                    train_args = [
                        "--gens=" + str(config.GENERATIONS),
                        "--pop=" + str(config.POP_SIZE),
                        "--parallel=" + str(config.PARALLEL_EVAL),
                        "--episodes=" + str(config.EPISODES_PER_GENOME),
                    ]

                    self.log("Training started via subprocess...")
                    cmd = [sys.executable, "-m", "neat_snake.entrypoint", "--run-main", *train_args]
                    subprocess.run(cmd, check=False)
                    self.log("Training finished.")

                    if mode == "train_then_render":
                        self.log("Rendering best via subprocess...")
                        render_cmd = [sys.executable, "-m", "neat_snake.entrypoint", "--run-main", "--render"]
                        subprocess.run(render_cmd, check=False)
                        self.log("Render finished.")
                    return

                if mode == "render_best":
                    self.log("Rendering best via subprocess...")
                    cmd = [sys.executable, "-m", "neat_snake.entrypoint", "--run-main", "--render"]
                    subprocess.run(cmd, check=False)
                    self.log("Render finished.")
                    return

                self.log(f"Unknown mode: {mode}")

            except Exception as e:
                self.log(f"ERROR: {type(e).__name__}: {e}")
                messagebox.showerror("Error", f"{type(e).__name__}: {e}")
            finally:
                self._set_buttons(running=False)

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _set_buttons(self, running: bool) -> None:
        def apply():
            self.render_btn.configure(state="disabled" if running else "normal")
            self.stop_btn.configure(state="normal" if running else "disabled")

        self.after(0, apply)

    def on_stop(self) -> None:
        # Best-effort stop: current training loop doesn't poll this flag.
        # We disable the UI; user can close the window or terminate the EXE.
        self.log("Stop requested. Current implementation is best-effort (may not interrupt instantly).")
        self._stop_flag.set()


def main() -> None:
    GuiWizard().mainloop()


if __name__ == "__main__":
    main()
