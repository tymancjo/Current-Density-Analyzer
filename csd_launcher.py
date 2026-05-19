#!/usr/bin/env python3
"""CSD Launcher — simple GUI front-end for cliCSD.py

Requires customtkinter (already in requirements.txt).
Run from the project root:
    python csd_launcher.py
"""

import os
import re
import sys
import queue
import threading
import tempfile
import subprocess
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import customtkinter as ctk

# ── theme ──────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SCRIPT_DIR = Path(__file__).parent
CLI_SCRIPT  = SCRIPT_DIR / "cliCSD.py"
PYTHON_EXE  = sys.executable

# Colour palette — mirrors webCSD's dark theme
C_BG2  = "#0d1117"
C_BG   = "#161b22"
C_PNL  = "#1e2736"
C_PNL2 = "#0f172a"
C_ACC  = "#3b82f6"
C_TXT  = "#e2e8f0"
C_DIM  = "#64748b"
C_OK   = "#22c55e"
C_ERR  = "#ef4444"
C_WARN = "#f59e0b"

# Verbose log keyword → progress fraction (order matters: first match wins)
_PROGRESS = [
    (r"reading|loading|file type",           .05),
    (r"splitting|subdivid",                  .15),
    (r"adjusted geometry|adjusted",          .25),
    (r"building|matrix|impedance",           .40),
    (r"solving|lu |factor",                  .60),
    (r"total power|phase result",            .75),
    (r"bar|conductor|detect",                .88),
    (r"done|complete|finished",              .95),
]


def _parse_ic_params(text: str) -> dict:
    """Quick regex scan for the 5 analysis-param directives in .ic text."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        for key in ("freq", "length", "temp", "htc", "cellsize"):
            m = re.match(rf"^{key}\(([^)]+)\)", line, re.IGNORECASE)
            if m:
                try:
                    out[key] = float(m.group(1))
                except ValueError:
                    pass
    return out


# ══════════════════════════════════════════════════════════════════════
class CSDLauncher(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("CSD Launcher")
        self.geometry("1220x800")
        self.minsize(900, 620)

        self._proc    = None          # active subprocess
        self._q       = queue.Queue() # stdout lines → main thread
        self._prog    = 0.0           # current progress fraction
        self._tmpfile = None          # temp .ic path

        self._build_ui()
        self._poll()                  # start queue polling loop

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)   # editor + params
        self.grid_rowconfigure(2, weight=2)   # log + results
        self._build_toolbar()
        self._build_main()
        self._build_bottom()

    # ── toolbar ───────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = ctk.CTkFrame(self, height=46, corner_radius=0, fg_color=C_PNL2)
        tb.grid(row=0, column=0, sticky="ew")
        tb.grid_columnconfigure(99, weight=1)

        ctk.CTkLabel(tb, text="⚡ CSD Launcher",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=C_TXT
                     ).grid(row=0, column=0, padx=14, pady=10)

        buttons = [
            ("⤵ Load .ic", self._load, {}),
            ("⬇ Save .ic", self._save, {}),
            ("✕  Clear",   self._clear,
             {"fg_color": "transparent", "border_width": 1}),
        ]
        for col, (label, cmd, kw) in enumerate(buttons, start=1):
            ctk.CTkButton(tb, text=label, width=105, command=cmd,
                          **kw).grid(row=0, column=col, padx=4, pady=8)

    # ── main pane (editor left, params right) ─────────────────────────

    def _build_main(self):
        mf = ctk.CTkFrame(self, corner_radius=0, fg_color=C_BG)
        mf.grid(row=1, column=0, sticky="nsew")
        mf.grid_columnconfigure(0, weight=1)
        mf.grid_columnconfigure(1, weight=0, minsize=295)
        mf.grid_rowconfigure(0, weight=1)

        self._build_editor(mf)
        self._build_params(mf)

    def _build_editor(self, parent):
        ef = ctk.CTkFrame(parent, corner_radius=6, fg_color=C_PNL)
        ef.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        ef.grid_rowconfigure(1, weight=1)
        ef.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(ef, fg_color="transparent", height=28)
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(7, 0))

        ctk.CTkLabel(hdr, text=".ic Code",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C_TXT).pack(side="left")

        self._hint = ctk.CTkLabel(hdr,
            text="Load a file or paste/type the .ic geometry code",
            font=ctk.CTkFont(size=10), text_color=C_DIM)
        self._hint.pack(side="left", padx=8)

        self._editor = ctk.CTkTextbox(
            ef, font=ctk.CTkFont(family="Courier", size=12), wrap="none")
        self._editor.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 6))
        self._editor.bind("<KeyRelease>", lambda _e: self._sync_params())

    def _build_params(self, parent):
        pf = ctk.CTkFrame(parent, corner_radius=6, fg_color=C_PNL, width=295)
        pf.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        pf.grid_propagate(False)
        pf.grid_columnconfigure(1, weight=1)

        r = 0
        ctk.CTkLabel(pf, text="Analysis Parameters",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C_TXT
                     ).grid(row=r, column=0, columnspan=2,
                            padx=12, pady=(10, 8), sticky="w")

        # parameter fields  ── key, label, default, tooltip
        self._pv = {}
        fields = [
            ("cellsize", "Cell size  (mm)",   "2",    "Grid resolution; smaller = slower"),
            ("freq",     "Frequency  (Hz)",   "50",   "AC frequency"),
            ("temp",     "Temp.  (°C)",       "130",  "Conductor temperature"),
            ("length",   "Length  (mm)",      "1000", "Busbar length for losses & cooling"),
            ("htc",      "HTC  (W/m²K)",      "5",    "Heat-transfer coefficient; 5=natural air"),
            ("current",  "Fallback I  (A)",   "1000", "Used only when .ic has no current() lines"),
        ]
        for key, label, default, tip in fields:
            r += 1
            ctk.CTkLabel(pf, text=label,
                         font=ctk.CTkFont(size=11),
                         text_color=C_DIM, anchor="w"
                         ).grid(row=r, column=0, padx=(12, 4), pady=3, sticky="w")
            sv = ctk.StringVar(value=default)
            self._pv[key] = sv
            entry = ctk.CTkEntry(pf, textvariable=sv, width=90,
                                 font=ctk.CTkFont(size=11))
            entry.grid(row=r, column=1, padx=(0, 12), pady=3, sticky="e")

        # separator
        r += 1
        ctk.CTkFrame(pf, height=1, fg_color=C_DIM
                     ).grid(row=r, column=0, columnspan=2,
                            sticky="ew", padx=12, pady=(10, 6))

        # option checkboxes
        r += 1
        self._chk_bars = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(pf, text="Detect conductors  (-b)",
                        variable=self._chk_bars,
                        font=ctk.CTkFont(size=11)
                        ).grid(row=r, column=0, columnspan=2,
                               padx=12, pady=3, sticky="w")

        r += 1
        self._chk_plot = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(pf, text="Show result plot  (-r)",
                        variable=self._chk_plot,
                        font=ctk.CTkFont(size=11)
                        ).grid(row=r, column=0, columnspan=2,
                               padx=12, pady=3, sticky="w")

        # separator
        r += 1
        ctk.CTkFrame(pf, height=1, fg_color=C_DIM
                     ).grid(row=r, column=0, columnspan=2,
                            sticky="ew", padx=12, pady=(10, 6))

        # Run / Stop buttons
        r += 1
        bf = ctk.CTkFrame(pf, fg_color="transparent")
        bf.grid(row=r, column=0, columnspan=2, padx=10, pady=4, sticky="ew")
        bf.grid_columnconfigure((0, 1), weight=1)

        self._run_btn = ctk.CTkButton(
            bf, text="▶  Run",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=C_ACC, hover_color="#2563eb",
            command=self._run)
        self._run_btn.grid(row=0, column=0, padx=(0, 3), sticky="ew")

        self._stop_btn = ctk.CTkButton(
            bf, text="■  Stop",
            fg_color="#7f1d1d", hover_color="#991b1b",
            state="disabled", command=self._stop)
        self._stop_btn.grid(row=0, column=1, padx=(3, 0), sticky="ew")

        # progress bar + label
        r += 1
        self._pbar = ctk.CTkProgressBar(pf)
        self._pbar.grid(row=r, column=0, columnspan=2,
                        padx=12, pady=(10, 2), sticky="ew")
        self._pbar.set(0)

        r += 1
        self._plbl = ctk.CTkLabel(pf, text="Ready",
                                  font=ctk.CTkFont(size=10),
                                  text_color=C_DIM)
        self._plbl.grid(row=r, column=0, columnspan=2,
                        padx=12, pady=(0, 14))

    # ── bottom pane (log | results tabs) ──────────────────────────────

    def _build_bottom(self):
        bf = ctk.CTkFrame(self, corner_radius=0, fg_color=C_BG)
        bf.grid(row=2, column=0, sticky="nsew")
        bf.grid_columnconfigure(0, weight=1)
        bf.grid_rowconfigure(0, weight=1)

        tabs = ctk.CTkTabview(bf, fg_color=C_PNL)
        tabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=(0, 8))
        tabs.grid_columnconfigure(0, weight=1)
        tabs.grid_rowconfigure(0, weight=1)

        for title, attr in (("Log", "_log"), ("Results", "_res")):
            tab = tabs.add(title)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

            txt = tk.Text(tab,
                          bg=C_BG2, fg=C_TXT,
                          font=("Courier", 10),
                          wrap="none", relief="flat",
                          state="disabled",
                          selectbackground="#334155",
                          insertbackground=C_TXT)
            txt.grid(row=0, column=0, sticky="nsew")

            sb = tk.Scrollbar(tab, orient="vertical",
                              command=txt.yview,
                              bg=C_PNL, troughcolor=C_BG2,
                              activebackground=C_DIM, relief="flat")
            sb.grid(row=0, column=1, sticky="ns")
            txt.configure(yscrollcommand=sb.set)

            # shared colour tags
            txt.tag_config("err",  foreground=C_ERR)
            txt.tag_config("warn", foreground=C_WARN)
            txt.tag_config("ok",   foreground=C_OK)
            txt.tag_config("dim",  foreground=C_DIM)
            txt.tag_config("head", foreground=C_ACC,
                           font=("Courier", 10, "bold"))
            txt.tag_config("val",  foreground=C_OK)

            setattr(self, attr, txt)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _fval(self, key: str, fallback: float = 0.0) -> float:
        try:
            return float(self._pv[key].get())
        except (ValueError, KeyError):
            return fallback

    def _tw(self, widget: tk.Text, text: str, tag: str = ""):
        """Append text to a read-only tk.Text widget."""
        widget.configure(state="normal")
        if tag:
            widget.insert("end", text, tag)
        else:
            widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _tc(self, widget: tk.Text):
        """Clear a tk.Text widget."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.configure(state="disabled")

    def _set_progress(self, frac: float, label: str = ""):
        self._prog = max(self._prog, frac)
        self._pbar.set(self._prog)
        if label:
            s = label.strip()
            self._plbl.configure(text=s[:46] + "…" if len(s) > 46 else s)

    def _reset_progress(self):
        self._prog = 0.0
        self._pbar.set(0)
        self._plbl.configure(text="Starting…")

    # ──────────────────────────────────────────────────────────────────
    # File I/O
    # ──────────────────────────────────────────────────────────────────

    def _init_dir(self) -> str:
        d = SCRIPT_DIR / "input"
        return str(d) if d.is_dir() else str(SCRIPT_DIR)

    def _load(self):
        path = filedialog.askopenfilename(
            title="Load .ic file",
            filetypes=[("IC geometry", "*.ic *.txt"), ("All files", "*.*")],
            initialdir=self._init_dir())
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            messagebox.showerror("Load error", str(e))
            return
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", text)
        self._hint.configure(text=os.path.basename(path))
        self._sync_params()

    def _save(self):
        path = filedialog.asksaveasfilename(
            title="Save .ic file",
            defaultextension=".ic",
            filetypes=[("IC geometry", "*.ic"), ("All files", "*.*")],
            initialdir=self._init_dir())
        if not path:
            return
        try:
            Path(path).write_text(
                self._editor.get("1.0", "end").rstrip(), encoding="utf-8")
            self._hint.configure(text=os.path.basename(path))
        except OSError as e:
            messagebox.showerror("Save error", str(e))

    def _clear(self):
        self._editor.delete("1.0", "end")
        self._hint.configure(
            text="Load a file or paste/type the .ic geometry code")

    # ──────────────────────────────────────────────────────────────────
    # Param ↔ IC sync
    # ──────────────────────────────────────────────────────────────────

    def _sync_params(self):
        """Read analysis params from the IC text and update the GUI fields."""
        text = self._editor.get("1.0", "end")
        for key, val in _parse_ic_params(text).items():
            if key in self._pv:
                self._pv[key].set(str(val))

    # ──────────────────────────────────────────────────────────────────
    # Run / Stop
    # ──────────────────────────────────────────────────────────────────

    def _run(self):
        ic_text = self._editor.get("1.0", "end").strip()
        if not ic_text:
            messagebox.showwarning("No geometry",
                "Load or paste .ic code before running.")
            return

        # Write IC code to a temporary file
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".ic", delete=False, encoding="utf-8")
            tmp.write(ic_text)
            tmp.close()
            self._tmpfile = tmp.name
        except OSError as e:
            messagebox.showerror("Temp file error", str(e))
            return

        # Build command — GUI fields always override IC-file values
        cmd = [
            PYTHON_EXE, str(CLI_SCRIPT),
            "-s",   str(self._fval("cellsize",  2.0)),
            "-f",   str(self._fval("freq",      50.0)),
            "-T",   str(self._fval("temp",     130.0)),
            "-l",   str(self._fval("length",  1000.0)),
            "-htc", str(self._fval("htc",        5.0)),
            "-v",   # always on so we can parse progress keywords
        ]
        if self._chk_bars.get():
            cmd.append("-b")
        if self._chk_plot.get():
            cmd.append("-r")
        cmd += [self._tmpfile, str(self._fval("current", 1000.0))]

        # Reset display
        self._tc(self._log)
        self._tc(self._res)
        self._reset_progress()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

        # Show the command (abbreviated)
        short_cmd = " ".join(
            os.path.basename(c) if c.endswith(".py") else c for c in cmd)
        self._tw(self._log, f"$ {short_cmd}\n\n", "dim")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(SCRIPT_DIR))
        except OSError as e:
            messagebox.showerror("Launch error", str(e))
            self._finish(False)
            return

        threading.Thread(
            target=self._reader, args=(self._proc,), daemon=True).start()

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._tw(self._log, "\n[Stopped by user]\n", "warn")
        self._finish(False)

    def _finish(self, success: bool):
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        if success:
            self._prog = 1.0
            self._pbar.set(1.0)
            self._plbl.configure(text="Done ✓")
        else:
            self._plbl.configure(text="Stopped")
        if self._tmpfile:
            try:
                os.unlink(self._tmpfile)
            except OSError:
                pass
            self._tmpfile = None

    # ──────────────────────────────────────────────────────────────────
    # Subprocess I/O (background thread + queue)
    # ──────────────────────────────────────────────────────────────────

    def _reader(self, proc: subprocess.Popen):
        """Background thread: read stdout line-by-line → queue."""
        buf = []
        for line in proc.stdout:
            buf.append(line)
            self._q.put(("line", line))
        proc.wait()
        self._q.put(("done", proc.returncode, "".join(buf)))

    def _poll(self):
        """Main-thread timer: drain queue and update widgets."""
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "line":
                    self._on_line(msg[1])
                else:   # "done"
                    rc, full_output = msg[1], msg[2]
                    self._finish(rc == 0)
                    if rc != 0:
                        self._tw(self._log,
                            f"\n[Process exited with code {rc}]\n", "err")
                    self._render_results(full_output)
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _on_line(self, line: str):
        """Classify one stdout line, append to log, advance progress."""
        ll = line.lower()

        # pick colour tag
        if re.search(r"error|traceback|exception", ll):
            tag = "err"
        elif "warning" in ll:
            tag = "warn"
        elif re.search(r"total power|done|= \d", ll):
            tag = "ok"
        else:
            tag = ""

        self._tw(self._log, line, tag)

        # advance progress bar
        for pattern, frac in _PROGRESS:
            if re.search(pattern, ll):
                if frac > self._prog:
                    self._set_progress(frac, line.strip())
                break

    # ──────────────────────────────────────────────────────────────────
    # Results renderer
    # ──────────────────────────────────────────────────────────────────

    def _render_results(self, output: str):
        """Extract the result section from full output and display it."""
        lines = output.splitlines(keepends=True)

        # Skip verbose setup preamble; start from first result-looking line
        result_lines = []
        started = False
        for ln in lines:
            ll = ln.lower()
            if not started and re.search(
                    r"phase\s*\d|total power|loss|result|bar\s*\d|===", ll):
                started = True
            if started:
                result_lines.append(ln)

        if not result_lines:
            result_lines = lines   # fallback: show everything

        self._res.configure(state="normal")
        self._res.delete("1.0", "end")
        for ln in result_lines:
            ll = ln.lower()
            if re.search(r"phase\s*\d|===|---|\|\s*phase|bar\s*\d", ll):
                tag = "head"
            elif re.search(r"\d+\.\d{2,}", ln):
                tag = "val"
            elif not ln.strip() or ln.strip().startswith(("#", "=")):
                tag = "dim"
            else:
                tag = ""
            self._res.insert("end", ln, tag)
        self._res.see("1.0")
        self._res.configure(state="disabled")


# ── entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CSDLauncher()
    app.mainloop()
