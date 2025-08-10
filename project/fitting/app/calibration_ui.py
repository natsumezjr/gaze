"""Tkinter-based calibration UI (placeholder).

This module provides a minimal window skeleton to guide user fixation
on predefined points and visualize gaze overlays. Implementation is kept
as placeholders for later integration with core algorithms.
"""

import tkinter as tk


class CalibrationApp:
    def __init__(self, window_title: str = "Gaze Calibration") -> None:
        self.root = tk.Tk()
        self.root.title(window_title)
        self.canvas = tk.Canvas(self.root, width=960, height=540, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._bind_events()

    def _bind_events(self) -> None:
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def run(self) -> None:
        self.root.mainloop()


