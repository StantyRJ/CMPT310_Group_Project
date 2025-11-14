#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import random

# ---------------------------------------------------------------
# Grid sizes
DISPLAY_W, DISPLAY_H = 64, 64
HR_SCALE = 3
HR_W, HR_H = DISPLAY_W * HR_SCALE, DISPLAY_H * HR_SCALE

PIXEL_SCALE = 8
BRUSH_SIZE = 9

DRAW_COLOR = (0, 0, 0)
ERASE_COLOR = (255, 255, 255)

# ---------------------------------------------------------------
# Character classes (62)
CLASSES = (
    [chr(ord("a") + i) for i in range(26)] +
    [chr(ord("A") + i) for i in range(26)] +
    [str(i) for i in range(10)]
)

# ---------------------------------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------
# Pixel Canvas
class PixelCanvas:
    def __init__(self, parent_frame):

        # ---- High-resolution drawing grid ----
        self.hr_pixels = [
            [ERASE_COLOR for _ in range(HR_W)] for __ in range(HR_H)
        ]

        # ---- Visible canvas (64x64) ----
        self.canvas = tk.Canvas(
            parent_frame,
            width=DISPLAY_W * PIXEL_SCALE,
            height=DISPLAY_H * PIXEL_SCALE,
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nw")

        parent_frame.rowconfigure(0, weight=0)
        parent_frame.columnconfigure(0, weight=0)

        self.rects = [[None] * DISPLAY_W for _ in range(DISPLAY_H)]
        self.display_pixels = [["#000000"] * DISPLAY_W for _ in range(DISPLAY_H)]

        self._create_rects()

        # Mouse drawing
        self.canvas.bind("<Button-1>", self.on_draw)
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<Button-3>", self.on_erase)
        self.canvas.bind("<B3-Motion>", self.on_erase)

        self.update_display()

    # -------------------------------------------------------------
    def _create_rects(self):
        s = PIXEL_SCALE
        for y in range(DISPLAY_H):
            for x in range(DISPLAY_W):
                x0 = x * s
                y0 = y * s
                rect = self.canvas.create_rectangle(
                    x0, y0, x0 + s, y0 + s,
                    outline="",
                    fill="#000000"
                )
                self.rects[y][x] = rect

    # -------------------------------------------------------------
    def event_to_hr(self, event):
        # Convert screen px → display px → highres px
        disp_x = event.x / PIXEL_SCALE
        disp_y = event.y / PIXEL_SCALE

        hx = int(disp_x * HR_SCALE)
        hy = int(disp_y * HR_SCALE)

        return clamp(hx, 0, HR_W - 1), clamp(hy, 0, HR_H - 1)

    # -------------------------------------------------------------
    def paint_hr_area(self, cx, cy, color):
        half = BRUSH_SIZE // 2

        for y in range(cy - half, cy + half + 1):
            if 0 <= y < HR_H:
                for x in range(cx - half, cx + half + 1):
                    if 0 <= x < HR_W:
                        self.hr_pixels[y][x] = color

        self.update_display()

    # -------------------------------------------------------------
    def update_display(self):
        for dy in range(DISPLAY_H):
            for dx in range(DISPLAY_W):
                r = g = b = 0
                for j in range(HR_SCALE):
                    for i in range(HR_SCALE):
                        px = dx * HR_SCALE + i
                        py = dy * HR_SCALE + j
                        cr, cg, cb = self.hr_pixels[py][px]
                        r += cr; g += cg; b += cb

                r //= HR_SCALE * HR_SCALE
                g //= HR_SCALE * HR_SCALE
                b //= HR_SCALE * HR_SCALE
                hexcol = f"#{r:02x}{g:02x}{b:02x}"

                if self.display_pixels[dy][dx] != hexcol:
                    self.display_pixels[dy][dx] = hexcol
                    self.canvas.itemconfig(self.rects[dy][dx], fill=hexcol)

    # -------------------------------------------------------------
    def on_draw(self, event):
        hx, hy = self.event_to_hr(event)
        self.paint_hr_area(hx, hy, DRAW_COLOR)

    def on_erase(self, event):
        hx, hy = self.event_to_hr(event)
        self.paint_hr_area(hx, hy, ERASE_COLOR)


# ---------------------------------------------------------------
# Confidence Table
class ConfidenceTable:
    def __init__(self, parent_frame):

        parent_frame.rowconfigure(0, weight=1)
        parent_frame.columnconfigure(0, weight=1)

        columns = ("class", "svm", "cnn", "knn")

        self.tree = ttk.Treeview(
            parent_frame,
            columns=columns,
            show="headings"
        )
        self.tree.grid(row=0, column=0, sticky="nsw")

        # Scrollbar
        vsb = ttk.Scrollbar(parent_frame, orient="vertical",
                            command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        # Column names
        self.tree.heading("class", text="Class")
        self.tree.heading("svm", text="SVM")
        self.tree.heading("cnn", text="CNN")
        self.tree.heading("knn", text="KNN")

        # Make columns stretch
        for col in columns:
            self.tree.column(col, width=80, stretch=True)

        # Insert rows
        self.rows = []
        for c in CLASSES:
            row_id = self.tree.insert("", "end", values=(c, "0.00", "0.00", "0.00"))
            self.rows.append(row_id)

    # -----------------------------------------------------------
    def update_model_confidence(self, result_dict):
        svm = result_dict.get("svm", [0]*62)
        cnn = result_dict.get("cnn", [0]*62)
        knn = result_dict.get("knn", [0]*62)

        for i, row_id in enumerate(self.rows):
            self.tree.item(row_id, values=(
                CLASSES[i],
                f"{svm[i]:.4f}",
                f"{cnn[i]:.4f}",
                f"{knn[i]:.4f}",
            ))


# ---------------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Model Playground")

    # Left side: canvas
    left = tk.Frame(root)
    left.pack(side="left", padx=10, pady=10)

    canvas = PixelCanvas(left)

    # Right side: table
    right = tk.Frame(root)
    right.pack(side="left", padx=10, pady=10, fill="both", expand=True)

    table = ConfidenceTable(right)

    # Randomize scores (press 'r')
    def randomize_scores(event=None):
        d = {
            "svm": [random.random() for _ in range(62)],
            "cnn": [random.random() for _ in range(62)],
            "knn": [random.random() for _ in range(62)],
        }
        table.update_model_confidence(d)

    root.bind("r", randomize_scores)

    root.mainloop()
