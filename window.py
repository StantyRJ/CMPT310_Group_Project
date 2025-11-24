#!/usr/bin/env python3
import os
import sys

# Ensure Tcl/Tk runtime can be found when using virtual environments or unusual installs.
# Try a few likely locations (sys.base_prefix, sys.prefix, and the common AppData path).
def _ensure_tcl_tk_env():
    # common candidate directories for init.tcl on Windows
    candidates = []
    try:
        candidates.append(os.path.join(sys.base_prefix, "tcl", "tcl8.6"))
    except Exception:
        pass
    try:
        candidates.append(os.path.join(sys.prefix, "tcl", "tcl8.6"))
    except Exception:
        pass
    try:
        candidates.append(os.path.join(sys.exec_prefix, "tcl", "tcl8.6"))
    except Exception:
        pass
    # user local AppData Python install (common on Windows installers)
    user_appdata = os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "Python")
    if os.path.isdir(user_appdata):
        # search for any Python*/tcl/tcl8.6 folders under that tree
        for entry in os.listdir(user_appdata):
            candidate = os.path.join(user_appdata, entry, "tcl", "tcl8.6")
            candidates.append(candidate)

    for c in candidates:
        if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "init.tcl")):
            os.environ.setdefault("TCL_LIBRARY", c)
            # guess tk path by replacing tcl8.6 with tk8.6 when possible
            tk_candidate = c.replace("tcl8.6", "tk8.6")
            if os.path.isdir(tk_candidate):
                os.environ.setdefault("TK_LIBRARY", tk_candidate)
            return True
    return False


# Try to set the environment variables before importing tkinter to avoid TclError
_ensure_tcl_tk_env()

import tkinter as tk
from tkinter import ttk
import random
import numpy as np

from lib.models import CNNClassifier, KNNClassifier, SVMClassifier
from lib.datasets.png_dataset import PNGDataset

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
    def __init__(self, parent_frame, on_change=None):

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

        # optional callback called whenever the visible canvas changes
        # (the UI can debounce calls if needed)
        self.on_change = on_change

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
        # notify listener that drawing changed
        if hasattr(self, "on_change") and self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                pass

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

    def get_image(self):
        """Return the current drawing as a numpy array shaped (1,1,64,64) of floats [0..1].

        The canvas stores higher-resolution RGB pixels in self.hr_pixels; we average
        the HR block for each display pixel and convert to grayscale.
        """
        arr = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.float32)
        for dy in range(DISPLAY_H):
            for dx in range(DISPLAY_W):
                r = g = b = 0.0
                for j in range(HR_SCALE):
                    for i in range(HR_SCALE):
                        px = dx * HR_SCALE + i
                        py = dy * HR_SCALE + j
                        cr, cg, cb = self.hr_pixels[py][px]
                        r += cr; g += cg; b += cb
                # average and convert to 0..1 where 0 is black, 1 is white
                n = HR_SCALE * HR_SCALE
                avg = (r + g + b) / (3.0 * n * 255.0)
                # invert so that black strokes become 1.0 (as many models expect white background)
                arr[dy, dx] = 1.0 - avg

        # shape -> (1,1,H,W)
        return arr.reshape(1, 1, DISPLAY_H, DISPLAY_W)

    # -------------------------------------------------------------
    def clear(self):
        """Clear the high-resolution canvas back to the erase color and update the view.

        Notifies the optional on_change callback so the UI can recompute predictions.
        """
        for y in range(HR_H):
            for x in range(HR_W):
                self.hr_pixels[y][x] = ERASE_COLOR

        self.update_display()

        # notify listener that drawing changed
        if hasattr(self, "on_change") and self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                pass


# ---------------------------------------------------------------
class ConfidenceTable:
    def __init__(self, parent_frame):

        parent_frame.rowconfigure(1, weight=1)  # row 1 will be tree
        parent_frame.columnconfigure(0, weight=1)

        # Add Predicted labels at top
        self.pred_label = tk.Label(parent_frame, text="Predicted Val: CNN= , SVM= , KNN= ")
        self.pred_label.grid(row=0, column=0, sticky="w", pady=(0,4))

        columns = ("class", "svm", "cnn", "knn")

        self.tree = ttk.Treeview(
            parent_frame,
            columns=columns,
            show="headings"
        )
        self.tree.grid(row=1, column=0, sticky="nsw")

        # Scrollbar
        vsb = ttk.Scrollbar(parent_frame, orient="vertical",
                            command=self.tree.yview)
        vsb.grid(row=1, column=1, sticky="ns")
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

        # Update the table
        for i, row_id in enumerate(self.rows):
            self.tree.item(row_id, values=(
                CLASSES[i],
                f"{svm[i]:.4f}",
                f"{cnn[i]:.4f}",
                f"{knn[i]:.4f}",
            ))

        # Update Predicted Val label
        pred_cnn = CLASSES[np.argmax(cnn)] if len(cnn) > 0 else ""
        pred_svm = CLASSES[np.argmax(svm)] if len(svm) > 0 else ""
        pred_knn = CLASSES[np.argmax(knn)] if len(knn) > 0 else ""

        self.pred_label.config(
            text=f"Predicted Val: CNN={pred_cnn}, SVM={pred_svm}, KNN={pred_knn}"
        )


# ---------------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------------
if __name__ == "__main__":
    cnn = CNNClassifier(device="cpu")
    cnn.load("models/cnn_png_20251123_203830.pt")
    svm = SVMClassifier()
    svm.load("models/svm_png_20251124_085830.pt")

    # Prepare KNN trained on the PNG dataset (uses same pre-processing as in main.py)
    try:
        png_ds = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.1)
        X_train, y_train, _, _ = png_ds.load()

        # Flatten and normalize to [-1,1] like CNN expects
        X_train_flat = X_train.reshape(len(X_train), -1).astype(np.float32)
        # PNGDataset might already give floats in [0,1]; if int, convert
        #if X_train_flat.max() > 1.0:
        #    X_train_flat = X_train_flat / 127.5 - 1.0  # scale 0..255 -> -1..1
        #else:
        #    X_train_flat = X_train_flat * 2.0 - 1.0   # scale 0..1 -> -1..1

        knn = KNNClassifier(k=1)
        knn.fit(X_train_flat, y_train)
        print(f"KNN trained on {len(X_train)} samples")
    except Exception as e:
        print("Failed to prepare KNN dataset:", e)
        knn = None

    root = tk.Tk()
    root.title("Model Playground")

    # Left side: canvas
    left = tk.Frame(root)
    left.pack(side="left", padx=10, pady=10)

    # Debounce scheduling container
    poll_after = {"id": None}
    POLL_DELAY = 300  # ms

    def schedule_predict():
        # cancel previous scheduled call and schedule a new one
        if poll_after["id"] is not None:
            try:
                root.after_cancel(poll_after["id"])
            except Exception:
                pass
            poll_after["id"] = None
        poll_after["id"] = root.after(POLL_DELAY, do_predict)

    def do_predict():
        poll_after["id"] = None
        try:
            arr = canvas.get_image()

            # Convert drawing to same normalization as dataset: PNGDataset uses ToTensor()+Normalize((0.5,),(0.5,))
            # which maps image in [0,1] -> [-1,1]. get_image() returns [0,1] inverted (black=1).
            arr01 = 1.0 - arr
            arr_norm = arr01 * 2.0 - 1.0

            # --- CNN probabilities mapped into UI ordering (length 62) ---
            cnn_probs62 = [0.0] * len(CLASSES)
            try:
                cnn_probs62 = cnn.predict_conf(arr_norm)[0]
            except Exception as e:
                print("CNN predict error:", e)

            # --- KNN: compute neighbor vote distribution and map into UI ordering ---
            knn_probs62 = [0.0] * len(CLASSES)
            try:
                if knn is not None and getattr(knn, "tree", None) is not None:
                    # KDTree expects flattened samples
                    sample = arr_norm.reshape(1, -1)
                    dist, idx = knn.tree.query(sample, k=knn.K)
                    neighbor_labels = knn.y_train[idx[0]]  # already 0..61

                    counts = np.bincount(neighbor_labels, minlength=len(CLASSES)).astype(float)
                    if counts.sum() > 0:
                        knn_probs62 = (counts / counts.sum()).tolist()
            except Exception as e:
                print("KNN predict error:", e)


            # --- SVM: compute class scores and softmax to probabilities ---
            svm_probs62 = [0.0] * len(CLASSES)
            try:
                svm_probs62 = svm.predict_conf(arr_norm)[0]
            except Exception as e:
                print("SVM predict error:", e)

            table.update_model_confidence({"cnn": cnn_probs62, "svm": svm_probs62, "knn": knn_probs62})
        except Exception as e:
            # keep UI live even if prediction fails
            print("Prediction error:", e)

    canvas = PixelCanvas(left, on_change=schedule_predict)

    # Controls under the canvas (clear button)
    controls = tk.Frame(left)
    controls.grid(row=1, column=0, pady=6, sticky="w")

    clear_btn = tk.Button(controls, text="Clear", command=canvas.clear)
    clear_btn.pack(side="left", padx=(0, 6))

    # Load a random sample from the PNGDataset and paint it to the canvas
    def _load_random_sample(event=None):
        try:
            # prefer training set if available
            ds_X = None
            try:
                # X_train from earlier load
                if 'X_train' in globals() and isinstance(X_train, (list, tuple, np.ndarray)):
                    ds_X = X_train
            except Exception:
                ds_X = None

            if ds_X is None:
                # try to (re)load dataset
                try:
                    ds = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.1)
                    X_train_local, y_train_local, X_test_local, y_test_local = ds.load()
                    ds_X = X_train_local
                except Exception as e:
                    print("Failed to load dataset for random sample:", e)
                    return

            # Ensure numpy array
            ds_X = np.asarray(ds_X)
            if ds_X.size == 0:
                print("Dataset appears empty; cannot load random sample.")
                return

            idx = int(np.random.randint(0, ds_X.shape[0]))
            sample = ds_X[idx]
            # sample shape may be (C,H,W) or (1,H,W) or (H,W)
            if sample.ndim == 3:
                # (C,H,W) -> take first channel
                img = sample[0]
            elif sample.ndim == 2:
                img = sample
            else:
                img = sample.reshape(sample.shape[-2], sample.shape[-1])

            # img values are in [-1,1] (PNGDataset). Convert back to [0,1]
            img01 = (img + 1.0) / 2.0

            # Paint into high-res pixel grid
            for dy in range(DISPLAY_H):
                for dx in range(DISPLAY_W):
                    v = float(img01[dy, dx])
                    c = int(clamp(round(v * 255.0), 0, 255))
                    col = (c, c, c)
                    # fill HR block
                    for j in range(HR_SCALE):
                        for i in range(HR_SCALE):
                            py = dy * HR_SCALE + j
                            px = dx * HR_SCALE + i
                            canvas.hr_pixels[py][px] = col

            canvas.update_display()
            # schedule prediction update
            schedule_predict()
        except Exception as e:
            print("Error loading random sample:", e)

    rand_btn = tk.Button(controls, text="Random", command=_load_random_sample)
    rand_btn.pack(side="left", padx=(0, 6))

    # bind 'c' key to clear the canvas
    def _on_clear_key(event=None):
        try:
            canvas.clear()
        except Exception:
            pass

    root.bind("c", _on_clear_key)

    # Right side: table
    right = tk.Frame(root)
    right.pack(side="left", padx=10, pady=10, fill="both", expand=True)

    table = ConfidenceTable(right)

    # initial prediction for blank canvas
    try:
        do_predict()
    except Exception:
        pass

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
