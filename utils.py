"""
utils.py
--------
Stateless helper functions for:
  - toolbar / UI rendering
  - shape drawing
  - canvas merge
  - undo / redo stack
  - on-screen HUD overlays
"""

import cv2
import numpy as np
from datetime import datetime
import os


# ──────────────────────────────────────────────
#  Colour palette  (BGR)
# ──────────────────────────────────────────────
COLORS = {
    "Blue":   (255, 100,   0),
    "Green":  (  0, 200,   0),
    "Red":    (  0,   0, 220),
    "Yellow": (  0, 220, 220),
    "Purple": (200,   0, 200),
    "Orange": (  0, 140, 255),
    "White":  (255, 255, 255),
    "Eraser": (  0,   0,   0),   # draws in black → "erases" on black canvas
}

COLOR_ORDER = ["Blue", "Green", "Red", "Yellow", "Purple", "Orange", "White", "Eraser"]

# Toolbar geometry
TOOLBAR_HEIGHT = 80          # pixels
SWATCH_MARGIN  = 10
SWATCH_RADIUS  = 28

# Shape modes
SHAPE_FREEHAND  = "freehand"
SHAPE_LINE      = "line"
SHAPE_RECTANGLE = "rectangle"
SHAPE_CIRCLE    = "circle"

SHAPE_ORDER = [SHAPE_FREEHAND, SHAPE_LINE, SHAPE_RECTANGLE, SHAPE_CIRCLE]
SHAPE_ICONS = {
    SHAPE_FREEHAND:  "~",
    SHAPE_LINE:      "/",
    SHAPE_RECTANGLE: "[]",
    SHAPE_CIRCLE:    "O",
}

# ──────────────────────────────────────────────
#  Toolbar helpers
# ──────────────────────────────────────────────

def build_toolbar(width: int) -> np.ndarray:
    """
    Render the colour-palette + shape-selector toolbar as an image.
    Returns a (TOOLBAR_HEIGHT, width, 3) uint8 array.
    """
    bar = np.zeros((TOOLBAR_HEIGHT, width, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)   # dark background

    # ── colour swatches ─────────────────────
    for i, name in enumerate(COLOR_ORDER):
        cx = SWATCH_MARGIN + i * (SWATCH_RADIUS * 2 + SWATCH_MARGIN) + SWATCH_RADIUS
        cy = TOOLBAR_HEIGHT // 2
        color = COLORS[name]
        cv2.circle(bar, (cx, cy), SWATCH_RADIUS, color, -1)
        cv2.circle(bar, (cx, cy), SWATCH_RADIUS, (200, 200, 200), 1)

    return bar


def get_color_at(x: int, width: int) -> tuple | None:
    """
    Given finger x-position on the toolbar, return the (BGR) color or None.
    """
    for i, name in enumerate(COLOR_ORDER):
        cx = SWATCH_MARGIN + i * (SWATCH_RADIUS * 2 + SWATCH_MARGIN) + SWATCH_RADIUS
        if abs(x - cx) <= SWATCH_RADIUS:
            return name, COLORS[name]
    return None


def draw_toolbar_highlight(bar: np.ndarray, color_name: str, shape_mode: str) -> np.ndarray:
    """
    Draw a white ring around the active colour swatch.
    Returns a copy of *bar* with the highlight drawn.
    """
    bar_copy = bar.copy()

    # highlight active colour
    if color_name in COLOR_ORDER:
        i  = COLOR_ORDER.index(color_name)
        cx = SWATCH_MARGIN + i * (SWATCH_RADIUS * 2 + SWATCH_MARGIN) + SWATCH_RADIUS
        cy = TOOLBAR_HEIGHT // 2
        cv2.circle(bar_copy, (cx, cy), SWATCH_RADIUS + 4, (255, 255, 255), 3)

    return bar_copy


# ──────────────────────────────────────────────
#  Canvas operations
# ──────────────────────────────────────────────

def create_canvas(height: int, width: int) -> np.ndarray:
    """Return a black (all-zero) uint8 canvas of the given size."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def merge_canvas_with_frame(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """
    Overlay *canvas* on *frame* using bitwise operations so that
    only pixels actually drawn on the canvas are visible.

    Steps:
      1. Convert canvas to grayscale, threshold to get a mask.
      2. Zero out those pixels in the frame.
      3. Add the canvas pixels in their place.
    """
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)

    # Invert mask: pixels to keep from frame
    mask_inv = cv2.bitwise_not(mask)

    frame_bg  = cv2.bitwise_and(frame,  frame,  mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    return cv2.add(frame_bg, canvas_fg)


# ──────────────────────────────────────────────
#  Undo / Redo
# ──────────────────────────────────────────────

class HistoryManager:
    """
    Simple undo/redo stack for the drawing canvas.
    Stores full canvas snapshots (memory-light for typical webcam resolutions).
    """

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

    def push(self, canvas: np.ndarray):
        """Save a copy of *canvas* as the latest history state."""
        self._undo_stack.append(canvas.copy())
        if len(self._undo_stack) > self.max_steps:
            self._undo_stack.pop(0)
        self._redo_stack.clear()   # new action invalidates redo history

    def undo(self, current_canvas: np.ndarray) -> np.ndarray:
        """Return the previous canvas state, pushing current onto redo stack."""
        if not self._undo_stack:
            return current_canvas
        self._redo_stack.append(current_canvas.copy())
        return self._undo_stack.pop()

    def redo(self, current_canvas: np.ndarray) -> np.ndarray:
        """Re-apply the most recently undone action."""
        if not self._redo_stack:
            return current_canvas
        self._undo_stack.append(current_canvas.copy())
        return self._redo_stack.pop()

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0


# ──────────────────────────────────────────────
#  Shape drawing
# ──────────────────────────────────────────────

def draw_shape(
    canvas: np.ndarray,
    shape_mode: str,
    start: tuple,
    end: tuple,
    color: tuple,
    thickness: int = 4,
) -> np.ndarray:
    """
    Draw *shape_mode* from *start* to *end* on *canvas*.
    Returns a fresh copy so the preview doesn't pollute the persistent canvas.
    """
    c = canvas.copy()

    if shape_mode == SHAPE_LINE:
        cv2.line(c, start, end, color, thickness)

    elif shape_mode == SHAPE_RECTANGLE:
        cv2.rectangle(c, start, end, color, thickness)

    elif shape_mode == SHAPE_CIRCLE:
        radius = int(np.hypot(end[0] - start[0], end[1] - start[1]))
        cv2.circle(c, start, radius, color, thickness)

    # FREEHAND is handled stroke-by-stroke in main.py, not here

    return c


# ──────────────────────────────────────────────
#  Save drawing
# ──────────────────────────────────────────────

def save_drawing(canvas: np.ndarray, output_dir: str = ".") -> str:
    """
    Save *canvas* to a PNG file with a timestamped filename.
    Returns the full path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"drawing_{ts}.png")
    cv2.imwrite(filepath, canvas)
    print(f"[Save] Drawing saved → {filepath}")
    return filepath


# ──────────────────────────────────────────────
#  HUD / on-screen overlays
# ──────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    mode: str,
    color_name: str,
    color_bgr: tuple,
    shape_mode: str,
    gesture_label: str,
    fps: float,
    toolbar_h: int,
    history: "HistoryManager",
    collecting: bool = False,
    collect_label: str = "",
) -> np.ndarray:
    """
    Draw a semi-transparent HUD panel in the bottom-left corner.
    """
    panel_x, panel_y = 10, frame.shape[0] - 170
    panel_w, panel_h = 320, 155

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def put(text, row, color=(220, 220, 220)):
        cv2.putText(frame, text, (panel_x + 10, panel_y + row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    put(f"Mode    : {mode}",            22)
    put(f"Colour  : {color_name}",      44,  color_bgr)
    put(f"Shape   : {shape_mode}",      66)
    put(f"Gesture : {gesture_label}",   88)
    put(f"FPS     : {fps:.1f}",        110)
    put(f"Undo: {'Y' if history.can_undo else 'N'}   "
        f"Redo: {'Y' if history.can_redo else 'N'}",   132)

    if collecting:
        cv2.putText(frame, f"COLLECTING: {collect_label}", (panel_x, panel_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)

    return frame


def draw_gesture_banner(frame: np.ndarray, text: str, color=(0, 220, 255)):
    """Flash a large banner at the top-centre of the frame."""
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (w // 2 - len(text) * 8, 130),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)


def draw_finger_cursor(frame: np.ndarray, pos: tuple, color: tuple, drawing: bool):
    """Draw a small circle at the index-finger tip."""
    if pos is None:
        return
    radius = 12 if drawing else 8
    thickness = -1 if drawing else 2
    cv2.circle(frame, pos, radius, color, thickness)
    cv2.circle(frame, pos, radius + 2, (255, 255, 255), 1)
