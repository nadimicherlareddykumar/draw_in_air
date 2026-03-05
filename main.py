"""
main.py
-------
AI Hand Gesture Controlled Virtual Whiteboard
=============================================

Controls (gestures detected on RIGHT hand):
  ✦ Index finger up              → DRAW mode     (freehand / shape)
  ✦ Index + Middle up            → SELECT mode   (hover to pick colour)
  ✦ Index + Middle + Ring up     → SHAPE CYCLE   (cycle shape mode)
  ✦ All 4 fingers up (no thumb)  → SAVE drawing
  ✦ Open palm (all 5)            → CLEAR board

Keyboard shortcuts (always available):
  Z       – Undo          Y / Shift+Z – Redo
  S       – Save drawing  C           – Clear board
  1-4     – Shape mode    Q / Esc     – Quit
  T       – Train ML model (after collecting data)
  0-4     – Collect ML sample (while in collect mode, toggle with M)
"""

import cv2
import numpy as np
import time

from hand_tracker       import HandTracker
from gesture_classifier import GestureClassifier, GESTURE_LABELS
from utils import (
    COLORS, COLOR_ORDER, TOOLBAR_HEIGHT,
    SHAPE_FREEHAND, SHAPE_LINE, SHAPE_RECTANGLE, SHAPE_CIRCLE, SHAPE_ORDER,
    build_toolbar, get_color_at, draw_toolbar_highlight,
    create_canvas, merge_canvas_with_frame,
    draw_shape, save_drawing,
    HistoryManager,
    draw_hud, draw_gesture_banner, draw_finger_cursor,
)


# ──────────────────────────────────────────────────────────────
#  App constants
# ──────────────────────────────────────────────────────────────
CAMERA_ID        = 0          # Change if your webcam index differs
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720
BRUSH_THICKNESS  = 5
ERASER_THICKNESS = 40

# How long (seconds) a gesture banner stays visible
BANNER_DURATION  = 1.5

# Confidence threshold for ML prediction
ML_CONFIDENCE_THRESHOLD = 0.70

# Smoothing: blend current finger pos with previous (0 = no smooth, 1 = freeze)
SMOOTH_ALPHA = 0.45


# ──────────────────────────────────────────────────────────────
#  Gesture → mode mapping  (rule-based, no ML needed)
# ──────────────────────────────────────────────────────────────
def rule_based_gesture(fingers: list) -> str:
    """
    fingers: [thumb, index, middle, ring, pinky]  (1=up, 0=down)
    Returns a gesture label string.
    """
    t, i, m, r, p = fingers

    if   [t, i, m, r, p] == [0, 1, 0, 0, 0]:   return "draw"
    elif [t, i, m, r, p] == [0, 1, 1, 0, 0]:   return "erase"
    elif [t, i, m, r, p] == [0, 1, 1, 1, 0]:   return "change_color"
    elif [t, i, m, r, p] == [0, 1, 1, 1, 1]:   return "save"
    elif [t, i, m, r, p] == [1, 1, 1, 1, 1]:   return "clear"
    elif [t, i, m, r, p] == [0, 0, 0, 0, 0]:   return "idle"
    else:                                        return "unknown"


# ──────────────────────────────────────────────────────────────
#  Application state class
# ──────────────────────────────────────────────────────────────
class WhiteboardApp:

    def __init__(self):
        # ── Camera ──────────────────────────────────────────
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot open webcam. Check CAMERA_ID in main.py")

        self.h = test_frame.shape[0]
        self.w = test_frame.shape[1]

        # ── Core modules ────────────────────────────────────
        self.tracker    = HandTracker(max_hands=2)
        self.classifier = GestureClassifier()
        self.history    = HistoryManager(max_steps=20)

        # ── Canvas ──────────────────────────────────────────
        self.canvas     = create_canvas(self.h, self.w)
        self.toolbar    = build_toolbar(self.w)

        # ── Drawing state ───────────────────────────────────
        self.active_color_name = "Blue"
        self.active_color      = COLORS["Blue"]
        self.shape_mode        = SHAPE_FREEHAND
        self.is_drawing        = False          # True while finger is down in draw mode
        self.shape_start       = None           # anchor for line/rect/circle
        self.prev_pt           = None           # previous finger position (freehand)
        self.smooth_pt         = None           # smoothed finger position

        # Temp canvas for live shape preview (not committed until finger lifts)
        self.preview_canvas    = None

        # ── Mode tracking ───────────────────────────────────
        self.current_mode      = "idle"
        self.gesture_label     = "unknown"
        self.ml_gesture        = "unknown"

        # ── Banner / notification ────────────────────────────
        self.banner_text       = ""
        self.banner_until      = 0.0

        # ── FPS ─────────────────────────────────────────────
        self.fps               = 0.0
        self._t_last           = time.time()

        # ── ML data-collection mode ──────────────────────────
        self.collecting        = False
        self.collect_label     = GESTURE_LABELS[0]
        self.collect_idx       = 0

        # ── Shape cycling ────────────────────────────────────
        self._shape_gesture_cooldown = 0.0

        print("\n" + "="*55)
        print("  Virtual Whiteboard  –  press H for help")
        print("="*55 + "\n")

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────
    def _smooth_point(self, pt: tuple) -> tuple:
        """Exponential moving average to smooth jitter."""
        if self.smooth_pt is None:
            self.smooth_pt = pt
            return pt
        sx = int(SMOOTH_ALPHA * self.smooth_pt[0] + (1 - SMOOTH_ALPHA) * pt[0])
        sy = int(SMOOTH_ALPHA * self.smooth_pt[1] + (1 - SMOOTH_ALPHA) * pt[1])
        self.smooth_pt = (sx, sy)
        return self.smooth_pt

    def _show_banner(self, text: str):
        self.banner_text  = text
        self.banner_until = time.time() + BANNER_DURATION

    def _commit_shape(self):
        """Merge preview canvas into persistent canvas."""
        if self.preview_canvas is not None:
            self.canvas = self.preview_canvas.copy()
            self.preview_canvas = None

    def _thickness(self):
        return ERASER_THICKNESS if self.active_color_name == "Eraser" else BRUSH_THICKNESS

    # ──────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────
    def run(self):
        print("Webcam open – press Q or Esc to quit.\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame capture failed.")
                break

            # ── Mirror frame so it feels like a mirror ──────
            frame = cv2.flip(frame, 1)

            # ── FPS calculation ──────────────────────────────
            now       = time.time()
            self.fps  = 1.0 / max(now - self._t_last, 1e-6)
            self._t_last = now

            # ── Hand tracking ────────────────────────────────
            self.tracker.process(frame)
            self.tracker.draw_landmarks(frame)
            hands = self.tracker.get_hand_data(frame.shape)

            # ── Process each hand ────────────────────────────
            right_hand = next((h for h in hands if h["label"] == "Right"), None)
            left_hand  = next((h for h in hands if h["label"] == "Left"),  None)

            # Left hand → colour picker
            if left_hand:
                self._handle_left_hand(left_hand)

            # Right hand → drawing / gestures
            if right_hand:
                self._handle_right_hand(right_hand, frame)
            else:
                # Finger lifted → commit any in-progress shape
                if self.is_drawing and self.shape_mode != SHAPE_FREEHAND:
                    self._commit_shape()
                self.is_drawing  = False
                self.shape_start = None
                self.prev_pt     = None
                self.gesture_label = "no hand"

            # ── Merge canvas + frame ─────────────────────────
            draw_canvas = (
                self.preview_canvas
                if self.preview_canvas is not None
                else self.canvas
            )
            frame = merge_canvas_with_frame(frame, draw_canvas)

            # ── Toolbar overlay ──────────────────────────────
            bar = draw_toolbar_highlight(self.toolbar.copy(),
                                         self.active_color_name,
                                         self.shape_mode)
            frame[:TOOLBAR_HEIGHT, :] = bar

            # ── HUD ──────────────────────────────────────────
            frame = draw_hud(
                frame,
                mode          = self.current_mode,
                color_name    = self.active_color_name,
                color_bgr     = self.active_color,
                shape_mode    = self.shape_mode,
                gesture_label = self.gesture_label,
                fps           = self.fps,
                toolbar_h     = TOOLBAR_HEIGHT,
                history       = self.history,
                collecting    = self.collecting,
                collect_label = self.collect_label,
            )

            # ── Banner ───────────────────────────────────────
            if time.time() < self.banner_until:
                draw_gesture_banner(frame, self.banner_text)

            # ── Shape mode label (top-right) ─────────────────
            sm_text = f"Shape: {self.shape_mode.upper()}"
            cv2.putText(frame, sm_text, (self.w - 240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Virtual Whiteboard  [Q/Esc = quit]", frame)

            # ── Keyboard handler ─────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if self._handle_key(key):
                break

        self._cleanup()

    # ──────────────────────────────────────────────
    #  Right-hand logic
    # ──────────────────────────────────────────────
    def _handle_right_hand(self, hand: dict, frame: "np.ndarray"):
        lms     = hand["landmarks"]
        fingers = HandTracker.fingers_up(lms)
        idx_tip = HandTracker.landmark_px(lms, 8)   # index fingertip
        idx_tip = self._smooth_point(idx_tip)

        # ── Rule-based gesture ────────────────────
        rb_gesture = rule_based_gesture(fingers)

        # ── ML gesture (if model loaded) ──────────
        if self.classifier.is_trained:
            flat = HandTracker.flatten_landmarks(lms)
            ml_lbl, ml_conf = self.classifier.predict(flat)
            self.ml_gesture = f"{ml_lbl} ({ml_conf*100:.0f}%)"
            # Override rule-based only when confident
            if ml_conf >= ML_CONFIDENCE_THRESHOLD:
                rb_gesture = ml_lbl

        self.gesture_label = rb_gesture

        # ── Collect training data ─────────────────
        if self.collecting:
            flat = HandTracker.flatten_landmarks(lms)
            self.classifier.collect_sample(flat, self.collect_label)

        # ── Act on gesture ────────────────────────
        if rb_gesture == "draw":
            self.current_mode = "DRAW"
            draw_finger_cursor(frame, idx_tip, self.active_color, drawing=True)

            if idx_tip[1] < TOOLBAR_HEIGHT:
                # Finger in toolbar → pick colour
                result = get_color_at(idx_tip[0], self.w)
                if result:
                    cname, cbgr = result
                    self.active_color_name = cname
                    self.active_color      = cbgr
                    self._show_banner(f"Colour: {cname}")
            else:
                # Drawing on canvas
                if self.shape_mode == SHAPE_FREEHAND:
                    if self.prev_pt:
                        cv2.line(self.canvas, self.prev_pt, idx_tip,
                                 self.active_color, self._thickness())
                    self.prev_pt = idx_tip
                    self.is_drawing = True
                else:
                    # Shape drawing – live preview
                    if not self.is_drawing:
                        # Start of new shape
                        self.history.push(self.canvas)
                        self.shape_start    = idx_tip
                        self.preview_canvas = self.canvas.copy()
                        self.is_drawing     = True
                    else:
                        # Update live preview
                        self.preview_canvas = draw_shape(
                            self.canvas, self.shape_mode,
                            self.shape_start, idx_tip,
                            self.active_color, self._thickness()
                        )

        elif rb_gesture == "erase":
            self.current_mode = "SELECT"
            draw_finger_cursor(frame, idx_tip, (200, 200, 200), drawing=False)

            # Commit any in-progress shape
            if self.is_drawing and self.shape_mode != SHAPE_FREEHAND:
                self._commit_shape()
            self.is_drawing  = False
            self.shape_start = None
            self.prev_pt     = None

            # Hover over toolbar → pick colour
            if idx_tip[1] < TOOLBAR_HEIGHT:
                result = get_color_at(idx_tip[0], self.w)
                if result:
                    cname, cbgr = result
                    self.active_color_name = cname
                    self.active_color      = cbgr

        elif rb_gesture == "change_color":
            # Cycle shape mode (with cooldown so it doesn't flicker)
            if time.time() > self._shape_gesture_cooldown:
                idx = SHAPE_ORDER.index(self.shape_mode)
                self.shape_mode = SHAPE_ORDER[(idx + 1) % len(SHAPE_ORDER)]
                self._shape_gesture_cooldown = time.time() + 1.0
                self._show_banner(f"Shape: {self.shape_mode}")
            self.current_mode = "SHAPE"
            self._reset_drawing()

        elif rb_gesture == "save":
            self.current_mode = "SAVE"
            self._save_action()
            self._reset_drawing()

        elif rb_gesture == "clear":
            self.current_mode = "CLEAR"
            self._clear_action()
            self._reset_drawing()

        elif rb_gesture == "idle":
            self.current_mode = "idle"
            self._reset_drawing()
        else:
            self.current_mode = rb_gesture
            draw_finger_cursor(frame, idx_tip, (150, 150, 150), drawing=False)
            self._reset_drawing()

    # ──────────────────────────────────────────────
    #  Left-hand logic (colour picking)
    # ──────────────────────────────────────────────
    def _handle_left_hand(self, hand: dict):
        lms     = hand["landmarks"]
        idx_tip = HandTracker.landmark_px(lms, 8)

        # If left index finger is in toolbar zone, pick colour
        if idx_tip[1] < TOOLBAR_HEIGHT:
            result = get_color_at(idx_tip[0], self.w)
            if result:
                cname, cbgr = result
                if cname != self.active_color_name:
                    self.active_color_name = cname
                    self.active_color      = cbgr
                    self._show_banner(f"Colour: {cname}")

    # ──────────────────────────────────────────────
    #  Actions
    # ──────────────────────────────────────────────
    def _save_action(self):
        path = save_drawing(self.canvas, output_dir="saved_drawings")
        self._show_banner(f"Saved!")

    def _clear_action(self):
        self.history.push(self.canvas)
        self.canvas = create_canvas(self.h, self.w)
        self.preview_canvas = None
        self._show_banner("Board cleared")

    def _undo_action(self):
        if self.history.can_undo:
            self.canvas = self.history.undo(self.canvas)
            self._show_banner("Undo")

    def _redo_action(self):
        if self.history.can_redo:
            self.canvas = self.history.redo(self.canvas)
            self._show_banner("Redo")

    def _reset_drawing(self):
        """Reset per-stroke state when not in draw mode."""
        if self.is_drawing and self.shape_mode != SHAPE_FREEHAND:
            self._commit_shape()
        self.is_drawing     = False
        self.shape_start    = None
        self.prev_pt        = None

    # ──────────────────────────────────────────────
    #  Keyboard handler
    # ──────────────────────────────────────────────
    def _handle_key(self, key: int) -> bool:
        """Returns True to quit."""
        if key in (ord('q'), ord('Q'), 27):   # Q or Esc
            return True

        elif key == ord('z'):                 # Undo
            self._undo_action()

        elif key in (ord('y'), ord('Z')):     # Redo
            self._redo_action()

        elif key == ord('s'):                 # Save
            self._save_action()

        elif key == ord('c'):                 # Clear
            self._clear_action()

        elif key == ord('1'):                 # Freehand
            self.shape_mode = SHAPE_FREEHAND
            self._show_banner("Shape: Freehand")

        elif key == ord('2'):                 # Line
            self.shape_mode = SHAPE_LINE
            self._show_banner("Shape: Line")

        elif key == ord('3'):                 # Rectangle
            self.shape_mode = SHAPE_RECTANGLE
            self._show_banner("Shape: Rectangle")

        elif key == ord('4'):                 # Circle
            self.shape_mode = SHAPE_CIRCLE
            self._show_banner("Shape: Circle")

        elif key == ord('t'):                 # Train ML model
            print("\n[Train] Starting training…")
            acc = self.classifier.train()
            if acc > 0:
                self._show_banner(f"Model trained! Acc: {acc*100:.0f}%")

        elif key == ord('m'):                 # Toggle ML data-collection
            self.collecting = not self.collecting
            self._show_banner(
                f"Collecting: {'ON' if self.collecting else 'OFF'} [{self.collect_label}]"
            )

        elif ord('0') <= key <= ord('4') and self.collecting:
            # 0-4 → choose collection label
            self.collect_idx   = key - ord('0')
            self.collect_label = GESTURE_LABELS[self.collect_idx]
            self._show_banner(f"Label → {self.collect_label}")

        elif key == ord('h'):                 # Help
            self._print_help()

        return False

    # ──────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────
    def _cleanup(self):
        self.cap.release()
        self.tracker.release()
        cv2.destroyAllWindows()
        print("\nApp closed.")

    @staticmethod
    def _print_help():
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║          Virtual Whiteboard  –  Controls             ║
  ╠══════════════════════════════════════════════════════╣
  ║  GESTURES (right hand)                               ║
  ║    Index up          → Draw / freehand               ║
  ║    Index+Middle up   → Select mode (pick colour)     ║
  ║    3 fingers up      → Cycle shape mode              ║
  ║    4 fingers up      → Save drawing                  ║
  ║    Open palm (5)     → Clear board                   ║
  ╠══════════════════════════════════════════════════════╣
  ║  KEYBOARD                                            ║
  ║    Z       Undo             Y / Shift-Z  Redo        ║
  ║    S       Save             C            Clear       ║
  ║    1       Freehand         2            Line        ║
  ║    3       Rectangle        4            Circle      ║
  ║    T       Train ML model                            ║
  ║    M       Toggle ML data-collect mode               ║
  ║    0-4     Choose collect label (in collect mode)    ║
  ║    H       This help        Q / Esc      Quit        ║
  ╚══════════════════════════════════════════════════════╝
        """)


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = WhiteboardApp()
    app.run()
