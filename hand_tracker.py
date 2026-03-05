"""
hand_tracker.py
---------------
Handles all MediaPipe-based hand detection and landmark extraction.
Detects up to 2 hands, classifies handedness (left/right),
and exposes finger-state + landmark helpers used by the rest of the app.
"""

import cv2
import mediapipe as mp
import math


# ──────────────────────────────────────────────
#  Landmark index constants (MediaPipe 21-point)
# ──────────────────────────────────────────────
WRIST          = 0
THUMB_TIP      = 4
INDEX_TIP      = 8
INDEX_MCP      = 5
MIDDLE_TIP     = 12
RING_TIP       = 16
PINKY_TIP      = 20

# Each finger's tip and its immediate PIP joint (used for "is finger up?" test)
FINGER_TIPS  = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS  = [3,          6,         10,          14,        18]


class HandTracker:
    """
    Wraps MediaPipe Hands and provides convenience methods for:
      - detecting landmarks in a BGR frame
      - querying whether individual fingers are extended
      - computing pixel coordinates of any landmark
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.75,
        tracking_confidence: float = 0.75,
    ):
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        # Stores the raw MediaPipe results for the most recent frame
        self.results = None

    # ──────────────────────────────────────────
    #  Core processing
    # ──────────────────────────────────────────
    def process(self, frame_bgr: "np.ndarray") -> "np.ndarray":
        """
        Run MediaPipe on *frame_bgr* (in-place RGB conversion, then back).
        Populates self.results and returns the same frame for convenience.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)
        rgb.flags.writeable = True
        return frame_bgr

    def draw_landmarks(self, frame: "np.ndarray") -> "np.ndarray":
        """Overlay skeleton + landmark dots on *frame* (modifies in-place)."""
        if not self.results or not self.results.multi_hand_landmarks:
            return frame

        for hand_lms in self.results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        return frame

    # ──────────────────────────────────────────
    #  Per-hand helpers
    # ──────────────────────────────────────────
    def get_hand_data(self, frame_shape: tuple) -> list:
        """
        Returns a list of dicts, one per detected hand:
          {
            'label':     'Left' | 'Right',
            'landmarks': list of (x_px, y_px, z) tuples,
            'lm_raw':    mediapipe NormalizedLandmarkList,
          }
        frame_shape = (height, width, channels)
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return []

        h, w = frame_shape[:2]
        hands_data = []

        handedness_list = (
            self.results.multi_handedness
            if self.results.multi_handedness
            else [None] * len(self.results.multi_hand_landmarks)
        )

        for hand_lms, handedness in zip(
            self.results.multi_hand_landmarks, handedness_list
        ):
            # MediaPipe reports the *mirrored* label when the camera is flipped,
            # so we swap Left↔Right to match what the user sees on screen.
            if handedness:
                raw_label = handedness.classification[0].label
                label = "Right" if raw_label == "Left" else "Left"
            else:
                label = "Unknown"

            landmarks = [
                (int(lm.x * w), int(lm.y * h), lm.z)
                for lm in hand_lms.landmark
            ]

            hands_data.append(
                {"label": label, "landmarks": landmarks, "lm_raw": hand_lms}
            )

        return hands_data

    # ──────────────────────────────────────────
    #  Finger-state utilities
    # ──────────────────────────────────────────
    @staticmethod
    def fingers_up(landmarks: list) -> list:
        """
        Returns a 5-element list [thumb, index, middle, ring, pinky]
        where 1 = finger extended, 0 = finger folded.

        landmarks: list of (x_px, y_px, z) from get_hand_data()
        """
        up = []

        # ── Thumb: compare tip x vs MCP x (works for right hand on mirrored feed)
        thumb_tip = landmarks[THUMB_TIP]
        thumb_mcp = landmarks[INDEX_MCP]
        up.append(1 if thumb_tip[0] < thumb_mcp[0] else 0)

        # ── Other four fingers: tip y < PIP y  →  extended
        for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            up.append(1 if landmarks[tip_idx][1] < landmarks[pip_idx][1] else 0)

        return up  # [thumb, index, middle, ring, pinky]

    @staticmethod
    def landmark_px(landmarks: list, idx: int) -> tuple:
        """Return (x, y) pixel coordinates for landmark *idx*."""
        return landmarks[idx][0], landmarks[idx][1]

    @staticmethod
    def distance(p1: tuple, p2: tuple) -> float:
        """Euclidean distance between two (x, y) points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def flatten_landmarks(landmarks: list) -> list:
        """
        Flatten 21 × (x, y, z) into a 63-element float list.
        Used as feature vector for the ML classifier.
        """
        flat = []
        for x, y, z in landmarks:
            flat.extend([x, y, z])
        return flat

    def release(self):
        """Clean up MediaPipe resources."""
        self.hands.close()
