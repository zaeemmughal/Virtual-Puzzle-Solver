"""
Hand-Gesture Controlled Selfie Merge Puzzle
=============================================

Take a selfie with your webcam, watch it get chopped into an N x N grid of
puzzle pieces, and reassemble it into the full picture using nothing but
your hand -- no mouse, no keyboard.

State machine
-------------
    DIFFICULTY -> CAPTURE -> SETUP -> GAMEPLAY -> WIN -> (back to CAPTURE)

    DIFFICULTY  : hold up 1 / 2 / 3 fingers to choose a 3x3 / 4x4 / 5x5 grid
                  (press SPACE any time to skip and use the default grid).
    CAPTURE     : center your face in the guide box and hold still; a 3-2-1
                  countdown fires automatically (or press SPACE to force it).
    SETUP       : (instant) the captured selfie is sliced into pieces and
                  the pieces are shuffled into random board positions.
    GAMEPLAY    : point your index finger at a piece and pinch (thumb +
                  index fingertip touching) to pick it up, move your hand
                  to another cell, then pinch again to drop it there --
                  swapping it with whatever piece was already sitting in
                  that cell. Keep swapping until every piece has merged
                  back into its correct spot.
    WIN         : the reconstructed selfie is shown along with your time and
                  move count. Show an open palm to play again, a fist to quit.

Run
---
    pip install -r requirements.txt
    python selfie_puzzle.py

Press 'q' at any time to quit.

Why this design
----------------
Everything is intentionally kept in a small number of pure-ish functions
(capture_selfie, build_puzzle, shuffle_board, detect_pinch,
map_hand_to_grid, check_win, render_frame) so each piece of computer-vision
or game logic can be pointed to individually -- handy if you ever need to
explain a specific piece of this project (e.g. in an interview).
"""

import os
import cv2
import json
import time
import random
import numpy as np
import mediapipe as mp

# --------------------------------------------------------------------------
# Tunable constants -- change these first if something feels miscalibrated.
# --------------------------------------------------------------------------

DEFAULT_GRID_SIZE = 3          # N -> produces an N x N puzzle (3 or 4 typical)
MIN_GRID_SIZE = 2
MAX_GRID_SIZE = 5

CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480

PUZZLE_PANEL_SIZE = 480        # rendered size (px) of the square puzzle panel
CAM_PANEL_WIDTH = 320          # rendered size (px) of the live camera panel
WINDOW_NAME = "Hand-Gesture Selfie Merge Puzzle"

FACE_STEADY_SECONDS = 1.0      # how long a centered face must be held
COUNTDOWN_SECONDS = 3          # 3-2-1 countdown length
FACE_CENTER_TOLERANCE = 0.12   # how far (fraction of frame) face center may
                                # drift from the frame center and still count
                                # as "steady"

# Pinch is measured as normalized landmark distance (0-1 range, independent
# of resolution) divided by an estimate of the hand's own size (wrist to
# middle-finger MCP), so the threshold stays roughly correct whether the
# hand is close to or far from the camera. Lower = must pinch tighter.
PINCH_THRESHOLD_RATIO = 0.35
MOVE_COOLDOWN_SECONDS = 0.45   # debounce so a jittery pinch registers as one
                                # clean pick-up or drop, not several

WIN_GESTURE_HOLD_SECONDS = 0.6  # how long to hold palm/fist to confirm choice

LEADERBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "leaderboard.json")

TILE_GAP = 3            # px gap drawn between tiles (purely cosmetic)
SWAP_ANIM_FRAMES = 6    # frames to animate two pieces swapping places

HOVER_COLOR = (0, 220, 255)         # cell outline while just hovering
DROP_TARGET_COLOR = (0, 255, 120)   # cell outline while holding a piece
HELD_BORDER_COLOR = (0, 220, 255)   # outline drawn around the floating piece
HELD_SCALE = 0.82                   # floating piece is drawn a bit smaller
                                     # than a full cell so it visibly "lifts"
                                     # off the board while you're carrying it

FINGER_SMOOTHING_ALPHA = 0.5   # 0-1 exponential smoothing on the fingertip
                                # position; lower = smoother but laggier,
                                # higher = snappier but jitterier
PINCH_LOCK_RATIO = 0.55        # once the pinch ratio drops below this (the
                                # finger starting to close), the target cell
                                # is frozen -- otherwise the inward curl of a
                                # closing pinch shifts the index fingertip
                                # toward the thumb just enough to flip the
                                # hover onto a neighboring cell right as the
                                # pinch actually triggers

# --------------------------------------------------------------------------
# MediaPipe setup
# --------------------------------------------------------------------------

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices we care about (see MediaPipe Hands documentation)
LM_THUMB_TIP = 4
LM_INDEX_TIP = 8
LM_WRIST = 0
LM_MIDDLE_MCP = 9

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------

STATE_DIFFICULTY = "difficulty"
STATE_CAPTURE = "capture"
STATE_SETUP = "setup"
STATE_GAMEPLAY = "gameplay"
STATE_WIN = "win"


# ==========================================================================
# Pure-logic helper functions (no OpenCV window / camera dependency)
# ==========================================================================

def count_extended_fingers(hand_landmarks, handedness_label="Right"):
    """Rough heuristic finger counter used for the difficulty selector.

    Compares each fingertip's y-coordinate to its PIP joint's y-coordinate
    (a raised finger has its tip above -- i.e. smaller y than -- its pip).
    Thumb uses an x-comparison instead since it moves sideways, not up/down;
    the sign flips depending on which hand is shown to a mirrored selfie
    camera. This is a heuristic, not a robust classifier, but it's more
    than enough to distinguish "1 finger" from "2 fingers" from "3 fingers".
    """
    lm = hand_landmarks.landmark
    count = 0

    # Four fingers: index, middle, ring, pinky
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        if lm[tip_idx].y < lm[pip_idx].y:
            count += 1

    # Thumb: compare tip to the joint below it (IP, landmark 3) on the x axis.
    # The frame is mirrored (selfie view) before landmarks are computed, so
    # a "Right" hand shown to the mirrored camera reads as MediaPipe's "Left"
    # and vice versa -- we just try both directions and don't over-engineer.
    thumb_tip_x = lm[4].x
    thumb_ip_x = lm[3].x
    if handedness_label == "Left":
        if thumb_tip_x > thumb_ip_x:
            count += 1
    else:
        if thumb_tip_x < thumb_ip_x:
            count += 1

    return count


def is_open_palm(hand_landmarks, handedness_label="Right"):
    """True when 4+ fingers (including thumb) read as extended."""
    return count_extended_fingers(hand_landmarks, handedness_label) >= 4


def is_fist(hand_landmarks, handedness_label="Right"):
    """True when 0-1 fingers read as extended (a closed hand)."""
    return count_extended_fingers(hand_landmarks, handedness_label) <= 1


def detect_pinch(hand_landmarks):
    """Return True if thumb tip and index fingertip are touching.

    Distance is normalized by an estimate of the hand's on-screen size
    (wrist -> middle-finger-MCP distance) so the same threshold works
    whether the hand is near or far from the camera.
    """
    lm = hand_landmarks.landmark
    thumb = lm[LM_THUMB_TIP]
    index = lm[LM_INDEX_TIP]
    wrist = lm[LM_WRIST]
    middle_mcp = lm[LM_MIDDLE_MCP]

    pinch_dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
    hand_scale = ((wrist.x - middle_mcp.x) ** 2 + (wrist.y - middle_mcp.y) ** 2) ** 0.5

    if hand_scale < 1e-6:
        return False, 999.0

    ratio = pinch_dist / hand_scale
    return ratio < PINCH_THRESHOLD_RATIO, ratio


def map_hand_to_grid(norm_x, norm_y, grid_size):
    """Map a normalized (0-1, 0-1) fingertip position to an (row, col) cell.

    norm_x / norm_y are the *entire camera frame* treated as the puzzle's
    interaction surface (an "air trackpad"), so moving your finger to the
    top-left of frame hovers the top-left tile, etc.
    Returns None if the point falls outside the 0-1 range (finger off-frame).
    """
    if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
        return None
    col = min(grid_size - 1, max(0, int(norm_x * grid_size)))
    row = min(grid_size - 1, max(0, int(norm_y * grid_size)))
    return row, col


def build_puzzle(image, n):
    """Slice `image` (assumed already square, BGR) into an n x n list of
    puzzle-piece images, numbered row-major from 0 .. n*n-1.

    Unlike a sliding puzzle there's no reserved "blank" -- every cell always
    holds a real piece, and pieces are rearranged by picking one up and
    dropping it somewhere else (which swaps it with whatever was there).
    """
    h, w = image.shape[:2]
    tile_h, tile_w = h // n, w // n
    tiles = []
    for r in range(n):
        for c in range(n):
            tile = image[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
            tiles.append(tile.copy())
    return tiles


def solved_board(n):
    """Return the solved board as a flat list: board[pos] = tile_index."""
    return list(range(n * n))


def shuffle_board(n, rng=None):
    """Return a random permutation of the solved board to start a round.

    A sliding puzzle has to shuffle via legal slide moves because roughly
    half of raw permutations of an (n^2-1)-puzzle are mathematically
    unsolvable. That constraint doesn't apply here: since you can pick up
    *any* piece and drop it into *any* cell (swapping the two), every
    permutation is reachable from every other one. A plain random shuffle
    is all that's needed.
    """
    rng = rng or random
    board = solved_board(n)
    rng.shuffle(board)
    return board


def check_win(board, n):
    """True if `board` matches the solved order."""
    return board == solved_board(n)


def crop_face_square(frame, face_bbox, padding_ratio=0.6):
    """Given a face bounding box (x, y, w, h) in pixel coords, crop a padded
    SQUARE region around it (clamped to the frame) suitable for slicing into
    puzzle pieces.
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = face_bbox
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x0 = x - pad_w
    y0 = y - pad_h
    x1 = x + w + pad_w
    y1 = y + h + pad_h

    # Make it square around the same center before clamping
    box_w, box_h = x1 - x0, y1 - y0
    side = max(box_w, box_h)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, x1 = cx - side // 2, cx + side // 2
    y0, y1 = cy - side // 2, cy + side // 2

    # Clamp to frame bounds, shifting the box inward if needed rather than
    # shrinking it unevenly
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > fw:
        x0 -= (x1 - fw)
        x1 = fw
    if y1 > fh:
        y0 -= (y1 - fh)
        y1 = fh
    x0, y0 = max(0, x0), max(0, y0)

    return frame[y0:y1, x0:x1].copy()


# ==========================================================================
# Rendering helpers
# ==========================================================================

def render_puzzle_panel(tiles, board, n, size=PUZZLE_PANEL_SIZE,
                         hover_cell=None, anim=None, held=None):
    """Draw the current board state into a `size` x `size` BGR canvas.

    `anim`, if given, is a dict describing an in-flight swap animation --
    two pieces trading places simultaneously:
        {"pos_a": flat index, "tile_a": tile index,
         "pos_b": flat index, "tile_b": tile index,
         "progress": 0..1 float}
    Both pieces are drawn interpolated between their start and end cells
    instead of snapping instantly, and are skipped in the normal grid draw
    for those two cells.

    `held`, if given, describes a piece that's currently picked up and not
    yet dropped:
        {"from_pos": flat index the piece was lifted from,
         "tile": tile index,
         "fx", "fy": pixel coords within this canvas to float it at}
    Its home cell is dimmed (so the board still reads as "n pieces, one
    just isn't home right now") and the piece itself is drawn as a floating
    thumbnail that tracks the fingertip.
    """
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    cell = size // n

    skip_positions = set()
    if anim is not None:
        skip_positions.add(anim["pos_a"])
        skip_positions.add(anim["pos_b"])
    if held is not None:
        skip_positions.add(held["from_pos"])

    for pos in range(n * n):
        tile_idx = board[pos]
        r, c = divmod(pos, n)
        y0, x0 = r * cell, c * cell
        if pos in skip_positions:
            if held is not None and pos == held["from_pos"]:
                tile_img = cv2.resize(tiles[tile_idx], (cell - TILE_GAP, cell - TILE_GAP))
                dimmed = (tile_img * 0.35).astype(np.uint8)
                canvas[y0:y0 + cell - TILE_GAP, x0:x0 + cell - TILE_GAP] = dimmed
            continue
        tile_img = cv2.resize(tiles[tile_idx], (cell - TILE_GAP, cell - TILE_GAP))
        canvas[y0:y0 + cell - TILE_GAP, x0:x0 + cell - TILE_GAP] = tile_img

    if anim is not None:
        t = anim["progress"]
        for tile_key, from_key, to_key in (("tile_a", "pos_a", "pos_b"),
                                            ("tile_b", "pos_b", "pos_a")):
            tile_img = cv2.resize(tiles[anim[tile_key]], (cell - TILE_GAP, cell - TILE_GAP))
            fr, fc = divmod(anim[from_key], n)
            tr, tc = divmod(anim[to_key], n)
            y = int((fr + (tr - fr) * t) * cell)
            x = int((fc + (tc - fc) * t) * cell)
            y1 = min(size, y + cell - TILE_GAP)
            x1 = min(size, x + cell - TILE_GAP)
            h_avail, w_avail = y1 - y, x1 - x
            if h_avail > 0 and w_avail > 0:
                canvas[y:y1, x:x1] = tile_img[:h_avail, :w_avail]

    # grid lines + hover highlight
    for i in range(n + 1):
        cv2.line(canvas, (0, i * cell), (size, i * cell), (60, 60, 60), 1)
        cv2.line(canvas, (i * cell, 0), (i * cell, size), (60, 60, 60), 1)

    if hover_cell is not None:
        hr, hc = hover_cell
        color = DROP_TARGET_COLOR if held is not None else HOVER_COLOR
        cv2.rectangle(canvas, (hc * cell, hr * cell),
                       ((hc + 1) * cell, (hr + 1) * cell), color, 3)

    if held is not None:
        piece_size = max(1, int((cell - TILE_GAP) * HELD_SCALE))
        tile_img = cv2.resize(tiles[held["tile"]], (piece_size, piece_size))
        fx = int(np.clip(held["fx"], piece_size // 2, size - piece_size // 2))
        fy = int(np.clip(held["fy"], piece_size // 2, size - piece_size // 2))
        x0, y0 = fx - piece_size // 2, fy - piece_size // 2
        canvas[y0:y0 + piece_size, x0:x0 + piece_size] = tile_img
        cv2.rectangle(canvas, (x0, y0), (x0 + piece_size, y0 + piece_size),
                       HELD_BORDER_COLOR, 3)

    return canvas


def draw_text(img, text, org, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def compose_window(cam_frame, puzzle_panel, status_lines):
    """Stitch the camera preview and puzzle panel side by side, plus a
    status bar of instruction text along the bottom.
    """
    cam_resized = cv2.resize(
        cam_frame, (CAM_PANEL_WIDTH, int(CAM_PANEL_WIDTH * cam_frame.shape[0] / cam_frame.shape[1]))
    )
    panel_h = max(puzzle_panel.shape[0], cam_resized.shape[0])
    bar_h = 30 * max(1, len(status_lines)) + 10

    out = np.full((panel_h + bar_h, puzzle_panel.shape[1] + cam_resized.shape[1], 3),
                   20, dtype=np.uint8)
    out[0:puzzle_panel.shape[0], 0:puzzle_panel.shape[1]] = puzzle_panel
    out[0:cam_resized.shape[0], puzzle_panel.shape[1]:puzzle_panel.shape[1] + cam_resized.shape[1]] = cam_resized

    for i, line in enumerate(status_lines):
        draw_text(out, line, (10, panel_h + 25 + i * 28), scale=0.55)

    return out


# ==========================================================================
# Leaderboard (stretch goal)
# ==========================================================================

def load_leaderboard():
    if not os.path.exists(LEADERBOARD_FILE):
        return []
    try:
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_leaderboard_entry(grid_size, elapsed_seconds, moves):
    board_data = load_leaderboard()
    board_data.append({
        "grid_size": grid_size,
        "time_seconds": round(elapsed_seconds, 2),
        "moves": moves,
        "timestamp": time.time(),
    })
    # keep best 10 per grid size, sorted by time
    try:
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(board_data, f, indent=2)
    except OSError:
        pass


def best_for_grid(grid_size):
    entries = [e for e in load_leaderboard() if e.get("grid_size") == grid_size]
    if not entries:
        return None
    return min(entries, key=lambda e: e["time_seconds"])


# ==========================================================================
# Main application class -- owns the camera, MediaPipe models, and the
# state machine loop. The heavy logic above stays free of any of this so it
# can be unit-tested / reasoned about in isolation.
# ==========================================================================

class SelfiePuzzleApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6)
        self.hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.6,
            min_tracking_confidence=0.5)

        self.state = STATE_DIFFICULTY
        self.grid_size = DEFAULT_GRID_SIZE

        self.target_image = None
        self.tiles = None
        self.board = None

        self.held_pos = None           # flat board index of the piece
                                        # currently picked up, or None
        self._prev_pinch_active = False  # for edge-triggered pick-up/drop
        self._smoothed_finger = None   # exponentially-smoothed (x, y)
        self._locked_cell = None       # target cell frozen while pinching

        self.last_move_time = 0.0
        self.move_count = 0
        self.start_time = None
        self.win_time = None

        self.anim = None  # active swap animation, or None

        # capture-state bookkeeping
        self._steady_since = None
        self._countdown_start = None

        # win-state gesture confirmation bookkeeping
        self._win_gesture_since = None
        self._win_gesture_kind = None

        # difficulty-state bookkeeping
        self._difficulty_since = None
        self._difficulty_choice = None

    # ------------------------------------------------------------------
    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Camera read failed -- exiting.")
                    break
                frame = cv2.flip(frame, 1)  # mirror for natural selfie view

                if self.state == STATE_DIFFICULTY:
                    display = self._step_difficulty(frame)
                elif self.state == STATE_CAPTURE:
                    display = self._step_capture(frame)
                elif self.state == STATE_SETUP:
                    self._do_setup()
                    display = frame
                elif self.state == STATE_GAMEPLAY:
                    display = self._step_gameplay(frame)
                elif self.state == STATE_WIN:
                    display = self._step_win(frame)
                else:
                    display = frame

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                self._handle_key(key)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.face_detector.close()
            self.hands.close()

    # ------------------------------------------------------------------
    def _handle_key(self, key):
        if key == ord(' '):
            if self.state == STATE_DIFFICULTY:
                self.state = STATE_CAPTURE
            elif self.state == STATE_CAPTURE:
                self._countdown_start = time.time() - COUNTDOWN_SECONDS - 1  # force fire

    # ------------------------------------------------------------------
    # DIFFICULTY state: hold up 1/2/3 fingers to pick 3x3/4x4/5x5
    # ------------------------------------------------------------------
    def _step_difficulty(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        chosen = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = "Right"
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            n_fingers = count_extended_fingers(hand_landmarks, handedness)
            if 1 <= n_fingers <= 3:
                chosen = n_fingers + 2  # 1->3x3, 2->4x4, 3->5x5

        now = time.time()
        if chosen is not None and chosen == self._difficulty_choice:
            if self._difficulty_since is None:
                self._difficulty_since = now
            elif now - self._difficulty_since >= WIN_GESTURE_HOLD_SECONDS:
                self.grid_size = chosen
                self.state = STATE_CAPTURE
        else:
            self._difficulty_choice = chosen
            self._difficulty_since = now if chosen is not None else None

        draw_text(frame, "Choose difficulty: hold up 1, 2, or 3 fingers", (10, 30))
        draw_text(frame, "(1=3x3  2=4x4  3=5x5)  or press SPACE for default 3x3", (10, 55), scale=0.55)
        if chosen is not None:
            draw_text(frame, f"Detected: {chosen}x{chosen}  (hold steady...)", (10, 85), color=(0, 220, 255))
        return frame

    # ------------------------------------------------------------------
    # CAPTURE state: wait for a steady, centered face, then auto-capture
    # ------------------------------------------------------------------
    def _step_capture(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)
        fh, fw = frame.shape[:2]

        face_bbox = None
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * fw)
            y = int(bbox.ymin * fh)
            w = int(bbox.width * fw)
            h = int(bbox.height * fh)
            face_bbox = (max(0, x), max(0, y), w, h)
            cv2.rectangle(frame, (face_bbox[0], face_bbox[1]),
                          (face_bbox[0] + w, face_bbox[1] + h), (0, 255, 0), 2)

        now = time.time()
        centered = False
        if face_bbox is not None:
            cx = (face_bbox[0] + face_bbox[2] / 2) / fw
            cy = (face_bbox[1] + face_bbox[3] / 2) / fh
            centered = (abs(cx - 0.5) < FACE_CENTER_TOLERANCE + face_bbox[2] / fw / 2
                        and abs(cy - 0.5) < FACE_CENTER_TOLERANCE + face_bbox[3] / fh / 2)

        if centered:
            if self._steady_since is None:
                self._steady_since = now
            if self._countdown_start is None and now - self._steady_since >= FACE_STEADY_SECONDS:
                self._countdown_start = now
        else:
            self._steady_since = None
            # don't cancel an already-started countdown just for a jitter frame

        draw_text(frame, "Center your face in the box and hold still", (10, 30))
        draw_text(frame, "(SPACE = capture now)", (10, 55), scale=0.55)

        if self._countdown_start is not None:
            elapsed = now - self._countdown_start
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining > 0:
                draw_text(frame, str(int(remaining) + 1), (fw // 2 - 20, fh // 2),
                          scale=3, color=(0, 220, 255), thickness=4)
            else:
                # fire capture
                if face_bbox is not None:
                    self.target_image = crop_face_square(frame, face_bbox)
                else:
                    # SPACE was pressed with no face -- just grab center square
                    side = min(fw, fh)
                    x0 = (fw - side) // 2
                    y0 = (fh - side) // 2
                    self.target_image = frame[y0:y0 + side, x0:x0 + side].copy()
                self._countdown_start = None
                self._steady_since = None
                self.state = STATE_SETUP

        return frame

    # ------------------------------------------------------------------
    def _do_setup(self):
        n = self.grid_size
        self.tiles = build_puzzle(self.target_image, n)
        self.board = shuffle_board(n)
        # guard against a (astronomically unlikely) already-solved shuffle
        while check_win(self.board, n):
            self.board = shuffle_board(n)

        self.move_count = 0
        self.start_time = time.time()
        self.held_pos = None
        self._prev_pinch_active = False
        self._smoothed_finger = None
        self._locked_cell = None
        self.state = STATE_GAMEPLAY

    # ------------------------------------------------------------------
    # GAMEPLAY state: pinch a piece to pick it up, pinch a cell to drop it
    # ------------------------------------------------------------------
    def _step_gameplay(self, frame):
        n = self.grid_size
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hover_cell = None
        finger_norm = None
        pinch_active = False
        pinch_ratio = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_drawing_styles.get_default_hand_landmarks_style(),
                                       mp_drawing_styles.get_default_hand_connections_style())
            index_tip = hand_landmarks.landmark[LM_INDEX_TIP]
            pinch_active, pinch_ratio = detect_pinch(hand_landmarks)

            # Smooth the raw landmark position a little -- per-frame
            # detector noise on its own is enough to flip which cell you're
            # over when your fingertip sits near a cell boundary.
            raw_x, raw_y = index_tip.x, index_tip.y
            if self._smoothed_finger is None:
                self._smoothed_finger = (raw_x, raw_y)
            else:
                sx, sy = self._smoothed_finger
                a = FINGER_SMOOTHING_ALPHA
                self._smoothed_finger = (sx + a * (raw_x - sx), sy + a * (raw_y - sy))
            finger_norm = self._smoothed_finger
            hover_cell = map_hand_to_grid(finger_norm[0], finger_norm[1], n)
        else:
            # hand lost -- don't keep smoothing toward a stale position, and
            # don't hold onto a half-formed lock from before it disappeared
            self._smoothed_finger = None
            self._locked_cell = None

        now = time.time()
        # advance any in-flight swap animation
        if self.anim is not None:
            self.anim["progress"] += 1.0 / SWAP_ANIM_FRAMES
            if self.anim["progress"] >= 1.0:
                self.anim = None

        # Freeze the *target* cell as soon as the fingers start closing
        # (pinch_ratio dropping below PINCH_LOCK_RATIO, well before the
        # pinch actually triggers at PINCH_THRESHOLD_RATIO). The last bit of
        # motion while closing a pinch curls the index fingertip in toward
        # the thumb, which can shift it onto a neighboring cell right as the
        # pinch fires -- locking early means that drift no longer matters.
        # While the hand is open/relaxed we keep tracking normally.
        if hover_cell is not None:
            closing = pinch_ratio is not None and pinch_ratio < PINCH_LOCK_RATIO
            if closing:
                if self._locked_cell is None:
                    self._locked_cell = hover_cell
            else:
                self._locked_cell = hover_cell

        action_cell = self._locked_cell if self._locked_cell is not None else hover_cell

        # edge-trigger on the *start* of a pinch, not the whole time it's
        # held -- otherwise picking up a piece and immediately holding
        # still would also count as (accidentally) starting a drop.
        pinch_started = pinch_active and not self._prev_pinch_active
        self._prev_pinch_active = pinch_active

        if (pinch_started and action_cell is not None and self.anim is None
                and now - self.last_move_time > MOVE_COOLDOWN_SECONDS):
            self._try_pick_or_drop(action_cell)
            self.last_move_time = now

        held_render = None
        if self.held_pos is not None:
            if finger_norm is not None:
                fx = finger_norm[0] * PUZZLE_PANEL_SIZE
                fy = finger_norm[1] * PUZZLE_PANEL_SIZE
            else:
                # hand briefly lost -- keep floating the piece over its
                # last known cell instead of snapping it back
                r, c = divmod(self.held_pos, n)
                cell = PUZZLE_PANEL_SIZE // n
                fx = c * cell + cell / 2
                fy = r * cell + cell / 2
            held_render = {
                "from_pos": self.held_pos,
                "tile": self.board[self.held_pos],
                "fx": fx,
                "fy": fy,
            }

        panel = render_puzzle_panel(self.tiles, self.board, n,
                                     hover_cell=action_cell, anim=self.anim,
                                     held=held_render)

        elapsed = now - self.start_time
        if self.held_pos is not None:
            instruction = "Point at a cell and pinch to drop the piece there"
        else:
            instruction = "Point at a piece and pinch (thumb+index) to pick it up"
        status = [
            instruction,
            f"Moves: {self.move_count}    Time: {elapsed:0.1f}s"
            + (f"    Pinch: {pinch_ratio:.2f}" if pinch_ratio is not None else ""),
        ]
        if results.multi_hand_landmarks is None:
            status.insert(0, "Show your hand to the camera")

        composed = compose_window(frame, panel, status)

        if check_win(self.board, n):
            self.win_time = now
            self.state = STATE_WIN

        return composed

    def _try_pick_or_drop(self, hover_cell):
        n = self.grid_size
        r, c = hover_cell
        pos = r * n + c

        if self.held_pos is None:
            # nothing in hand yet -- pick up whatever piece sits here
            self.held_pos = pos
            return

        if pos == self.held_pos:
            # pinched the same cell you picked up from -- put it back down
            # without spending a move
            self.held_pos = None
            return

        # drop -- swap the held piece into this cell, and send whichever
        # piece was already here back to the held piece's old home
        tile_a, tile_b = self.board[self.held_pos], self.board[pos]
        self.anim = {
            "pos_a": self.held_pos, "tile_a": tile_a,
            "pos_b": pos, "tile_b": tile_b,
            "progress": 0.0,
        }
        self.board[self.held_pos], self.board[pos] = tile_b, tile_a
        self.move_count += 1
        self.held_pos = None

    # ------------------------------------------------------------------
    # WIN state: show the finished selfie + stats, gesture to replay/quit
    # ------------------------------------------------------------------
    def _step_win(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = "Right"
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_open_palm(hand_landmarks, handedness):
                gesture = "again"
            elif is_fist(hand_landmarks, handedness):
                gesture = "quit"

        now = time.time()
        if gesture is not None and gesture == self._win_gesture_kind:
            if self._win_gesture_since is None:
                self._win_gesture_since = now
            elif now - self._win_gesture_since >= WIN_GESTURE_HOLD_SECONDS:
                if gesture == "again":
                    self._reset_for_replay()
                else:
                    self.cap.release()
                    cv2.destroyAllWindows()
                    self.face_detector.close()
                    self.hands.close()
                    raise SystemExit(0)
        else:
            self._win_gesture_kind = gesture
            self._win_gesture_since = now if gesture is not None else None

        panel = cv2.resize(self.target_image, (PUZZLE_PANEL_SIZE, PUZZLE_PANEL_SIZE))
        elapsed = self.win_time - self.start_time
        best = best_for_grid(self.grid_size)

        if not self._leaderboard_saved:
            save_leaderboard_entry(self.grid_size, elapsed, self.move_count)
            self._leaderboard_saved = True

        status = [
            f"Solved! Time: {elapsed:0.1f}s   Moves: {self.move_count}",
            f"Best for {self.grid_size}x{self.grid_size}: "
            + (f"{best['time_seconds']:0.1f}s / {best['moves']} moves" if best else "-"),
            "Open palm = play again    Fist = quit",
        ]
        return compose_window(frame, panel, status)

    def _reset_for_replay(self):
        self.state = STATE_DIFFICULTY
        self.target_image = None
        self.tiles = None
        self.board = None
        self.held_pos = None
        self._prev_pinch_active = False
        self._smoothed_finger = None
        self._locked_cell = None
        self.move_count = 0
        self.start_time = None
        self.win_time = None
        self.anim = None
        self._win_gesture_since = None
        self._win_gesture_kind = None
        self._difficulty_since = None
        self._difficulty_choice = None
        self._leaderboard_saved = False

    _leaderboard_saved = False  # class-level default, per-instance after first win


if __name__ == "__main__":
    app = SelfiePuzzleApp()
    app.run()
