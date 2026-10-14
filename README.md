# Hand-Gesture Controlled Selfie Merge Puzzle

Take a selfie with your webcam, watch it get sliced into a grid of puzzle
pieces, and reassemble it into the full picture using only your hand — no
mouse, no keyboard.

## Setup

```bash
pip install -r requirements.txt
python selfie_puzzle.py
```

A single OpenCV window will open. Press **q** at any time to quit.

> **Note on the `mediapipe` version pin:** newer `mediapipe` wheels (0.10.2x+)
> dropped the legacy `mp.solutions` API (Hands / FaceDetection) that this
> project uses, in favor of the newer Tasks API. `requirements.txt` pins
> `mediapipe==0.10.14`, the last release that reliably ships `solutions` on
> most desktop platforms. If you want the newer Tasks API instead, the CV
> logic in this file (`detect_pinch`, `count_extended_fingers`, etc.) is
> already decoupled from MediaPipe's model-loading code, so only the
> `SelfiePuzzleApp.__init__`/`_step_*` methods that call `self.hands.process`
> / `self.face_detector.process` would need updating.

## How to play

1. **Difficulty** — hold up 1 / 2 / 3 fingers to pick a 3×3 / 4×4 / 5×5
   grid, or press SPACE for the 3×3 default.
2. **Capture** — center your face in the green box and hold still; a
   3‑2‑1 countdown fires automatically. SPACE forces an immediate capture.
3. **Merge** — point your index finger at a piece and pinch (touch thumb
   and index fingertip together) to pick it up. Move your hand to another
   cell and pinch again to drop it there — this swaps it with whatever
   piece was already sitting in that cell. Pinch the same cell you picked
   up from to put the piece back down without spending a move. Keep going
   until every piece has merged back into its correct spot.
4. **Win** — your reconstructed selfie, solve time, and move count are
   shown. Show an **open palm** to play again or a **fist** to quit.

Your best time per grid size is saved to `leaderboard.json` next to the
script.

## Code layout

Logic is deliberately split so each concern can be pointed to on its own
(handy for walking through in an interview):

| Function | Responsibility |
|---|---|
| `capture_selfie` flow (in `_step_capture`) | Face detection, steady/centered check, countdown, crop |
| `crop_face_square` | Pads and squares the face bounding box for tiling |
| `build_puzzle(image, n)` | Slices the selfie into an n×n list of pieces (no reserved blank) |
| `shuffle_board(n)` | Random permutation of piece positions to start a round |
| `detect_pinch(hand_landmarks)` | Thumb/index distance, normalized by hand size |
| `map_hand_to_grid(x, y, n)` | Maps a normalized fingertip position to a grid cell |
| `_try_pick_or_drop(hover_cell)` | Pick-up/drop/swap state machine for a single pinch event |
| `check_win(board, n)` | Compares board to the solved order |
| `count_extended_fingers` / `is_open_palm` / `is_fist` | Gesture heuristics for difficulty select and the win-screen prompt |
| `render_puzzle_panel` / `compose_window` | All drawing, including the held-piece overlay and swap animation |

### Design notes / known limitations

- **Shuffle correctness**: a sliding puzzle has to shuffle via legal slide
  moves, because roughly half of raw permutations of an (n²−1)-puzzle are
  provably unsolvable. That constraint doesn't exist here — since you can
  pick up *any* piece and drop it into *any* cell (swapping the two),
  every permutation is reachable, so `shuffle_board` is a plain
  `random.shuffle`.
- **Pick-up/drop is edge-triggered, not a continuous hold-drag**: a pinch
  is detected once per "squeeze" (the transition from not-pinching to
  pinching), not for as long as the pinch is held. So the flow is
  pinch-to-grab, release, move your hand, pinch-again-to-drop — rather
  than holding a pinch shut while dragging across the frame. This was a
  deliberate choice: maintaining a tight pinch while also moving your
  hand accurately is harder on noisy hand-tracking than two discrete
  pinch taps. If you'd rather have true continuous drag, the hook point
  is `_step_gameplay`'s `pinch_started` logic — swap it for "pick up on
  pinch-start, drop on pinch-release" instead.
- **Pinch threshold** is normalized by an estimate of on-screen hand size
  (wrist → middle-finger-MCP distance) rather than raw pixel distance, so
  the same `PINCH_THRESHOLD_RATIO` constant works whether your hand is
  close to or far from the camera. Tune it at the top of the file if pinch
  detection feels too sensitive or too strict.
- **Finger counting** (used for the difficulty selector and win-screen
  gestures) is a simple tip-vs-pip heuristic, not a trained classifier —
  it's reliable for "1 vs 2 vs 3 vs open palm vs fist" but isn't meant to
  be a general-purpose gesture recognizer.
- **Hand→grid mapping** treats the *entire camera frame* as the
  interaction surface (an "air trackpad"), so moving your fingertip to the
  top-left of frame hovers the top-left cell. This keeps the mapping a
  simple linear scale instead of needing a separate calibration step.
- **Held-piece visuals**: while a piece is picked up, its home cell is
  dimmed on the board and the piece itself floats as a thumbnail that
  tracks your fingertip, with the hover-cell outline switching color to
  show it's now a drop target rather than just a hover. Dropping animates
  both swapped pieces trading places at once.
- Held-piece drag visuals, difficulty-by-gesture, and the JSON leaderboard
  are all implemented from the stretch-goal list. Sound effects were left
  out since this environment has no reliable audio backend to test
  against — the natural hook is inside `_try_pick_or_drop` (on pick-up/
  drop) and the win branch of `_step_win` (on solve) if you want to add
  e.g. `playsound` yourself.
