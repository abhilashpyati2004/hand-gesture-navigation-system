
#---------------------------Gesture Controlled Navigation System---------------------------------


import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
import pyautogui
import screen_brightness_control as sbc
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key


# CONFIGURATION


SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# ---- Pinch Gesture Controller Settings ----
PINCH_TOUCH_THRESH     = 0.045
ZOOM_HOLD_TIME         = 0.30
ZOOM_SENSITIVITY       = 400
SWIPE_VELOCITY_THRESH  = 0.6
SCREENSHOT_PATTERN     = "screenshot_{}.png"
EXT_MARGIN             = 0.02

TWOHAND_SMOOTH_WINDOW  = 6
FAST_CLICK_TIME        = 0.25
DRAG_HOLD_TIME         = 0.15
SCROLL_STEP            = 30
SCROLL_INTERVAL        = 0.06
HUD_DURATION           = 0.8

# Drag OPTIMIZATION
drag_release_start = 0
DRAG_RELEASE_GRACE = 0.06
drag_intent = False


# ---- MODE (Volume / Brightness) Settings ----
VOLBR_MODE_HOLD = 2.3

# VOLUME/BRIGHTNESS ANGLE/DISTANCE BEHAVIOR
ANGLE_SMOOTH = 0.55
DIST_SMOOTH  = 0.45

H_DEG = 35     # horizontal angle threshold = Volume
V_DEG = 55     # vertical angle threshold   = Brightness

DIST_MIN = 0.07
DIST_MAX = 0.33

VOL_CENTER = 50
VOL_UPDATE_DELAY = 0.025

BR_MIN = 10
BR_MAX = 90
BR_EASE = 0.22
BR_UPDATE_DELAY = 0.035

# ---- ARROW MODE SETTINGS ----
ARROW_MODE_HOLD_TIME = 1.5
ARROW_PINCH_THRESH  = 0.04
ARROW_ACTION_COOLDOWN = 0.6



# INITIALIZATION


mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
ids      = mp_hands.HandLandmark

hands_module = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mouse    = MouseController()
keyboard = KeyboardController()


# HELPERS


def norm_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def finger_extended_strict(hand, tip_idx, mcp_idx):
    tip = hand.landmark[tip_idx]
    mcp = hand.landmark[mcp_idx]
    return tip.y < mcp.y - EXT_MARGIN

def index_extended_and_others_closed(hand):
    try:
        ids = mp_hands.HandLandmark
        return (
            finger_extended_strict(hand, ids.INDEX_FINGER_TIP, ids.INDEX_FINGER_MCP) and
            not finger_extended_strict(hand, ids.MIDDLE_FINGER_TIP, ids.MIDDLE_FINGER_MCP) and
            not finger_extended_strict(hand, ids.RING_FINGER_TIP,   ids.RING_FINGER_MCP) and
            not finger_extended_strict(hand, ids.PINKY_TIP,         ids.PINKY_MCP)
        )
    except:
        return False

def simple_pinch_index(hand):
    try:
        ids = mp_hands.HandLandmark
        return norm_dist(hand.landmark[ids.THUMB_TIP],
                         hand.landmark[ids.INDEX_FINGER_TIP]) < PINCH_TOUCH_THRESH
    except:
        return False

def simple_pinch_middle(hand):
    try:
        ids = mp_hands.HandLandmark
        return norm_dist(hand.landmark[ids.THUMB_TIP],
                         hand.landmark[ids.MIDDLE_FINGER_TIP]) < PINCH_TOUCH_THRESH
    except:
        return False


# MODE V-SIGN DETECTION

def strict_vsign(lm):
    # Two fingers up, two fingers down
    idx = lm[ids.INDEX_FINGER_TIP].y < lm[ids.INDEX_FINGER_PIP].y - 0.02
    mid = lm[ids.MIDDLE_FINGER_TIP].y < lm[ids.MIDDLE_FINGER_PIP].y - 0.02
    rng = not (lm[ids.RING_FINGER_TIP].y < lm[ids.RING_FINGER_PIP].y - 0.015)
    pnk = not (lm[ids.PINKY_TIP].y       < lm[ids.PINKY_PIP].y       - 0.015)
    if not (idx and mid and rng and pnk): return False

    # Angle between index/middle
    i   = lm[ids.INDEX_FINGER_TIP]
    m   = lm[ids.MIDDLE_FINGER_TIP]
    mcp = lm[ids.INDEX_FINGER_MCP]

    v1 = np.array([i.x - mcp.x, i.y - mcp.y])
    v2 = np.array([m.x - mcp.x, m.y - mcp.y])
    dot = v1.dot(v2)
    mag = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag == 0: return False

    ang = math.degrees(math.acos(dot / mag))
    if ang < 25 or ang > 55: return False

    if norm_dist(i, m) < 0.045: return False
    return True


# BRIGHTNESS HELPERS

def brightness_get():
    try: return sbc.get_brightness()[0]
    except: return None

def brightness_set(v):
    try: sbc.set_brightness(int(max(BR_MIN, min(BR_MAX, v))))
    except: pass


# HUD TEXT SYSTEM

hud_enabled = True
hud_text    = ""
hud_last_time = 0

def set_hud(msg):
    global hud_text, hud_last_time
    hud_text = msg
    hud_last_time = time.time()


# STATE VARIABLES

mode_active = False
vsign_start = 0

angle_filtered = 0
dist_filtered  = 0

last_vol_time  = 0
last_br_time   = 0

# Original pinch gesture controller states
twohand_dist_smoother   = deque(maxlen=TWOHAND_SMOOTH_WINDOW)
hand_positions_for_swipe= deque(maxlen=6)

last_swipe_time         = 0
pinch_dragging          = False
drag_hold_start         = 0

zoom_mode               = False
zoom_hold_start         = 0
zoom_release_time       = 0

screenshot_count        = 0
screenshot_hold_start   = 0

index_pinch_prev        = False
index_pinch_start       = 0
middle_pinch_prev       = False
middle_pinch_start      = 0

last_index_tap_time     = 0
last_scroll_time        = 0

arrow_mode = False
arrow_mode_start = 0
last_arrow_action = 0
arrow_progress = 0.0
arrow_toggle_cooldown = 0


#---------MAIN LOOP + MODE SYSTEM + VOLUME / BRIGHTNESS


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("\nMerged Gesture Controller Running. Press Q to quit, H to toggle HUD.\n")

try:
    while True:

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_module.process(rgb)

        hands = results.multi_hand_landmarks or []
        handedness = results.multi_handedness or []

        # Determine left & right hands
        left_hand = None
        right_hand = None

        for i, handinfo in enumerate(handedness):
            label = handinfo.classification[0].label
            if label == "Left":
                left_hand = hands[i]
            else:
                right_hand = hands[i]

        # ------------ARROW mode-------------
        arrow_detected = False

        if left_hand and not mode_active:
            lm = left_hand.landmark

            thumb  = lm[ids.THUMB_TIP]
            index  = lm[ids.INDEX_FINGER_TIP]
            middle = lm[ids.MIDDLE_FINGER_TIP]
            ring   = lm[ids.RING_FINGER_TIP]
            pinky  = lm[ids.PINKY_TIP]

            middle_ring_joined = norm_dist(middle, ring) < 0.035
            index_separated    = norm_dist(index, middle) > 0.06
            pinky_separated    = norm_dist(pinky, ring) > 0.06

            arrow_gesture = (
                middle_ring_joined and
                index_separated and
                pinky_separated
            )

            if arrow_gesture:
                arrow_detected = True
                if arrow_mode_start == 0:
                    arrow_mode_start = now
                else:
                    arrow_progress = min(
                        (now - arrow_mode_start) / ARROW_MODE_HOLD_TIME,
                        1.0
                    )

                if arrow_progress >= 1.0 and now - arrow_toggle_cooldown > 0.6:
                    arrow_mode = not arrow_mode

                    zoom_mode = False
                    pinch_dragging = False
                    drag_intent = False
                    hand_positions_for_swipe.clear()

                    arrow_mode_start = 0
                    arrow_progress = 0
                    arrow_toggle_cooldown = now
            else:
                arrow_mode_start = 0
                arrow_progress = 0

        if not arrow_detected:
            arrow_progress = 0  

        # DRAW HANDS
        for hnd in hands:
            mp_draw.draw_landmarks(frame, hnd, mp_hands.HAND_CONNECTIONS)

        
        # MODE TOGGLE USING STRICT V-SIGN (LEFT HAND)
        
        if left_hand and not arrow_detected and not arrow_mode:
            lm = left_hand.landmark
            if strict_vsign(lm):

                # Show "MODE HOLD" while holding
                if vsign_start == 0:
                    vsign_start = now

                hold = now - vsign_start
                cv2.putText(frame, f"MODE HOLD: {hold:.1f}s",
                            (10, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

                # Toggle MODE after required hold
                if hold >= VOLBR_MODE_HOLD:
                    mode_active = not mode_active
                    vsign_start = 0
                    zoom_mode = False   # zoom disabled when mode ON
                    angle_filtered = 0
                    dist_filtered = 0
            else:
                vsign_start = 0

        # Display MODE ON text ONLY when mode is true
        if mode_active:
            cv2.putText(frame, "MODE ON", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 3)

        
        # MODE ON  → ONLY VOLUME AND BRIGHTNESS (DISABLE ALL OTHER FEATURES)
        
        if mode_active:

            if left_hand and right_hand:

                # landmarks
                iL = left_hand.landmark[ids.INDEX_FINGER_TIP]
                tL = left_hand.landmark[ids.THUMB_TIP]

                iR = right_hand.landmark[ids.INDEX_FINGER_TIP]
                tR = right_hand.landmark[ids.THUMB_TIP]

                # pinch detection for both hands
                pinchL = norm_dist(iL, tL) < PINCH_TOUCH_THRESH
                pinchR = norm_dist(iR, tR) < PINCH_TOUCH_THRESH

                if pinchL and pinchR:

                    # draw line between index tips
                    cv2.line(frame,
                             (int(iL.x * w), int(iL.y * h)),
                             (int(iR.x * w), int(iR.y * h)),
                             (0, 255, 0), 2)

                    dx = iR.x - iL.x
                    dy = iR.y - iL.y

                    raw_angle = abs(math.degrees(math.atan2(dy, dx)))
                    if raw_angle > 90:
                        raw_angle = 180 - raw_angle

                    raw_dist = norm_dist(iL, iR)

                    # Smooth angle + distance
                    
                    angle_filtered = ANGLE_SMOOTH * angle_filtered + (1 - ANGLE_SMOOTH) * raw_angle
                    dist_filtered  = DIST_SMOOTH  * dist_filtered  + (1 - DIST_SMOOTH ) * raw_dist

                    # Convert dist → 0–100%
                    d = max(DIST_MIN, min(DIST_MAX, dist_filtered))
                    logical = int(((d - DIST_MIN) / (DIST_MAX - DIST_MIN)) * 100)

                    
                    # VOLUME CONTROL (ANGLE < H_DEG)
                    
                    if angle_filtered <= H_DEG:

                        cv2.putText(frame, "VOLUME",
                                    (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2)

                        if now - last_vol_time > VOL_UPDATE_DELAY:

                            if logical > VOL_CENTER + 5:
                                pyautogui.press("volumeup")

                            elif logical < VOL_CENTER - 5:
                                pyautogui.press("volumedown")

                            last_vol_time = now

                    
                    # BRIGHTNESS CONTROL (ANGLE > V_DEG)
                    
                    elif angle_filtered >= V_DEG:

                        cv2.putText(frame, f"BRIGHTNESS {logical}%",
                                    (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 200, 255), 2)

                        if now - last_br_time > BR_UPDATE_DELAY:

                            cur = brightness_get()
                            if cur is not None:
                                target = BR_MIN + (logical / 100) * (BR_MAX - BR_MIN)
                                eased  = cur + BR_EASE * (target - cur)
                                brightness_set(eased)

                            last_br_time = now

                        # Draw brightness bar
                        bar_x, bar_y = 20, 160
                        bw = 220
                        fill = bar_x + int(bw * (logical / 100))
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bw, bar_y + 20),
                                      (255, 255, 255), 2)
                        cv2.rectangle(frame, (bar_x, bar_y), (fill, bar_y + 20),
                                      (0, 200, 255), -1)

            
            # DISPLAY FRAME + continue (skip all other gestures)
            
            cv2.imshow("Gesture Controller", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("h"): hud_enabled = not hud_enabled
            continue   # << SKIP MODE OFF section

       
        # END MODE ON BLOCK
    

#--------  MODE OFF GESTURES (Pointer, Scroll, Drag, Click,
#           Alt-tab swipe, Zoom, Screenshot, HUD, Final Loop)


        
        # MODE OFF  
        

        n = len(hands)
        primary = hands[0] if n >= 1 else None

        left_arrow = False
        right_arrow = False

        if arrow_mode and primary and not mode_active:

            tt = primary.landmark[ids.THUMB_TIP]
            mt = primary.landmark[ids.MIDDLE_FINGER_TIP]
            rt = primary.landmark[ids.RING_FINGER_TIP]

            left_pinch  = norm_dist(tt, mt) < ARROW_PINCH_THRESH
            right_pinch = norm_dist(tt, rt) < ARROW_PINCH_THRESH

            if left_pinch and now - last_arrow_action > ARROW_ACTION_COOLDOWN:
                keyboard.press(Key.left)
                keyboard.release(Key.left)
                last_arrow_action = now
                left_arrow = True

            elif right_pinch and now - last_arrow_action > ARROW_ACTION_COOLDOWN:
                keyboard.press(Key.right)
                keyboard.release(Key.right)
                last_arrow_action = now
                right_arrow = True


        # ---------------- ALT TAB SWIPE ----------------
        if primary and n == 1 and not arrow_mode:

            avg_x = np.mean([lm.x for lm in primary.landmark])
            hand_positions_for_swipe.append((now, avg_x))

            try:
                palm_open = (
                    finger_extended_strict(primary, ids.INDEX_FINGER_TIP, ids.INDEX_FINGER_MCP) and
                    finger_extended_strict(primary, ids.MIDDLE_FINGER_TIP, ids.MIDDLE_FINGER_MCP) and
                    finger_extended_strict(primary, ids.RING_FINGER_TIP,   ids.RING_FINGER_MCP) and
                    finger_extended_strict(primary, ids.PINKY_TIP,        ids.PINKY_MCP)        and
                    finger_extended_strict(primary, ids.THUMB_TIP,        ids.THUMB_MCP)
                )
            except:
                palm_open = False

            if palm_open and len(hand_positions_for_swipe) >= 3:
                t0, x0 = hand_positions_for_swipe[0]
                t1, x1 = hand_positions_for_swipe[-1]

                if (t1 - t0) > 0:
                    velocity = (x1 - x0) / (t1 - t0)

                    if abs(velocity) > SWIPE_VELOCITY_THRESH and now - last_swipe_time > 0.8:

                        if velocity > 0:
                            keyboard.press(Key.alt)
                            keyboard.press(Key.tab)
                            keyboard.release(Key.tab)
                            keyboard.release(Key.alt)
                        else:
                            keyboard.press(Key.alt)
                            keyboard.press(Key.shift)
                            keyboard.press(Key.tab)
                            keyboard.release(Key.tab)
                            keyboard.release(Key.shift)
                            keyboard.release(Key.alt)

                        last_swipe_time = now
                        set_hud("SWITCH")

        # ---------------- TWO HAND ZOOM ----------------
        def zoom_pinch_ok(hand):
            # Three fingers extended + index/thumb pinch
            try:
                mid_ok = finger_extended_strict(hand, ids.MIDDLE_FINGER_TIP, ids.MIDDLE_FINGER_MCP)
                ring_ok = finger_extended_strict(hand, ids.RING_FINGER_TIP, ids.RING_FINGER_MCP)
                pink_ok = finger_extended_strict(hand, ids.PINKY_TIP, ids.PINKY_MCP)

                pinch_ok = norm_dist(hand.landmark[ids.THUMB_TIP],
                                     hand.landmark[ids.INDEX_FINGER_TIP]) < PINCH_TOUCH_THRESH
                return mid_ok and ring_ok and pink_ok and pinch_ok
            except:
                return False

        if n >= 2 and not pinch_dragging and not arrow_mode:

            L = hands[0]
            R = hands[1]

            pinchL = zoom_pinch_ok(L)
            pinchR = zoom_pinch_ok(R)

            # Enter zoom
            if pinchL and pinchR and not zoom_mode:
                if zoom_hold_start == 0:
                    zoom_hold_start = now
                elif now - zoom_hold_start >= ZOOM_HOLD_TIME:
                    zoom_mode = True
                    zoom_release_time = 0
                    set_hud("ZOOM")
            else:
                zoom_hold_start = 0

            # Exit zoom
            if zoom_mode and (not pinchL or not pinchR):
                if zoom_release_time == 0:
                    zoom_release_time = now
                elif now - zoom_release_time > 0.4:
                    zoom_mode = False

            # Perform Zoom
            if zoom_mode:
                try:
                    p1 = L.landmark[ids.INDEX_FINGER_TIP]
                    p2 = R.landmark[ids.INDEX_FINGER_TIP]

                    dist_now = norm_dist(p1, p2)
                    twohand_dist_smoother.append(dist_now)

                    if len(twohand_dist_smoother) >= 2:
                        diff = twohand_dist_smoother[-1] - twohand_dist_smoother[-2]
                        if abs(diff) > 0.010:

                            amt = int(np.sign(diff) * min(abs(diff) * ZOOM_SENSITIVITY, 200))

                            keyboard.press(Key.ctrl)
                            pyautogui.scroll(amt)
                            keyboard.release(Key.ctrl)

                            bar = min(int(abs(diff) * 800), 250)
                            x0, y0 = 20, 200
                            col, txt = ((0,255,0), "ZOOM IN") if diff > 0 else ((0,0,255), "ZOOM OUT")

                            cv2.putText(frame, txt, (x0, y0 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                            cv2.rectangle(frame, (x0, y0), (x0 + 250, y0 + 20), (255,255,255), 2)
                            cv2.rectangle(frame, (x0, y0), (x0 + bar, y0 + 20), col, -1)

                except:
                    pass

        else:
            zoom_mode = False
            twohand_dist_smoother.clear()
            zoom_hold_start = 0

        # ---------------- SCROLL (Up / Down) ----------------
        scroll_active = False
        pointer_mode  = False
        click_mode    = False

        if primary:

            tip_i = primary.landmark[ids.INDEX_FINGER_TIP]
            tip_m = primary.landmark[ids.MIDDLE_FINGER_TIP]
            mcp_i = primary.landmark[ids.INDEX_FINGER_MCP]
            mcp_m = primary.landmark[ids.MIDDLE_FINGER_MCP]

            px, py = lm_to_px(tip_i, w, h)

            both_ext = (
                finger_extended_strict(primary, ids.INDEX_FINGER_TIP, ids.INDEX_FINGER_MCP) and
                finger_extended_strict(primary, ids.MIDDLE_FINGER_TIP, ids.MIDDLE_FINGER_MCP)
            )
            close_up = abs(tip_i.x - tip_m.x) < 0.03 and abs(tip_i.y - tip_m.y) < 0.03

            scroll_up_pose = both_ext and close_up

            idx_not_ext = not finger_extended_strict(primary, ids.INDEX_FINGER_TIP, ids.INDEX_FINGER_MCP)
            mid_not_ext = not finger_extended_strict(primary, ids.MIDDLE_FINGER_TIP, ids.MIDDLE_FINGER_MCP)

            idx_sl = (tip_i.y > mcp_i.y) and (tip_i.y < mcp_i.y + 0.12)
            mid_sl = (tip_m.y > mcp_m.y) and (tip_m.y < mcp_m.y + 0.12)

            both_sl = idx_not_ext and mid_not_ext and idx_sl and mid_sl
            close_dn = abs(tip_i.x - tip_m.x) < 0.05 and abs(tip_i.y - tip_m.y) < 0.06

            scroll_down_pose = both_sl and close_dn

            ring_curled  = not finger_extended_strict(primary, ids.RING_FINGER_TIP, ids.RING_FINGER_MCP)
            pinky_curled = not finger_extended_strict(primary, ids.PINKY_TIP, ids.PINKY_MCP)

            if (
                n == 1
                and ring_curled
                and pinky_curled
                and not pinch_dragging
                and not drag_intent
                and not zoom_mode
                and not arrow_mode
            ):


                if scroll_up_pose and now - last_scroll_time > SCROLL_INTERVAL:
                    scroll_active = True
                    pyautogui.scroll(SCROLL_STEP)
                    set_hud("SCROLL UP")

                    cv2.putText(frame, "Scroll Up", (px + 10, py + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (180,255,180), 2)

                    last_scroll_time = now

                elif scroll_down_pose and now - last_scroll_time > SCROLL_INTERVAL:
                    scroll_active = True
                    pyautogui.scroll(-SCROLL_STEP)
                    set_hud("SCROLL DOWN")

                    cv2.putText(frame, "Scroll Down", (px + 10, py + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (180,180,255), 2)

                    last_scroll_time = now

        # ---------------- POINTER MODE ----------------
        if primary:

            pointer_mode = (
                index_extended_and_others_closed(primary)
                and not zoom_mode
                and not scroll_active
                and not arrow_mode
            )


            tip_i = primary.landmark[ids.INDEX_FINGER_TIP]
            px, py = lm_to_px(tip_i, w, h)
            sx, sy = int(tip_i.x * SCREEN_WIDTH), int(tip_i.y * SCREEN_HEIGHT)

            if pointer_mode or pinch_dragging:
                try:
                    pyautogui.moveTo(sx, sy, 0.01)
                except:
                    pass
                cv2.circle(frame, (px, py), 8, (0,255,0), -1)

                if pointer_mode and not pinch_dragging:
                    set_hud("POINTER")
            else:
                cv2.circle(frame, (px, py), 8, (50,50,50), -1)

        # ---------------- DRAG MODE ----------------
        if primary:

            right_index_pinch = simple_pinch_index(primary)
            left_index_pinch  = simple_pinch_index(hands[1]) if n >= 2 else False

            drag_start_condition = (
                pointer_mode and
                right_index_pinch and
                not left_index_pinch and
                not scroll_active and
                not zoom_mode and
                not pinch_dragging
            )


            if drag_start_condition:
                drag_intent = True
                if drag_hold_start == 0:
                    drag_hold_start = now

                elif now - drag_hold_start >= DRAG_HOLD_TIME:

                    if not pinch_dragging:
                        pinch_dragging = True
                        drag_intent = False
                        try: mouse.press(Button.left)
                        except: pass
                        set_hud("DRAG")

            # Reset drag intent if pinch released before drag starts
            if not right_index_pinch and not pinch_dragging:
                drag_intent = False
                drag_hold_start = 0


            if pinch_dragging:
                cv2.putText(frame, "Dragging", (px + 10, py + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255,120,0), 2)

            # -------- STABLE DRAG RELEASE --------
            if pinch_dragging:
                if not right_index_pinch:
                    if drag_release_start == 0:
                        drag_release_start = now
                    elif now - drag_release_start > DRAG_RELEASE_GRACE:
                        pinch_dragging = False
                        drag_hold_start = 0
                        drag_release_start = 0
                        drag_intent = False
                        try:
                            mouse.release(Button.left)
                        except:
                            pass
                else:
                    drag_release_start = 0


        # ---------------- CLICK SYSTEM ----------------
        if primary and not pointer_mode and not pinch_dragging and not scroll_active and not zoom_mode and not arrow_mode:

            ring_ext  = finger_extended_strict(primary, ids.RING_FINGER_TIP, ids.RING_FINGER_MCP)
            pinky_ext = finger_extended_strict(primary, ids.PINKY_TIP, ids.PINKY_MCP)

            if not (ring_ext and pinky_ext):
                index_pinch_prev = False
                middle_pinch_prev = False
            else:

                tt = primary.landmark[ids.THUMB_TIP]
                it = primary.landmark[ids.INDEX_FINGER_TIP]
                mt = primary.landmark[ids.MIDDLE_FINGER_TIP]

                idx_now = norm_dist(it, tt) < PINCH_TOUCH_THRESH
                mid_now = norm_dist(mt, tt) < PINCH_TOUCH_THRESH

                # Left click / Double click
                if idx_now and not index_pinch_prev:
                    index_pinch_start = now

                if (not idx_now) and index_pinch_prev:
                    if index_pinch_start > 0 and (now - index_pinch_start) < FAST_CLICK_TIME:

                        if (now - last_index_tap_time) < FAST_CLICK_TIME:
                            mouse.click(Button.left, 2)
                            set_hud("DOUBLE CLICK")
                        else:
                            mouse.click(Button.left, 1)
                            set_hud("LEFT CLICK")

                        last_index_tap_time = now

                    index_pinch_start = 0

                index_pinch_prev = idx_now

                # Right click
                if mid_now and not middle_pinch_prev:
                    middle_pinch_start = now

                if (not mid_now) and middle_pinch_prev:
                    if middle_pinch_start > 0 and (now - middle_pinch_start) < FAST_CLICK_TIME:
                        mouse.click(Button.right, 1)
                        set_hud("RIGHT CLICK")

                    middle_pinch_start = 0

                middle_pinch_prev = mid_now

        # ---------------- SCREENSHOT ----------------
        def fist(hand):
            try:
                thumb_tip = hand.landmark[ids.THUMB_TIP]
                index_mcp = hand.landmark[ids.INDEX_FINGER_MCP]
                thumb_close = norm_dist(thumb_tip, index_mcp) < 0.06
                curled = (
                    not finger_extended_strict(hand, ids.INDEX_FINGER_TIP, ids.INDEX_FINGER_MCP) and
                    not finger_extended_strict(hand, ids.MIDDLE_FINGER_TIP, ids.MIDDLE_FINGER_MCP) and
                    not finger_extended_strict(hand, ids.RING_FINGER_TIP,   ids.RING_FINGER_MCP) and
                    not finger_extended_strict(hand, ids.PINKY_TIP,        ids.PINKY_MCP)
                )
                return curled and thumb_close
            except:
                return False

        if n >= 2 and fist(hands[0]) and fist(hands[1]) and not arrow_mode:

            if screenshot_hold_start == 0:
                screenshot_hold_start = now

            hold = now - screenshot_hold_start
            pct = min(hold / 0.5, 1.0)
            bar = int(pct * 250)

            x0, y0 = 20, h - 50
            cv2.putText(frame, "Screenshot", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,255), 2)
            cv2.rectangle(frame, (x0, y0), (x0 + 250, y0 + 20),
                          (255,255,255), 2)
            cv2.rectangle(frame, (x0, y0), (x0 + bar, y0 + 20),
                          (0,255,255), -1)

            if pct >= 1.0:
                screenshot_count += 1
                fname = SCREENSHOT_PATTERN.format(screenshot_count)
                pyautogui.screenshot(fname)
                set_hud("SCREENSHOT")
                screenshot_hold_start = 0

        else:
            screenshot_hold_start = 0

        # ---------------- HUD DISPLAY ----------------
        if hud_enabled and hud_text and (now - hud_last_time < HUD_DURATION):
            cv2.putText(frame, hud_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255,255,255), 2)

        if arrow_mode:
            cv2.putText(frame, "ARROW MODE",(20, 65),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)


        if arrow_detected:
            bx, by = 20, 85
            bw, bh = 200, 15

            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                        (255,255,255), 2)
            cv2.rectangle(frame,
                        (bx, by),
                        (bx + int(bw * arrow_progress), by + bh),
                        (255,255,0), -1)

            cv2.putText(frame, "HOLD TO TOGGLE ARROW MODE",
                        (bx, by + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,0), 2)

        # ---------------- ARROW HUD FEEDBACK ----------------
        if arrow_mode:
            if left_arrow:
                cv2.putText(frame, "<- LEFT ARROW",
                            (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2)

            elif right_arrow:
                cv2.putText(frame, "-> RIGHT ARROW",
                            (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2)


        # ---------------- SHOW FRAME ----------------
        cv2.imshow("Gesture Controller", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("h"):
            hud_enabled = not hud_enabled

finally:
    cap.release()
    cv2.destroyAllWindows()
