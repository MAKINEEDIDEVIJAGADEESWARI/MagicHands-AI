import os
import cv2
import mediapipe as mp
import pyautogui
import math
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 
# Optimize PyAutoGUI for real-time control
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
 
# ============================================================
# MAGIC HANDS AI - HAND DETECTION
# ============================================================
 
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    print("   Download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    exit()
 
PRIMARY_HAND = "Right"  # which hand controls the mouse; the other hand is ignored for actions
 
# Pinch hysteresis thresholds (enter/exit) to prevent click flicker
PINCH_ENTER_DIST = 0.045
PINCH_EXIT_DIST = 0.06
 
CLICK_COOLDOWN = 0.35  # minimum seconds between clicks, even across pinch cycles
 
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
 
cap = cv2.VideoCapture(0)
 
if not cap.isOpened():
    print("❌ Camera could not be opened!")
    exit()
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
# ============================================================
# HAND CONNECTIONS
# ============================================================
 
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (17, 13), (13, 9), (9, 5)
]
 
# ============================================================
# STATES
# ============================================================
 
pinch_is_active = False       # current debounced pinch state (hysteresis)
thumb_was_active = False
 
screen_width, screen_height = pyautogui.size()
 
smooth_x, smooth_y = screen_width // 2, screen_height // 2
scroll_start_y = None
 
last_action_time = 0     # thumbs-up cooldown
last_click_time = 0      # click cooldown
ACTION_COOLDOWN = 1.0
 
# ============================================================
# MAIN LOOP
# ============================================================
 
try:
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to capture frame")
            break
 
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)
 
        gesture = "NONE"
        number_of_hands = 0
 
        if result.hand_landmarks:
            number_of_hands = len(result.hand_landmarks)
 
            # Figure out which detected hand (if any) is the primary control hand.
            primary_index = None
            if result.handedness:
                for i, handedness in enumerate(result.handedness):
                    if handedness and handedness[0].category_name == PRIMARY_HAND:
                        primary_index = i
                        break
            if primary_index is None:
                # Fall back to first detected hand if the labeled primary hand isn't found
                primary_index = 0
 
            for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                is_primary = (hand_index == primary_index)
 
                # Landmarks
                index_tip = hand_landmarks[8]
                index_joint = hand_landmarks[6]
                thumb_tip = hand_landmarks[4]
                thumb_joint = hand_landmarks[3]
                middle_tip = hand_landmarks[12]
                ring_tip = hand_landmarks[16]
                little_tip = hand_landmarks[20]
 
                # Finger states
                index_up = index_tip.y < index_joint.y
                middle_up = middle_tip.y < hand_landmarks[10].y
                ring_down = ring_tip.y > hand_landmarks[14].y
                little_down = little_tip.y > hand_landmarks[18].y
                thumb_tucked = thumb_tip.x < hand_landmarks[5].x if PRIMARY_HAND == "Right" else thumb_tip.x > hand_landmarks[5].x
 
                # Gestures
                is_two_finger = index_up and middle_up and ring_down and little_down and thumb_tucked
                distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
 
                is_open_palm = (
                    index_tip.y < index_joint.y and
                    middle_tip.y < hand_landmarks[10].y and
                    ring_tip.y < hand_landmarks[14].y and
                    little_tip.y < hand_landmarks[18].y
                )
 
                is_thumb_up = (
                    thumb_tip.y < thumb_joint.y and
                    index_tip.y > index_joint.y and
                    middle_tip.y > hand_landmarks[10].y and
                    ring_tip.y > hand_landmarks[14].y and
                    little_tip.y > hand_landmarks[18].y
                )
 
                # Only the primary hand drives mouse/keyboard actions
                if is_primary:
                    # Pinch hysteresis: separate enter/exit thresholds prevent flicker
                    if pinch_is_active:
                        is_pinch = distance < PINCH_EXIT_DIST
                    else:
                        is_pinch = distance < PINCH_ENTER_DIST
 
                    # Decision Tree
                    if is_pinch:
                        gesture = "PINCH"
                        scroll_start_y = None
                        thumb_was_active = False
 
                        if not pinch_is_active:
                            current_time = time.time()
                            if current_time - last_click_time > CLICK_COOLDOWN:
                                pyautogui.click()
                                last_click_time = current_time
                            pinch_is_active = True
 
                    elif is_open_palm:
                        gesture = "OPEN_PALM"
                        pinch_is_active = False
                        thumb_was_active = False
                        scroll_start_y = None
 
                    elif is_thumb_up:
                        gesture = "THUMB_UP"
                        pinch_is_active = False
                        scroll_start_y = None
                        current_time = time.time()
 
                        if not thumb_was_active and (current_time - last_action_time > ACTION_COOLDOWN):
                            pyautogui.press("space")
                            last_action_time = current_time
                            thumb_was_active = True
 
                    elif is_two_finger:
                        gesture = "TWO_FINGER"
                        pinch_is_active = False
                        thumb_was_active = False
 
                        current_y = middle_tip.y
                        if scroll_start_y is None:
                            scroll_start_y = current_y
 
                        movement = scroll_start_y - current_y
                        
                        # Continuous scroll trigger based on step distance threshold
                        SCROLL_THRESHOLD = 0.02
                        if abs(movement) >= SCROLL_THRESHOLD:
                            scroll_amount = 120 if movement > 0 else -120
                            pyautogui.scroll(scroll_amount)
                            scroll_start_y = current_y  # Shift anchor point forward for seamless scrolling
 
                    elif index_up:
                        gesture = "INDEX_UP"
                        pinch_is_active = False
                        thumb_was_active = False
                        scroll_start_y = None
 
                        # Bound values to screen size safely
                        target_x = max(0, min(screen_width, int(index_tip.x * screen_width)))
                        target_y = max(0, min(screen_height, int(index_tip.y * screen_height)))
 
                        smooth_x = int(smooth_x + (target_x - smooth_x) * 0.35)
                        smooth_y = int(smooth_y + (target_y - smooth_y) * 0.35)
 
                        pyautogui.moveTo(smooth_x, smooth_y)
 
                    else:
                        gesture = "INDEX_DOWN"
                        pinch_is_active = False
                        thumb_was_active = False
                        scroll_start_y = None
 
                # Drawing (draw both hands regardless of primary status)
                points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
 
                line_color = (255, 255, 255) if is_primary else (140, 140, 140)
                dot_color = (255, 0, 255) if is_primary else (120, 120, 120)
 
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], line_color, 2)
 
                for x, y in points:
                    cv2.circle(frame, (x, y), 7, dot_color, -1)
                    cv2.circle(frame, (x, y), 9, (255, 255, 255), 2)
 
                if result.handedness and hand_index < len(result.handedness):
                    hand_label = result.handedness[hand_index][0].category_name
                    tag = f"{hand_label} Hand" + (" (control)" if is_primary else "")
                    cv2.putText(
                        frame, tag,
                        (points[0][0] - 60, points[0][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
 
            # Info Panel
            cv2.rectangle(frame, (20, 20), (350, 170), (20, 20, 20), -1)
            cv2.putText(frame, "MAGICHANDS AI", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Hand Detected", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Hands: {number_of_hands}", (40, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Gesture: {gesture}", (40, 135), cv2.FONT_HERSHEY_COMPLEX, 0.55, (255, 255, 255), 2)
 
        else:
            # No hand panel
            cv2.rectangle(frame, (20, 20), (350, 90), (20, 20, 20), -1)
            cv2.putText(frame, "MAGICHANDS AI", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Show your hand...", (40, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)
 
            pinch_is_active = False
            thumb_was_active = False
            scroll_start_y = None
 
        cv2.putText(frame, "Press Q to Exit", (width - 220, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("MagicHands AI", frame)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("MagicHands AI stopped.")