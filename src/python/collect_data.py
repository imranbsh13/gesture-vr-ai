import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Config ---
GESTURES = ["open_hand", "fist", "point_up", "peace", "thumbs_up"]
SAMPLES_PER_GESTURE = 200
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_landmarks(hand_landmarks):
    """Extract 21 hand landmarks as a flat array of 63 values (x, y, z each)."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks, dtype=np.float32)

def collect_gesture(gesture_name, num_samples):
    cap = cv2.VideoCapture(0)
    samples = []
    collected = 0

    print(f"\n--- Collecting: {gesture_name} ---")
    print("Get ready... starting in 3 seconds")
    time.sleep(3)

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        status = f"Collecting '{gesture_name}': {collected}/{num_samples}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            data = extract_landmarks(hand_lm)
            samples.append(data)
            collected += 1

        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(samples)

def main():
    for gesture in GESTURES:
        save_path = os.path.join(DATA_DIR, f"{gesture}.npy")
        if os.path.exists(save_path):
            print(f"Already collected {gesture}, skipping.")
            continue

        input(f"\nPress ENTER when ready to collect '{gesture}'...")
        samples = collect_gesture(gesture, SAMPLES_PER_GESTURE)

        np.save(save_path, samples)
        print(f"Saved {len(samples)} samples to {save_path}")

    print("\nAll gestures collected!")

if __name__ == "__main__":
    main()