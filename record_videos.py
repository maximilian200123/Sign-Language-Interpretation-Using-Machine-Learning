import cv2
import numpy as np
import mediapipe as mp
import time

input_size = 126
record_key = 'r'
output_dir = 'recorded_sequences'

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def extract_landmarks(results):
    def flatten(landmarks):
        return np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark]).flatten() if landmarks else np.zeros(21 * 3)
    raw = np.concatenate([flatten(results.left_hand_landmarks), flatten(results.right_hand_landmarks)])
    return normalize_frame(raw)

def normalize_frame(flat):
    landmarks = flat.reshape(-1, 3)
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    scale = np.sqrt(np.var(x_centered) + np.var(y_centered)) + 1e-6
    landmarks[:, 0] = x_centered / scale
    landmarks[:, 1] = y_centered / scale
    return landmarks.flatten()

cap = cv2.VideoCapture(1)
recording = False
sequence = []
sequence_id = 0

print(f"Press '{record_key}' to start/stop recording. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    if results.left_hand_landmarks or results.right_hand_landmarks:
        landmarks = extract_landmarks(results)
    else:
        landmarks = np.zeros(input_size)

    key = cv2.waitKey(1) & 0xFF

    if recording:
        sequence.append(landmarks)
        cv2.putText(frame, f"Recording... Frames: {len(sequence)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if key == ord(record_key):
        recording = not recording
        if not recording and len(sequence) > 10:
            print(f"\n Stopping recording. Sequence length: {len(sequence)}")

            start = time.time()
            elapsed = time.time() - start
            np.save(f"{output_dir}/sequence_{sequence_id}.npy", np.array(sequence))
            print(f"[SAVED] Sequence saved to sequence_{sequence_id}.npy")
            sequence = []
            sequence_id += 1

    elif key == ord('q'):
        break

    cv2.putText(frame, f"Press '{record_key}' to toggle recording", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Sign Language Recorder", frame)

cap.release()
cv2.destroyAllWindows()
holistic.close()
