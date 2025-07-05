import cv2
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
from torch.nn.utils.rnn import pad_sequence
import mediapipe as mp
from architecture.RNN import HandSignClassifier

input_size = 126
hidden_size = 256
num_classes = 50
model_path = 'checkpoints/2_layer_best_model_acc_50_classes.pth'
labels_path = 'idx_to_label_50_classes.npy'
max_window_size = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

idx_to_label = np.load(labels_path, allow_pickle=True).item()
model = HandSignClassifier(input_size=input_size,hidden_size=hidden_size, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

def flatten_landmarks(landmarks):
    return np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark]).flatten() if landmarks else np.zeros(63)

def normalize_sequence(seq):
    seq = np.array(seq).reshape(len(seq), -1, 3)
    x = seq[:, :, 0]
    y = seq[:, :, 1]
    x -= np.mean(x, axis=1, keepdims=True)
    y -= np.mean(y, axis=1, keepdims=True)
    scale = np.sqrt(np.var(x, axis=1, keepdims=True) + np.var(y, axis=1, keepdims=True)) + 1e-6
    seq[:, :, 0] = x / scale
    seq[:, :, 1] = y / scale
    return seq.reshape(len(seq), -1)

cap = cv2.VideoCapture(1)
sequence_buffer = deque(maxlen=max_window_size)
prediction = ""

print("Live sign prediction started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    left = flatten_landmarks(results.left_hand_landmarks)
    right = flatten_landmarks(results.right_hand_landmarks)
    landmarks = np.concatenate([left, right])
    sequence_buffer.append(landmarks)

    if len(sequence_buffer) >= 20: 
        raw_seq = list(sequence_buffer)
        norm_seq = normalize_sequence(raw_seq)
        seq_tensor = torch.tensor(norm_seq, dtype=torch.float32)

        padded_seq = pad_sequence([seq_tensor], batch_first=True) 
        lengths_tensor = torch.tensor([seq_tensor.size(0)], dtype=torch.long)

        with torch.no_grad():
            output = model(padded_seq.to(device), lengths_tensor.to(device))
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_idx].item()
            prediction = f"{idx_to_label[pred_idx]} ({conf:.2f})"

    cv2.putText(frame, f"Prediction: {prediction}", (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Running... (Press 'q' to quit)", (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()

