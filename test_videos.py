import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn.functional as F
from architecture.RNN import HandSignClassifier
import matplotlib.pyplot as plt

sequence_path = ""  #path to a recorded sequence with .npy extension
input_size = 126
hidden_size = 256
num_classes = 50
model_path = "checkpoints/2_layer_best_model_acc_50_classes.pth"
labels_path = "idx_to_label_50_classes.npy"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sequence = np.load(sequence_path)
idx_to_label = np.load(labels_path, allow_pickle=True).item()

model = HandSignClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    sequence_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)
    lengths = torch.tensor([len(sequence)], dtype=torch.long).to(device)
    output = model(sequence_tensor, lengths)
    probs = F.softmax(output, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    pred_label = idx_to_label[pred_idx]
    pred_conf = probs[pred_idx]

print(f" {pred_label} ({pred_conf:.2f})")

top5_idx = np.argsort(probs)[-5:][::-1]
top5_labels = [idx_to_label[i] for i in top5_idx]
top5_probs = probs[top5_idx]

plt.barh(top5_labels[::-1], top5_probs[::-1])
plt.title(f"Top-5 Prediction: {pred_label}")
plt.xlabel("Confidence")
plt.tight_layout()
plt.show()
