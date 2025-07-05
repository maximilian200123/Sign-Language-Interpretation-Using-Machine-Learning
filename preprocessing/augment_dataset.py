import os
import cv2
import numpy as np
import random
import shutil

def augment_video(input_path, output_path, aug_type, speed_factor=1.0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) * speed_factor
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_skip = int(1 / speed_factor) if speed_factor < 1.0 else 1
    idx = 0
    brightness_factor = random.uniform(1.1, 1.4)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip != 0:
            idx += 1
            continue

        if aug_type == "flip":
            frame = cv2.flip(frame, 1)
        elif aug_type == "brightness":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[..., 2] = np.clip(hsv[..., 2] * brightness_factor, 0, 255)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        out.write(frame)
        idx += 1

    cap.release()
    out.release()


def augment_all_videos_opencv(input_dir="processed_videos", output_dir="augmented_videos", max_total=40):
    os.makedirs(output_dir, exist_ok=True)
    augment_types = ["flip", "brightness", "speed_fast", "speed_slow"]

    for gloss in os.listdir(input_dir):
        gloss_path = os.path.join(input_dir, gloss)
        if not os.path.isdir(gloss_path):
            continue

        out_gloss_path = os.path.join(output_dir, gloss)
        os.makedirs(out_gloss_path, exist_ok=True)

        original_videos = [f for f in os.listdir(gloss_path) if f.endswith(".mp4") and "_aug_" not in f]
        for video_file in original_videos:
            src = os.path.join(gloss_path, video_file)
            dst = os.path.join(out_gloss_path, video_file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        print(f"\n[{gloss}] Copied {len(original_videos)} original videos.")

        aug_total = len([f for f in os.listdir(out_gloss_path) if f.endswith(".mp4")])
        augment_count = 0

        while aug_total < max_total:
            base_video = random.choice(original_videos)
            input_path = os.path.join(gloss_path, base_video)
            base_name = os.path.splitext(base_video)[0]

            aug_type = random.choice(augment_types)
            tag = f"{aug_type}_extra_{augment_count}"

            output_path = os.path.join(out_gloss_path, f"{base_name}_aug_{tag}.mp4")

            if os.path.exists(output_path):
                augment_count += 1
                continue

            if aug_type == "speed_fast":
                augment_video(input_path, output_path, "none", speed_factor=random.uniform(1.1, 1.4))
            elif aug_type == "speed_slow":
                augment_video(input_path, output_path, "none", speed_factor=random.uniform(0.7, 0.9))
            else:
                augment_video(input_path, output_path, aug_type)

            augment_count += 1
            aug_total += 1
            print(f"Created: {output_path}")

        print(f"[{gloss}] Final total: {aug_total} videos")

if __name__ == "__main__":
    augment_all_videos_opencv()
