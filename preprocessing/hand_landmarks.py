import os
import cv2
import multiprocessing
import numpy as np
import mediapipe as mp
from tqdm import tqdm

def extract_and_save_landmarks_from_video(video_args):
    video_path, output_dir = video_args
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    os.makedirs(output_dir, exist_ok=True)

    """
    Initializing MediaPipe Holistic with only the hand module
    Saves time in long processings and the model does not have to be called at every processing
    Can allow customization of the model as well to tweak parameters
    """

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            frame_landmarks = []
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    frame_landmarks.extend([[lm.x, lm.y, lm.visibility] for lm in hand_landmarks.landmark])
                else:
                    frame_landmarks.extend([[0, 0, 0]] * 21)

            if frame_landmarks:
                np.save(os.path.join(output_dir, f"{frame_idx:04d}.npy"), np.array(frame_landmarks))
            frame_idx += 1

    cap.release()


def get_all_video_tasks(input_dir="augmented_videos", output_root="frames_with_landmarks", min_frames_threshold=10):
    tasks = []
    skipped = []

    for gloss in os.listdir(input_dir):
        gloss_path = os.path.join(input_dir, gloss)
        if not os.path.isdir(gloss_path):
            continue

        for video_file in os.listdir(gloss_path):
            if not video_file.endswith(".mp4"):
                continue

            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(gloss_path, video_file)
            output_dir = os.path.join(output_root, gloss, video_id)

            if os.path.exists(output_dir):
                saved_frames = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
                if len(saved_frames) >= min_frames_threshold:
                    skipped.append(video_path)
                    continue

            tasks.append((video_path, output_dir))

    print(f"Found {len(tasks)} videos to process.")
    print(f"Skipped {len(skipped)} already-processed videos.")
    return tasks


def process_all_videos(input_dir="augmented_videos", output_root="frames_with_landmarks", num_workers=None):
    tasks = get_all_video_tasks(input_dir, output_root)

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(extract_and_save_landmarks_from_video, tasks), total=len(tasks)))


if __name__ == "__main__":
    process_all_videos()