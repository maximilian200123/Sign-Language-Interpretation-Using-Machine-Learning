import os
import shutil
import random
import math

def split_dataset(root_dir="wlasl100", train_ratio=0.6667, val_ratio=0.1667, test_ratio=0.1666):    #4:1:1 split
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    for gloss in os.listdir(root_dir):
        gloss_path = os.path.join(root_dir, gloss)
        if not os.path.isdir(gloss_path):
            continue

        print(f"\nProcessing gloss: '{gloss}'")

        video_dirs = [d for d in os.listdir(gloss_path) 
                      if os.path.isdir(os.path.join(gloss_path, d)) 
                      and d not in ['train', 'val', 'test']]

        if len(video_dirs) < 3:
            print(f" Skipping '{gloss}' — too few samples.")
            continue

        random.shuffle(video_dirs)
        n = len(video_dirs)

        train_end = math.floor(train_ratio * n)
        val_end = train_end + math.floor(val_ratio * n)

        train_videos = video_dirs[:train_end]
        val_videos = video_dirs[train_end:val_end]
        test_videos = video_dirs[val_end:]

        for subset, video_list in zip(["train", "val", "test"], [train_videos, val_videos, test_videos]):
            subset_path = os.path.join(gloss_path, subset)
            os.makedirs(subset_path, exist_ok=True)

            for vid in video_list:
                src = os.path.join(gloss_path, vid)
                dst = os.path.join(subset_path, vid)
                if os.path.exists(dst):
                    print(f" Skipping move (already exists): {dst}")
                    continue
                shutil.move(src, dst)
            print(f"  {subset}/: {len(video_list)} videos")

if __name__ == "__main__":
    split_dataset()
