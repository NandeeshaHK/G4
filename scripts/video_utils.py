import cv2
import os
from typing import List
from PIL import Image

def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 60) -> List[Image.Image]:
    """
    Extracts frames from a video file at a given sampling rate.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30.0 # fallback
        
    frame_interval = int(video_fps / fps) if video_fps >= fps else 1
    
    frames = []
    count = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
            
        count += 1
        
    cap.release()
    return frames

if __name__ == "__main__":
    # Test with a placeholder if needed
    print("Video utils loaded.")
