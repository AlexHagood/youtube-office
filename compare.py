 
import cv2
import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm 

def get_audio_mfcc(video_path, sr=22050):
    audio = VideoFileClip(video_path).audio
    y = audio.to_soundarray(fps=sr)[:, 0]  # mono
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

def get_video_color_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayscale_frame.flatten())
    cap.release()
    return np.array(frames)

def sliding_match(full_feat, clip_feat):
    # Compute the mean of the clip feature
    clip_mean = clip_feat.mean(axis=0)
    shape = clip_feat.shape[0]

    full_mean = full_feat.mean(axis=0)
    similarities = []
    for i in tqdm(range(len(full_feat) - shape), desc="Sliding match"):
        # Compute cosine similarity between the clip mean and the window mean
        sim = cosine_similarity([clip_mean], [full_mean[i:i + shape]])[0][0]
        similarities.append(sim)


    return similarities

full_vid = "pranks.mp4"
clip_vid = "clip69.mp4"
start_time = time.time()

def clk():
    return time.time() - start_time

print(f"Starting match process... (0.00s)")

# print(f"Extracing audio ({clk():.2f}s)")
# mfcc_full = get_audio_mfcc(full_vid)
# mfcc_clip = get_audio_mfcc(clip_vid)

print(f"Extracting video ({clk():.2f}s)")
video_full = get_video_color_sequence(full_vid)
video_clip = get_video_color_sequence(clip_vid)

print(f"Sliding matching ({clk():.2f}s)")
# audio_scores = sliding_match(mfcc_full.T, mfcc_clip.T)
video_scores = sliding_match(video_full, video_clip)

print(f"Idk ({clk():.2f}s)")
# best_audio_match = np.argmax(audio_scores[:len(video_scores)]) * (len(video_clip) / 23.976)  # ~time in seconds
best_video_match = np.argmax(video_scores) * (len(video_clip) / 23.976) / 3  # ~time in seconds

# 🔥 Example usage:
 
# print(f"Clip audio match appears at ~{best_audio_match:.2f} seconds")
print(f"Clip video match appears at ~{best_video_match:.2f} seconds")

