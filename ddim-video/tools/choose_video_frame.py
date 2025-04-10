import cv2
import os
import moviepy as mpy
import cv2
import os
import numpy as np
from PIL import ImageGrab
files = r"D:\mizunashi akari\video\400"
h = 32
for filename in os.listdir(files):
    if filename.endswith("mp4"):
        f = f"{files}/{filename}"
        print(filename)
        # 注意RGBA格式
        video = mpy.VideoFileClip(f)
        s = list(video.iter_frames())
        frames = []
        for frame in s:
            frame = cv2.resize(frame, (h,h), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        OUTPUT_FILE = f"{f"D:\mizunashi akari\\video\\{h}"}/{filename}"
        new_clip = mpy.ImageSequenceClip(frames, fps=4)
        new_clip.write_videofile(OUTPUT_FILE)
        video.close()