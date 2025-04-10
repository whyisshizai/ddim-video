import cv2
import os
import moviepy as mpy
import cv2
import os
import numpy as np
import re
import pyautogui
from PIL import ImageGrab
import threading
i = 0
file = r"D:\mizunashi akari\video\400"
for filename in os.listdir(file):
    if filename.endswith("mp4"):
        match = re.search(r'\d+', filename)  # 匹配一个或多个数字
        if match:
            number = int(match.group(0))  # 将匹配到的数字字符串转换为整数
            i = max(i, number)
cv2.namedWindow("tools")
x = 0
files = r"D:\mizunashi akari\video\400"
for filename in os.listdir(files):
    if filename.endswith("mp4"):
        x = 0
        f = f"{files}/{filename}"
        print(filename)
        # 注意RGBA格式
        video = mpy.VideoFileClip(f)
        frame_count = int(video.fps * video.duration)
        s = list(video.iter_frames())
        frames = []
        while x < len(s):
            fp = s[x]
            fp = cv2.resize(fp, (400,400), interpolation=cv2.INTER_AREA)
            cv2.imshow("tools", cv2.cvtColor(fp, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            OUTPUT_FILE = f"{file}/{i+1}.mp4"
            if key == ord("d"):
                x  = min(len(s)-1,x+1)
            if key == ord("a"):
                x  = max(0,x-1)
            if key == 13:
                frames.append(s[x])
                x+=1
            if key == 32:
                # 创建新的视频剪辑
                new_clip = mpy.ImageSequenceClip(frames, fps=4)
                new_clip.write_videofile(OUTPUT_FILE)
                i+=1
                print(f"New video saved as {OUTPUT_FILE}")
                video.close()
                os.remove(f)
                break
            if key == ord("q"):
                break
            if key == 27:  # ESC键退出程序
                cv2.destroyAllWindows()

                exit()


cv2.destroyAllWindows()