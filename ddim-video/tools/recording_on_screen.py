import cv2
import os
import numpy as np
import re
import pyautogui
from PIL import ImageGrab
import threading
# 配置参数 (初始默认区域)

h = 500

SCREEN_REGION = (0, 0, h, h)  # 默认区域
FPS = 24
scale_percent = 0.5  # 缩小到 50%
file = r"D:\mizunashi akari\video"

i = 0
for filename in os.listdir(file):
    if filename.endswith("mp4"):
        match = re.search(r'\d+', filename)  # 匹配一个或多个数字
        if match:
            number = int(match.group(0))  # 将匹配到的数字字符串转换为整数
            i = max(i, number)
print(f"数据库下标{i}")

selecting = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
region = SCREEN_REGION  # 初始为默认区域
region_modified = False  # 标志：区域是否被鼠标修改过
recording_started = False #标志：是否开始录制

def mouse_callback(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, selecting, region, region_modified
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        x_start, y_start = int(x / scale_percent) , int(y/scale_percent)
        x_end, y_end =  x_start+h,  y_start+h
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        # 计算并更新区域
        left = min(x_start, x_end)
        top = min(y_start, y_end)
        right = max(x_start, x_end)
        bottom = max(y_start, y_end)
        region = (left, top, right, bottom)
        region_modified = True  # 标记区域已被修改
    # elif event == cv2.EVENT_MBUTTONDOWN:
    #     try:
    #         global h
    #         h = int(input())
    #     except:
    #         print("h数值错误，请输入数字")
    #         pass

cv2.namedWindow("Select Region")
cv2.setMouseCallback("Select Region", mouse_callback)
FPS = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
color = (0,255,0)

while True:
    screen = np.array(ImageGrab.grab())
    img = screen.copy()
    #显示当前选区绿色
    cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), color, 2)
    if selecting: # 拖动过程中，实时更新
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2) # 拖动过程用蓝色

    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)

    # 调整图像大小
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Select Region", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)
    OUTPUT_FILE = f"{file}/{i + 1}.mp4"
    #enter
    if key == 13 and region_modified and not recording_started:
      recording_started = True
      video_writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (region[2] - region[0], region[3] - region[1]))
      print("开始录制...")
      while True:
        screen = np.array(ImageGrab.grab(bbox=region))
        cv2.imshow("Select Region", cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
        frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        img = cv2.resize( frame, (128,128), interpolation=cv2.INTER_AREA)
        video_writer.write(frame)
        key = cv2.waitKey(1)
        if key == 13 and recording_started:
          recording_started = False
          video_writer.release()
          break
      print(f"录制完成，保存到 {OUTPUT_FILE}")
      i+=1
    #esc
    if key == 27:  # ESC键退出
      cv2.destroyAllWindows()
      exit()


cv2.destroyAllWindows()