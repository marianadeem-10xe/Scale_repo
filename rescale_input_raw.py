import cv2
import os
import numpy as np

path = "/home/user3/infinite-isp_Internal/in_frames/normal/ColorCheckerRAW_1920x1080_12bit_BGGR.raw"
print(os.getcwd())
save_path = os.getcwd() + "/rescaled_ColorCheckerRAW_2592x1944_12bit_BGGR.raw"
raw_file = np.fromfile(path, dtype="uint16").reshape((1080, 1920))
print(raw_file.shape)
scaled_img = cv2.resize(raw_file, (2592, 1944), interpolation=cv2.INTER_CUBIC)

print(scaled_img.shape)
with open(save_path, "w") as file:
    scaled_img.tofile(file)
print("file_saved!")
