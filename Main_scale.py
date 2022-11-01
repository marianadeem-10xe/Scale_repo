import cv2
import numpy as np
from utils_scale import Scale
from matplotlib import pyplot as plt


raw_path        = "./results/Graph images/ColorDifferenceError_20221026_131952.png"
size            = (1944, 2592, 3)          # (height, width)
scale_to_size   = (1440, 2560, 3)
output_filename = "./Downscaled_ColorDifferenceError_{}x{}.jpg".format(scale_to_size[0], scale_to_size[1]) 

raw_file = cv2.imread(raw_path)
raw_file = cv2.cvtColor(raw_file, cv2.COLOR_BGR2RGB)

raw_file = cv2.resize(raw_file, (2592, 1944), interpolation= cv2.INTER_LINEAR)
# plt.imsave("./Input images/Density2_2592x1944.png", raw_file)

assert raw_file.shape==size, "Input size must be 2592x1944!"

print("-"*50)
print("original size: ", raw_file.shape)

scaled_img = np.empty(scale_to_size, dtype="uint16")

for i in range(3):
    ch_arr = raw_file[:, :, i]
    print(ch_arr.shape)
    scale = Scale(ch_arr, scale_to_size)
    scaled_img[:, :, i] = scale.execute()

print("scaled size: ", scaled_img.shape)        
print("-"*50)

plt.imsave(output_filename, scaled_img.astype("uint8"))
print("image saved")

