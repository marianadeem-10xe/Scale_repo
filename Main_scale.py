from utils import demosaic_raw, white_balance, gamma
from utils import scale_nearest_neighbor_v0, BiLinear_Scale
import numpy as np
from matplotlib import pyplot as plt


raw_path = "./HisiRAW_2592x1536_12bits_RGGB_Linear_20220407210640.raw"
size     = (1536, 2592)          # (height, width)
scale_to_size = (1536*2,2592*2)

raw_file = np.fromfile(raw_path, dtype="uint16").reshape(size)

print("-"*50)
print("original size: ", raw_file.shape)
scale = BiLinear_Scale(raw_file, scale_to_size)
scaled_img = scale.scale_bilinear()
# scaled_img = scale_nearest_neighbor_v0(raw_file, 0.3)
print("scaled size: ", scaled_img.shape)        
print("-"*50)

blc_corr = np.clip(np.float32(scaled_img)-200, 0, 4095).astype("uint16")
save_img = gamma(demosaic_raw(white_balance(blc_corr.copy(), 320/256, 740/256, 256/256), "RGGB"))
plt.imsave("./Himite_Upscaled_2x_img.png", save_img)
print("image saved")

################################################################
# patch = np.array([[1,2,3,4],[5,6,7,8], [9,10,11,12], [13,14,15,16]])
# print(patch, patch.shape)
# sc_patch = scale_nearest_neighbor_v0(patch, 0.5)
# print(sc_patch.shape)
# print(sc_patch)

#################################################################
# patch = np.array([[10,20],[30,40]])
# print(patch, patch.shape)
# scale = BiLinear_Scale(patch, (4,4))
# sc_patch = scale.scale_bilinear()
# print(sc_patch.shape)
# print(sc_patch)

# import cv2
# cv2_scaled = cv2.resize(patch.astype("uint8"), (4,4), interpolation= cv2.INTER_LINEAR)
# print(cv2_scaled)