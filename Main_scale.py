from utils import Downscale, demosaic_raw, white_balance, gamma
from utils import scale_nearest_neighbor, BiLinear_Scale, optimal_reduction_factor
import numpy as np
from matplotlib import pyplot as plt

raw_path = "./HisiRAW_2592x1536_12bits_RGGB_Linear_20220407210640.raw"
size     = (1536, 2592)          # (height, width)
scale_to_size = (1080, 1920)

raw_file = np.fromfile(raw_path, dtype="uint16").reshape(size)

print("-"*50)
print("original size: ", raw_file.shape)

scaled_img = scale_nearest_neighbor(raw_file, scale_to_size)

# scale = BiLinear_Scale(raw_file, scale_to_size)
# scaled_img = scale.bilinear_formula()

# scale = Downscale(raw_file, scale_to_size)
# scaled_img = scale.downscale_by_int_factor()

print("scaled size: ", scaled_img.shape)        
print("-"*50)

blc_corr = np.clip(np.float32(scaled_img)-200, 0, 4095).astype("uint16")
save_img = gamma(demosaic_raw(white_balance(blc_corr.copy(), 320/256, 740/256, 256/256), "RGGB"))
plt.imsave("./NN_downscaled_(1080x1920)_myCode.png", save_img)
print("image saved")

################################################################
# Compute reduction factor needed for minimum cropping.
"""min_crop_val, min_fact = optimal_reduction_factor(list(size), list(scale_to_size))
print(min_crop_val, min_fact)"""

################################################################
# patch = np.array([[1,2,3,4],[5,6,7,8], [9,10,11,12], [13,14,15,16]])
# print(patch, patch.shape)
# sc_patch = scale_nearest_neighbor_v0(patch, 0.5)
# print(sc_patch.shape)
# print(sc_patch)

#################################################################
# patch = np.array([[1,2,3,4],[5,6,7,8], [9,10,11,12], [13,14,15,16],
#                   [1,2,3,4],[5,6,7,8], [9,10,11,12], [13,14,15,16]])
# print(patch, patch.shape)
# scale = Downscale(patch, (3,3))
# sc_patch = scale.downscale_by_int_factor()
# # formula_sc = scale.bilinear_formula()
# print(50*"-")
# print(sc_patch.shape)
# print(sc_patch)
# print(50*"-")
# # print(formula_sc)
# # print(50*"-")

# import cv2
# cv2_scaled = cv2.resize(raw_file, (864,512), interpolation= cv2.INTER_LINEAR)
# print("shape with cv2:\n", cv2_scaled.shape)
# blc_corr_cv2 = np.clip(np.float32(cv2_scaled)-200, 0, 4095).astype("uint16")
# save_img_cv2 = gamma(demosaic_raw(white_balance(blc_corr_cv2.copy(), 320/256, 740/256, 256/256), "RGGB"))
# plt.imsave("./DownScaled_864x512_3x_cv2.png", save_img_cv2)
# print("image saved")