from utils import Downscale, demosaic_raw, white_balance, gamma
from utils import scale_nearest_neighbor, BiLinear_Scale, optimal_reduction_factor
from utils_scale import Scale, UpScale, DownScale
import numpy as np
from matplotlib import pyplot as plt
import cv2
raw_path = "./text_img_scaled_1944x2592.raw"
size     = (1944, 2592)          # (height, width)
scale_to_size = (1080,1920)


raw_file = np.fromfile(raw_path, dtype="uint16").reshape(size)
print("-"*50)
print("original size: ", raw_file.shape)

scale = Scale(raw_file, scale_to_size)
scaled_img = scale.execute()

# scale = BiLinear_Scale(raw_file, scale_to_size)
# scaled_img = scale.bilinear_formula()

# scale = Downscale(raw_file, scale_to_size)
# scaled_img = scale.downscale_by_int_factor()

print("scaled size: ", scaled_img.shape)        
print("-"*50)
save_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGB)
# blc_corr = np.clip(np.float32(scaled_img)-200, 0, 4095).astype("uint16")
# save_img = gamma(demosaic_raw(white_balance(blc_corr.copy(), 320/256, 740/256, 256/256), "RGGB"))
plt.imsave("./NN_DownScale_text_(1920x1080).png", save_img.astype("uint8"))
print("image saved")

################################################################
# Compute reduction factor needed for minimum cropping.
# scale = Scale(raw_file, scale_to_size)
# min_crop_val, min_fact = optimal_reduction_factor(list(size), list(scale_to_size))
# print(min_crop_val, min_fact)

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



################################
# Resize downloaded image with cv2
"""img  = cv2.imread(raw_path)
print(img.shape)
# raw_file = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
scaled_img = cv2.resize(img, (2592, 1944), interpolation=cv2.INTER_LINEAR)
plt.imsave("./text_img_scaled_1944x2592.png", scaled_img)
with open("./text_img_scaled_1944x2592.raw", "wb") as file:
    cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY).astype("uint16").tofile(file)
print("file saved")
exit()"""     
################################