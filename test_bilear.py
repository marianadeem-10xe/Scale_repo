import os
import cv2
import numpy as np
import utils_scale as utils
import matplotlib.pyplot as plt

# Path variables
file_path = "./results/Graph images/rescaled with GIMP/1944x2592/ColorDifferenceError_2592x1944.png"
# GT_path   = "./results/Graph images/rescaled with GIMP/1944x2592/ColorDifferenceError_1944x2592.png"
filename  = os.path.basename(file_path)

# parameters
scale_to_size = (1296, 2304) 
is_hardware   = True
Algo = ""
upscale_method = ""
downscale_method = ""

result = utils.Results()

img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
cv2_scaled = cv2.resize(img, (scale_to_size[1], scale_to_size[0]), interpolation=cv2.INTER_LINEAR)

 # Read ground truth img (rescaled using GIMP software with bicubic method)
# GT_file  = cv2.imread(GT_path)
# GT_file  = cv2.cvtColor(GT_file, cv2.COLOR_BGR2RGB)

# To compare with implemented algorithm
scaled_img = np.empty((scale_to_size[0], scale_to_size[1],3), dtype="uint16")

for i in range(3):
    scale = utils.Scale(img[:,:,i], scale_to_size)
    scaled_img[:,:,i] = scale.execute(Algo, is_hardware, [upscale_method, downscale_method])

eval = utils.Evaluation(scaled_img, cv2_scaled)
# eval = utils.Evaluation(GT_file, cv2_scaled)
# eval = utils.Evaluation(GT_file, scaled_img)

output_filename = "./results/Graph images/" + filename.split("_")[0] + "_{}x{}.jpg".format(scale_to_size[0], scale_to_size[1]) 
# plt.imsave(output_filename, scaled_img.astype("uint8"))
print(scaled_img.shape)

##########################################################
# Test downscale_by_int_factor with convolution
"""scaled_img = np.empty((1944//3, 2592//3, 3), dtype="uint16")
scaled_img_conv = np.empty((1944//3, 2592//3, 3), dtype="uint16")

for i in range(3):
    scale = utils.Scale(img[:,:,i], (1944//3, 2592//3))
    scaled_img[:,:,i] = scale.downscale_by_int_factor((1944//3, 2592//3))
    scaled_img_conv[:,:,i] = scale.downscale_conv((1944//3, 2592//3))
    
eval = utils.Evaluation(scaled_img, scaled_img_conv)"""

##########################################################
# Test crop class
"""crop = utils.crop(img, scale_to_size)
cropped_img = crop.execute()
output_filename = "./results/Graph images/" + filename.split("_")[0] + "_{}x{}.jpg".format(scale_to_size[0], scale_to_size[1]) 
plt.imsave(output_filename, cropped_img.astype("uint8"))
"""
##########################################################
"""arr = np.array([[10,20], [30,40]])
cv2_scaled = cv2.resize(arr.astype("uint8"), (6,6),interpolation=cv2.INTER_LINEAR)
scale = utils.UpScale(arr, (6,6))
scaled = scale.bilinear_interpolation()
print(cv2_scaled)
print(scaled)
print(cv2_scaled-scaled)"""

