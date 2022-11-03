from wsgiref import util
import cv2
import numpy as np
import utils_scale as utils


file_path = "./results/Graph images/comparison_cv_upscaleNN_downscaleBilinear/cv2_results/cv2_ColorDifferenceError_1080x1920.png"
scale_to_size = (int(1080*2.7), int(1920*2.7), 3) 
result = utils.Results()

img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
cv2_scaled = cv2.resize(img, (scale_to_size[1], scale_to_size[0]), interpolation=cv2.INTER_LINEAR)

"""
# To compare cv2 upscaling (any size-->1944x2592) with the original img
org_path = "./results/Graph images/GIMP_1944x2592/ColorDifferenceError_2592x1944.png"
original_img  = cv2.cvtColor(cv2.imread(org_path), cv2.COLOR_BGR2RGB)
"""


# To compare with implemented algorithm
scaled_img = np.empty(scale_to_size, dtype="uint16")
for i in range(3):
    upscale = utils.UpScale(img[:,:,i], scale_to_size)
    scaled_img[:,:,i] = upscale.execute("")

eval = utils.Evaluation(cv2_scaled, scaled_img)

