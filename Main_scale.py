import os
import cv2
import numpy as np
from utils_scale import Scale, Evaluation, Results
from matplotlib import pyplot as plt


folder          = "./results/Graph images/rescaled with GIMP/2592x1944/"
GT_path          = "./results/Graph images/rescaled with GIMP/"

size            = (1944, 2592)          # (height, width)
# scale_to_size   = (1440, 2560)
is_hardware      = False
Algo             = ""
upscale_method   = "Nearest_Neighbor"
downscale_method = ""
result           = Results()

for scale_to_size in [(1440, 2560), (960, 1280), (480, 640), (1080,1920)]:
    for filename in os.listdir(folder):
        print("scaling img: ", filename)
        output_filename = "./results/Graph images/Scale algo/" + filename.split("_")[0] + "_{}x{}.jpg".format(scale_to_size[0], scale_to_size[1]) 
        raw_path = folder + "/" + filename
        
        # Read ground truth img (rescaled using GIMP software with bicubic method)
        GT_file  = cv2.imread(GT_path + "{}x{}/".format(scale_to_size[1], scale_to_size[0]) +filename.split("_")[0] + "_{}x{}.png".format(scale_to_size[1], scale_to_size[0]))
        GT_file  = cv2.cvtColor(GT_file, cv2.COLOR_BGR2RGB)
        
        # Read the input image
        raw_file = cv2.imread(raw_path)
        raw_file = cv2.cvtColor(raw_file, cv2.COLOR_BGR2RGB)

        cv2_scaled_img = cv2.resize(raw_file, (scale_to_size[1], scale_to_size[0]), interpolation= cv2.INTER_LINEAR)
        plt.imsave("./results/Graph images/cv2_results/cv2_{}_{}x{}.png".format(filename.split("_")[0], scale_to_size[0], scale_to_size[1]), cv2_scaled_img)
        
        assert raw_file.shape==(size[0], size[1],3), "Input size must be 2592x1944!"

        print("-"*50)
        print("original size: ", raw_file.shape)

        scaled_img = np.empty((scale_to_size[0], scale_to_size[1], 3), dtype="uint16")

        for i in range(3):
            ch_arr = raw_file[:, :, i]
            print(ch_arr.shape)
            scale = Scale(ch_arr, scale_to_size)
            scaled_img[:, :, i] = scale.execute(Algo, is_hardware,[upscale_method, downscale_method])

        print("scaled size: ", scaled_img.shape)        
        print("-"*50)
        cv2_info   = ["cv_"+filename, size, scale_to_size] + Evaluation(GT_file, cv2_scaled_img)
        scale_info = ["scale_algo_"+filename, size, scale_to_size] + Evaluation(GT_file, scaled_img)
        result.add_row(cv2_info)
        result.add_row(scale_info)

        plt.imsave(output_filename, scaled_img.astype("uint8"))
        print("image saved")
        exit()
result.save_csv("./results/Graph images", "Results")
print("results saved.")