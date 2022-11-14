import os
import cv2
import numpy as np
from utils_scale import Scale,DownScale,UpScale, Evaluation, Results
from matplotlib import pyplot as plt


folder           = "./results/Graph images/rescaled with GIMP/640x480/"
GT_path          = "./results/Graph images/rescaled with GIMP/3_scale_fact/"
size             = (480, 640, 3)          # (height, width)
# scale_to_size   = (1440, 2560, 3)
upscale_method   = ""
downscale_method = ""
result           = Results()

for scale_to_size in [(int(np.round(480*3)), int(np.round(640*3)),3)]:#,(1440, 2560, 3), (720, 1280, 3), (480, 640, 3), (1080,1920, 3)]:
    for filename in os.listdir(folder):
        print("scaling img: ", filename)
        output_filename = "./results/Graph images/Scale algo/" + filename.split("_")[0] + "_{}x{}.jpg".format(scale_to_size[0], scale_to_size[1]) 
        raw_path = folder + "/" + filename
        
        # Read ground truth img (rescaled using GIMP software with bicubic method)
        GT_file  = cv2.imread(GT_path +filename.split("_")[0] + "_{}x{}.png".format(scale_to_size[1], scale_to_size[0]))
        GT_file  = cv2.cvtColor(GT_file, cv2.COLOR_BGR2RGB)
        
        # Read the input image
        raw_file = cv2.imread(raw_path)
        raw_file = cv2.cvtColor(raw_file, cv2.COLOR_BGR2RGB)

        cv2_scaled_img = cv2.resize(raw_file, (scale_to_size[1], scale_to_size[0]), interpolation= cv2.INTER_LINEAR)
        plt.imsave("./results/Graph images/cv2_results/cv2_{}_{}x{}.png".format(filename.split("_")[0], scale_to_size[0], scale_to_size[1]), cv2_scaled_img)
        
        # assert raw_file.shape==size, "Input size must be 2592x1944!"

        print("-"*50)
        print("original size: ", raw_file.shape)

        scaled_img = np.empty(scale_to_size, dtype="uint16")

        for i in range(3):
            ch_arr = raw_file[:, :, i]
            print(ch_arr.shape)
            scale = UpScale(ch_arr, scale_to_size)
            scaled_img[:, :, i] = scale.execute("")#[upscale_method, downscale_method], False) #downscale_method) 

        print("scaled size: ", scaled_img.shape)        
        print("-"*50)
        cv2_info   = ["cv_"+filename, size, scale_to_size] + Evaluation(GT_file, cv2_scaled_img)
        scale_info = [filename, size, scale_to_size] + Evaluation(GT_file, scaled_img)
        result.add_row(cv2_info)
        result.add_row(scale_info)

        plt.imsave(output_filename, scaled_img.astype("uint8"))
        print("image saved")

result.save_csv("./results/Graph images", "Results")
print("results saved.")