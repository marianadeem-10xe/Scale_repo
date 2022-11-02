import os
import cv2
import numpy as np
from utils_scale import Scale, Evaluation, Results
from matplotlib import pyplot as plt


folder          = "./results/Graph images/GIMP_1944x2592"
size            = (1944, 2592, 3)          # (height, width)
# scale_to_size   = (1440, 2560, 3)
result          = Results()
for scale_to_size in [(1440, 2560, 3), (720, 1280, 3), (480, 640, 3), (1080,1920, 3)]:
    for filename in os.listdir(folder):
        print("scaling img: ", filename)
        output_filename = "./results/Graph images/Scale algo/" + filename.split("_")[0] + "_{}x{}.jpg".format(scale_to_size[0], scale_to_size[1]) 
        raw_path = folder + "/" + filename
        
        raw_file = cv2.imread(raw_path)
        raw_file = cv2.cvtColor(raw_file, cv2.COLOR_BGR2RGB)

        cv2_scaled_img = cv2.resize(raw_file, (scale_to_size[1], scale_to_size[0]), interpolation= cv2.INTER_LINEAR)
        plt.imsave("./results/Graph images/cv2_results/cv2_{}_{}x{}.png".format(filename.split("_")[0], scale_to_size[0], scale_to_size[1]), cv2_scaled_img)
        
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
        save_info = [filename, size, scale_to_size] + Evaluation(cv2_scaled_img, scaled_img)
        result.add_row(save_info)

        plt.imsave(output_filename, scaled_img.astype("uint8"))
        print("image saved")

result.save_csv("./results/Graph images", "Results.csv")
print("results saved.")