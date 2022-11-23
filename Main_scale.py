from utils import Scale
import numpy as np
import yaml
import cv2
from matplotlib import pyplot as plt


raw_path = "./Raw input/Reschart_2592x1536.jpg"
config_path = './configs.yml'

#not to jumble any tags
yaml.preserve_quotes = True

with open(config_path, 'r') as f:
    c_yaml = yaml.safe_load(f)

# extract info
sensor_info = c_yaml['sensor_info']
parm_sca = c_yaml['scale']
parm_cro = c_yaml['crop']

size = (sensor_info["height"], sensor_info["width"])        # (height, width)
scale_to_size = (parm_sca["new_height"], parm_sca["new_width"])

img = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)

print("-"*50)
print("original size: ", img.shape)
scale = Scale(img, sensor_info, parm_sca)
scaled_img = scale.execute()
print("scaled size: ", scaled_img.shape)        
print("-"*50)
plt.imsave("./scaled_img.png", scaled_img)
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